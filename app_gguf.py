"""
GGUF Forge - Automatic GGUF Model Conversion Service
Main application entry point.
"""
import os
import sys
import secrets
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyCookie
from passlib.context import CryptContext
from dotenv import load_dotenv

# --- Configuration & Setup ---
load_dotenv()

# Enable HuggingFace's accelerated upload backends by default. These must be set
# before huggingface_hub is imported anywhere in the app.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GGUF_Forge")


# Custom filter to suppress frequent polling endpoints from access logs
class EndpointFilter(logging.Filter):
    """Filter out frequent polling endpoints from uvicorn access logs."""
    def __init__(self, endpoints_to_skip: list):
        super().__init__()
        self.endpoints_to_skip = endpoints_to_skip
    
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for endpoint in self.endpoints_to_skip:
            if endpoint in message:
                return False
        return True


# Apply filter to uvicorn access logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(EndpointFilter([
    "/api/status/all",
    "/api/status/model/",
    "/api/requests/all",
    "/api/requests/my",
    "/api/tickets/all",
    "/api/tickets/my",
    "/api/tickets/"
]))

# Handle paths for PyInstaller (Frozen) vs Dev
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
    BUNDLE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent.absolute()
    BUNDLE_DIR = BASE_DIR

TEMPLATES_DIR = BUNDLE_DIR / "templates"
STATIC_DIR = BUNDLE_DIR / "static"

# Model download path - configurable via MODEL_DOWNLOAD_PATH or HF_HOME environment variable
# Priority: MODEL_DOWNLOAD_PATH > HF_HOME > default .cache folder
model_download_env = os.getenv("MODEL_DOWNLOAD_PATH", "")
hf_home_env = os.getenv("HF_HOME", "")
if model_download_env:
    # Use explicit MODEL_DOWNLOAD_PATH - support both relative and absolute paths
    model_path = Path(model_download_env)
    if model_path.is_absolute():
        CACHE_DIR = model_path
    else:
        CACHE_DIR = (BASE_DIR / model_path).resolve()
elif hf_home_env:
    # Use HuggingFace home directory
    CACHE_DIR = Path(hf_home_env) / "gguf-forge"
else:
    # Default: .cache subdirectory
    CACHE_DIR = BASE_DIR / ".cache"

# Llama.cpp directory - configurable via LLAMA_CPP_DIR environment variable
llama_cpp_env = os.getenv("LLAMA_CPP_DIR", "")
if llama_cpp_env:
    # Use environment variable - support both relative and absolute paths
    llama_cpp_path = Path(llama_cpp_env)
    if llama_cpp_path.is_absolute():
        LLAMA_CPP_DIR = llama_cpp_path
    else:
        # Relative path - resolve relative to BASE_DIR
        LLAMA_CPP_DIR = (BASE_DIR / llama_cpp_path).resolve()
else:
    # Default: llama.cpp subdirectory
    LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"

# Llama.cpp source repository - configurable via LLAMA_CPP_REPO environment variable
# Lets users point GGUF Forge at a fork (e.g. one with extra model support).
LLAMA_CPP_REPO = os.getenv("LLAMA_CPP_REPO", "").strip() or "https://github.com/ggml-org/llama.cpp"

# Fork-specific compact GGUF outtypes — comma-separated list (e.g. "iq2_xxs,q8_0").
# Used as the env-level fallback; the admin UI persists per-fork values in the DB.
LLAMA_CPP_OUTTYPES = [
    s.strip().lower() for s in os.getenv("LLAMA_CPP_OUTTYPES", "").split(",") if s.strip()
]

DB_PATH = BASE_DIR / "gguf_app.db"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Llama.cpp Constants - available quant types
QUANTS = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]
PARALLEL_QUANT_JOBS = int(os.getenv("PARALLEL_QUANT_JOBS", "1"))  # Default 1 for safer resource usage

# Server configuration
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", "8000"))

# Security
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
cookie_sec = APIKeyCookie(name="session_token", auto_error=False)

# HuggingFace OAuth Configuration
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET", "")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")

# --- Initialize Modules ---
from database import init_db, get_db_connection, set_db_path
from security import RateLimiter, BotDetector, SpamProtection
import managers
from managers import set_paths as set_manager_paths
from workflow import set_workflow_config, running_workflows, ModelQueue, set_model_queue, get_model_queue
from websocket_manager import manager as ws_manager

# Set paths for modules
set_db_path(DB_PATH)
set_manager_paths(BASE_DIR, LLAMA_CPP_DIR, LLAMA_CPP_REPO, LLAMA_CPP_OUTTYPES)
set_workflow_config(CACHE_DIR, LLAMA_CPP_DIR, QUANTS, PARALLEL_QUANT_JOBS)

# Initialize security instances
rate_limiter = RateLimiter(requests_per_minute=120, requests_per_second=15)
bot_detector = BotDetector()
spam_protection = SpamProtection(max_requests_per_hour=10, max_pending_per_user=5)


# --- User Authentication Helpers ---
async def get_current_user(request: Request):
    """Get current user - checks both admin users and OAuth users.
    
    Returns a dict-like row with additional 'is_oauth' and 'avatar_url' fields
    to avoid needing separate get_oauth_user calls.
    """
    token = request.cookies.get("session_token")
    if not token: 
        return None
    conn = await get_db_connection()
    # Check admin users first (legacy password-based admins)
    await conn.execute("SELECT *, 'admin' as user_type, 0 as is_oauth, NULL as avatar_url FROM users WHERE api_key = ?", (token,))
    row = await conn.fetchone()
    if row:
        await conn.close()
        return row
    # Check OAuth users - role is now stored in database
    await conn.execute("SELECT *, 'oauth' as user_type, 1 as is_oauth FROM oauth_users WHERE session_token = ?", (token,))
    oauth_user = await conn.fetchone()
    await conn.close()
    return oauth_user


async def get_oauth_user(request: Request):
    """Get OAuth user only (not admin).
    
    DEPRECATED: Use get_current_user() and check 'is_oauth' field instead.
    Kept for backwards compatibility.
    """
    user = await get_current_user(request)
    if user and user.get('is_oauth'):
        return user
    return None


async def require_admin(request: Request):
    user = await get_current_user(request)
    if not user or user['role'] != 'admin':
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# --- App Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Reload llama.cpp config now that the DB is available, so admin-UI overrides
    # set in a previous session take effect at startup.
    await managers.refresh_llama_config()
    logger.info(f"llama.cpp source: repo={managers.LLAMA_CPP_REPO} dir={managers.LLAMA_CPP_DIR}")

    # Initialize and start the model queue worker
    queue = ModelQueue()
    set_model_queue(queue)
    queue.start_worker()
    logger.info("Model queue system initialized")
    
    conn = await get_db_connection()
    
    # Startup cleanup: Check for stuck 'processing' jobs from crashed server
    processing_statuses = ['pending', 'initializing', 'downloading', 'converting', 'quantizing', 'uploading']
    await conn.execute(
        f"SELECT * FROM models WHERE status IN ({','.join(['?']*len(processing_statuses))})",
        tuple(processing_statuses)
    )
    stuck_jobs = await conn.fetchall()
    
    if stuck_jobs:
        logger.warning(f"Found {len(stuck_jobs)} stuck processing jobs from previous session")
        for job in stuck_jobs:
            model_id = job['id']
            hf_repo_id = job['hf_repo_id']
            old_status = job['status']
            
            # Update the model status to indicate it was interrupted
            await conn.execute(
                "UPDATE models SET status = ?, error_details = ? WHERE id = ?",
                ("interrupted", f"Server shutdown while status was '{old_status}'. Job can be restarted.", model_id)
            )
            
            # Check if there was an associated request that needs to be reset
            await conn.execute(
                "SELECT * FROM requests WHERE hf_repo_id = ? AND status = 'approved'",
                (hf_repo_id,)
            )
            existing_request = await conn.fetchone()
            
            if existing_request:
                await conn.execute(
                    "UPDATE requests SET status = 'pending' WHERE id = ?",
                    (existing_request['id'],)
                )
                logger.info(f"Reset request #{existing_request['id']} for {hf_repo_id} back to pending")
            
            logger.info(f"Marked stuck job {model_id} ({hf_repo_id}) as interrupted")
        
        await conn.commit()
        logger.info("Startup cleanup complete")
    
    # Create admin user if not exists
    await conn.execute("SELECT * FROM users WHERE role = 'admin'")
    admin = await conn.fetchone()
    if not admin:
        key = secrets.token_urlsafe(16)
        pwd = secrets.token_urlsafe(8)
        hashed = pwd_context.hash(pwd)
        await conn.execute("INSERT INTO users (username, hashed_password, role, api_key) VALUES (?, ?, ?, ?)",
                     ("admin", hashed, "admin", key))
        await conn.commit()
        
        creds_text = f"""
==================================================
ADMIN CREDENTIALS (GENERATED)
==================================================
Username: admin
Password: {pwd}
API Key: {key}
==================================================
"""
        print(creds_text)
        try:
            with open(BASE_DIR / "creds.txt", "w") as f:
                f.write(creds_text)
        except Exception as e:
            print(f"Failed to write creds.txt: {e}")
            
    await conn.close()

    # Background cleanup loop for in-memory rate/spam limiters (prevents memory bloat on bot traffic)
    async def _security_cleanup_loop():
        while True:
            try:
                await rate_limiter.cleanup()
                await spam_protection.cleanup()
            except Exception:
                logger.exception("Security cleanup loop error")
            await asyncio.sleep(60)

    cleanup_task = asyncio.create_task(_security_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except Exception:
            pass
        # Close database connection pool on shutdown
        from database import close_pool
        await close_pool()


# --- FastAPI App ---
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# --- Security Middleware ---
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Apply rate limiting and bot detection to all requests."""
    # Get client IP (handle proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    
    path = request.url.path
    
    # Skip security checks for static files
    if path.startswith("/static"):
        return await call_next(request)
    
    # Skip rate limiting for frequent polling endpoints
    # These are called 2-4 times per second for live updates
    polling_endpoints = [
        "/api/status/all",
        "/api/status/model/",  # Dynamic: /api/status/model/{id}
        "/api/requests/all",
        "/api/requests/my",
        "/api/tickets/all",
        "/api/tickets/my",
        "/api/tickets/",  # Dynamic: /api/tickets/{id}/messages
    ]
    skip_rate_limit = any(path == ep or path.startswith(ep) for ep in polling_endpoints)
    
    if not skip_rate_limit:
        allowed, reason = await rate_limiter.is_allowed(client_ip)
        if not allowed:
            logger.warning(f"Rate limit: {client_ip} - {path} - {reason}")
            return JSONResponse(
                status_code=429,
                content={"detail": reason}
            )
    
    # Bot detection for non-API routes
    user_agent = request.headers.get("User-Agent", "")
    is_bot, bot_reason = bot_detector.is_suspicious(user_agent, path)
    if is_bot and not path.startswith("/api/"):
        logger.warning(f"Bot detected: {client_ip} - {user_agent[:50]} - {bot_reason}")
        return JSONResponse(
            status_code=403,
            content={"detail": "Access denied"}
        )
    
    return await call_next(request)


# --- Configure and Include Routes ---
from routes import auth, models, requests, tickets, settings

# Configure route modules with dependencies
auth.configure(templates, pwd_context, OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, OAUTH_REDIRECT_URI)
models.configure(require_admin)
requests.configure(require_admin, get_current_user, spam_protection)
tickets.configure(require_admin, get_current_user)
settings.configure(require_admin)

# Include routers
app.include_router(auth.router)
app.include_router(models.router)
app.include_router(requests.router)
app.include_router(tickets.router)
app.include_router(settings.router)


# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    # --- Basic security for WebSockets (middleware doesn't run for WS) ---
    # Get client IP (handle proxies)
    forwarded_for = websocket.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = websocket.client.host if websocket.client else "unknown"

    user_agent = websocket.headers.get("user-agent", "")

    # Rate limit WS handshakes to reduce bot churn
    allowed, reason = await rate_limiter.is_allowed(client_ip)
    if not allowed:
        try:
            await websocket.close(code=1008, reason=reason)
        finally:
            return

    # Bot detection (treat WS like a browser route)
    is_bot, bot_reason = bot_detector.is_suspicious(user_agent, "/ws")
    if is_bot:
        logger.warning(f"Bot detected (ws): {client_ip} - {user_agent[:50]} - {bot_reason}")
        try:
            await websocket.close(code=1008, reason="Access denied")
        finally:
            return

    # Resolve user from session cookie (can't reuse Request-based dependency here)
    async def get_ws_user():
        token = websocket.cookies.get("session_token")
        if not token:
            return None
        conn = await get_db_connection()
        try:
            # Admin users (legacy)
            await conn.execute("SELECT *, 'admin' as user_type FROM users WHERE api_key = ?", (token,))
            row = await conn.fetchone()
            if row:
                return row
            # OAuth users
            await conn.execute("SELECT *, 'oauth' as user_type FROM oauth_users WHERE session_token = ?", (token,))
            return await conn.fetchone()
        finally:
            await conn.close()

    user = await get_ws_user()

    # Parse channels from query params
    requested_channels = websocket.query_params.getlist("channel")
    if not requested_channels:
        requested_channels = ["models"]  # Default to models channel

    # Restrict channels based on user role
    allowed_channels = {"models"}
    if user:
        allowed_channels.add("my_requests")
        if user.get("role") == "admin":
            allowed_channels.update({"requests", "tickets"})

    channels = [c for c in requested_channels if c in allowed_channels]
    if not channels:
        channels = ["models"]

    await ws_manager.connect(websocket, channels)
    try:
        while True:
            # Keep connection alive, handle incoming messages if needed
            data = await websocket.receive_text()
            # Client can send ping to keep alive
            if data == "ping":
                await websocket.send_text('{"type": "pong"}')
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)


# --- Main Routes ---
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = await get_current_user(request)
    # User now includes is_oauth and avatar_url fields - no need for separate query
    return templates.TemplateResponse(request, "index.html", {
        "user": user['username'] if user else None,
        "role": user['role'] if user else 'guest',
        "oauth_avatar": user.get('avatar_url') if user else None,
        "is_oauth": bool(user.get('is_oauth')) if user else False
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page for user preferences."""
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(request, "settings.html", {
        "user": user['username'] if user else None,
        "role": user['role'] if user else 'guest',
        "oauth_avatar": user.get('avatar_url') if user else None,
        "is_oauth": bool(user.get('is_oauth')) if user else False
    })


@app.get("/api/health")
async def health_check():
    """Health check endpoint with database status."""
    from database import test_connection, DB_TYPE
    
    db_ok, db_msg = await test_connection()
    
    return {
        "status": "healthy" if db_ok else "degraded",
        "database": {
            "type": DB_TYPE,
            "connected": db_ok,
            "message": db_msg
        },
        "version": "1.0"
    }


@app.get("/api/admin/db-info")
async def get_db_info(request: Request):
    """Admin only: Get database information."""
    user = await require_admin(request)
    from database import DB_TYPE, test_connection
    
    db_ok, db_msg = await test_connection()
    
    info = {
        "type": DB_TYPE,
        "connected": db_ok,
        "message": db_msg
    }
    
    if DB_TYPE == "sqlite":
        info["path"] = str(DB_PATH)
    elif DB_TYPE == "mssql":
        from database import MSSQL_HOST, MSSQL_PORT, MSSQL_DATABASE
        info["host"] = MSSQL_HOST
        info["port"] = MSSQL_PORT
        info["database"] = MSSQL_DATABASE
    
    return info


@app.get("/api/admin/check-update")
async def check_for_update(request: Request):
    """Admin only: Check if a git update is available."""
    user = await require_admin(request)
    
    try:
        # Fetch latest from origin (without merging)
        fetch_proc = await asyncio.create_subprocess_exec(
            "git", "fetch", "origin",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await fetch_proc.wait()
        
        # Get current branch
        branch_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "--abbrev-ref", "HEAD",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await branch_proc.communicate()
        current_branch = stdout.decode().strip()
        
        # Get local commit hash
        local_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await local_proc.communicate()
        local_commit = stdout.decode().strip()[:7]
        
        # Get remote commit hash
        remote_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", f"origin/{current_branch}",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await remote_proc.communicate()
        remote_commit = stdout.decode().strip()[:7]
        
        # Count commits behind
        behind_proc = await asyncio.create_subprocess_exec(
            "git", "rev-list", "--count", f"HEAD..origin/{current_branch}",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await behind_proc.communicate()
        commits_behind = int(stdout.decode().strip()) if stdout else 0
        
        # Get latest commit message from origin
        msg_proc = await asyncio.create_subprocess_exec(
            "git", "log", "-1", "--format=%s", f"origin/{current_branch}",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await msg_proc.communicate()
        latest_message = stdout.decode().strip()[:100]
        
        return {
            "update_available": commits_behind > 0,
            "commits_behind": commits_behind,
            "local_commit": local_commit,
            "remote_commit": remote_commit,
            "branch": current_branch,
            "latest_message": latest_message if commits_behind > 0 else None
        }
    except Exception as e:
        logger.exception("Failed to check for updates")
        return {
            "update_available": False,
            "error": str(e)
        }


@app.post("/api/admin/update-app")
async def update_app(request: Request):
    """Admin only: Pull latest updates and restart the server."""
    user = await require_admin(request)
    
    try:
        # Get current branch
        branch_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "--abbrev-ref", "HEAD",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await branch_proc.communicate()
        current_branch = stdout.decode().strip()
        
        # Fetch and reset to origin (same as update scripts)
        fetch_proc = await asyncio.create_subprocess_exec(
            "git", "fetch", "--all",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await fetch_proc.wait()
        
        reset_proc = await asyncio.create_subprocess_exec(
            "git", "reset", "--hard", f"origin/{current_branch}",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await reset_proc.communicate()
        
        if reset_proc.returncode != 0:
            return {"status": "error", "message": f"Git reset failed: {stderr.decode()}"}
        
        # Schedule server restart after response is sent
        async def restart_server():
            await asyncio.sleep(1)  # Give time for response to be sent
            logger.info("Restarting server after update...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        asyncio.create_task(restart_server())
        
        return {
            "status": "success", 
            "message": "Update complete. Server restarting...",
            "branch": current_branch
        }
    except Exception as e:
        logger.exception("Failed to update app")
        return {"status": "error", "message": str(e)}


@app.get("/api/quants")
async def get_available_quants():
    """Get list of available quantization types."""
    return {
        "quants": QUANTS,
        "descriptions": {
            "Q2_K": "2-bit quantization (smallest, lowest quality)",
            "Q3_K_S": "3-bit small quantization",
            "Q3_K_M": "3-bit medium quantization",
            "Q3_K_L": "3-bit large quantization",
            "Q4_0": "4-bit legacy quantization",
            "Q4_K_S": "4-bit small quantization (recommended for low memory)",
            "Q4_K_M": "4-bit medium quantization (good balance)",
            "Q5_0": "5-bit legacy quantization",
            "Q5_K_S": "5-bit small quantization",
            "Q5_K_M": "5-bit medium quantization (good quality)",
            "Q6_K": "6-bit quantization (high quality)",
            "Q8_0": "8-bit quantization (highest quality, largest size)"
        }
    }


@app.get("/api/dashboard/init")
async def dashboard_init(request: Request):
    """Consolidated endpoint for initial dashboard data.
    
    Returns all data needed to initialize the dashboard in a single request,
    reducing initial page load from 4 HTTP requests to 1.
    """
    user = await get_current_user(request)
    is_admin = user and user.get('role') == 'admin'
    
    conn = await get_db_connection()
    
    # Always get models (public)
    await conn.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT 50")
    models = await conn.fetchall()
    
    result = {
        "models": [m.to_dict() for m in models],
        "requests": [],
        "tickets": [],
        "my_requests": []
    }
    
    if is_admin:
        # Admin gets pending requests and open tickets
        await conn.execute("SELECT * FROM requests WHERE status = 'pending' ORDER BY created_at DESC")
        requests = await conn.fetchall()
        result["requests"] = [r.to_dict() for r in requests]
        
        await conn.execute("""
            SELECT t.*, r.hf_repo_id, r.requested_by 
            FROM tickets t 
            JOIN requests r ON t.request_id = r.id 
            WHERE t.status = 'open'
            ORDER BY t.created_at DESC
        """)
        tickets = await conn.fetchall()
        result["tickets"] = [t.to_dict() for t in tickets]
    elif user:
        # Regular user gets their own requests
        await conn.execute(
            "SELECT * FROM requests WHERE requested_by = ? ORDER BY created_at DESC",
            (user['username'],)
        )
        my_requests = await conn.fetchall()
        result["my_requests"] = [r.to_dict() for r in my_requests]
    
    await conn.close()
    return result


if __name__ == "__main__":
    import uvicorn
    print(f"Starting GGUF Forge on {SERVER_HOST}:{SERVER_PORT}...")
    uvicorn.run("app_gguf:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
