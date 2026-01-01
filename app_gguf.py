"""
GGUF Forge - Automatic GGUF Model Conversion Service
Main application entry point.
"""
import os
import sys
import secrets
import logging
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
CACHE_DIR = BASE_DIR / ".cache"
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"
DB_PATH = BASE_DIR / "gguf_app.db"

CACHE_DIR.mkdir(exist_ok=True)

# Llama.cpp Constants
QUANTS = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]
PARALLEL_QUANT_JOBS = int(os.getenv("PARALLEL_QUANT_JOBS", "2"))

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
from managers import set_paths as set_manager_paths
from workflow import set_workflow_config, running_workflows
from websocket_manager import manager as ws_manager

# Set paths for modules
set_db_path(DB_PATH)
set_manager_paths(BASE_DIR, LLAMA_CPP_DIR)
set_workflow_config(CACHE_DIR, LLAMA_CPP_DIR, QUANTS, PARALLEL_QUANT_JOBS)

# Initialize security instances
rate_limiter = RateLimiter(requests_per_minute=120, requests_per_second=15)
bot_detector = BotDetector()
spam_protection = SpamProtection(max_requests_per_hour=10, max_pending_per_user=5)


# --- User Authentication Helpers ---
async def get_current_user(request: Request):
    """Get current user - checks both admin users and OAuth users."""
    token = request.cookies.get("session_token")
    if not token: 
        return None
    conn = await get_db_connection()
    # Check admin users first (legacy password-based admins)
    user = (await conn.execute("SELECT *, 'admin' as user_type FROM users WHERE api_key = ?", (token,))).cursor
    row = await conn.fetchone()
    if row:
        await conn.close()
        return row
    # Check OAuth users - role is now stored in database
    await conn.execute("SELECT *, 'oauth' as user_type FROM oauth_users WHERE session_token = ?", (token,))
    oauth_user = await conn.fetchone()
    await conn.close()
    return oauth_user


async def get_oauth_user(request: Request):
    """Get OAuth user only (not admin)."""
    token = request.cookies.get("session_token")
    if not token: 
        return None
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM oauth_users WHERE session_token = ?", (token,))
    oauth_user = await conn.fetchone()
    await conn.close()
    return oauth_user


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
    yield


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
from routes import auth, models, requests, tickets

# Configure route modules with dependencies
auth.configure(templates, pwd_context, OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, OAUTH_REDIRECT_URI)
models.configure(require_admin)
requests.configure(require_admin, get_current_user, spam_protection)
tickets.configure(require_admin, get_current_user)

# Include routers
app.include_router(auth.router)
app.include_router(models.router)
app.include_router(requests.router)
app.include_router(tickets.router)


# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    # Parse channels from query params
    channels = websocket.query_params.getlist("channel")
    if not channels:
        channels = ["models"]  # Default to models channel
    
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
    oauth_user = await get_oauth_user(request)
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "user": user['username'] if user else None,
        "role": user['role'] if user else 'guest',
        "oauth_avatar": oauth_user['avatar_url'] if oauth_user else None,
        "is_oauth": oauth_user is not None
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


if __name__ == "__main__":
    import uvicorn
    print("Starting GGUF Forge...")
    uvicorn.run("app_gguf:app", host="0.0.0.0", port=8000, reload=False)
