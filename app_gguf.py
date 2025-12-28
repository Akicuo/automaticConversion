
import os
import sys
import shutil
import subprocess
import asyncio
import logging
import secrets
import platform
import json
import uuid
import sqlite3
import traceback
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import httpx
from urllib.parse import urlencode

from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import APIKeyCookie
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download, create_repo
from passlib.context import CryptContext

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
    "/api/requests/all",
    "/api/requests/my"
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

# Security
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
cookie_sec = APIKeyCookie(name="session_token", auto_error=False)

# HuggingFace OAuth Configuration
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET", "")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")
OAUTH_AUTHORIZE_URL = "https://huggingface.co/oauth/authorize"
OAUTH_TOKEN_URL = "https://huggingface.co/oauth/token"
OAUTH_USERINFO_URL = "https://huggingface.co/oauth/userinfo"

# --- Security: Rate Limiting, Bot Detection, Anti-Spam ---

class RateLimiter:
    """IP-based rate limiter to prevent DDoS attacks."""
    def __init__(self, requests_per_minute: int = 60, requests_per_second: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.requests: dict = {}  # IP -> list of timestamps
        self.blocked_ips: dict = {}  # IP -> block_until timestamp
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, ip: str) -> tuple[bool, str]:
        """Check if request from IP is allowed. Returns (allowed, reason)."""
        import time
        now = time.time()
        
        async with self.lock:
            # Check if IP is blocked
            if ip in self.blocked_ips:
                if now < self.blocked_ips[ip]:
                    remaining = int(self.blocked_ips[ip] - now)
                    return False, f"IP blocked for {remaining}s due to rate limit abuse"
                else:
                    del self.blocked_ips[ip]
            
            # Initialize or clean old requests
            if ip not in self.requests:
                self.requests[ip] = []
            
            # Remove requests older than 1 minute
            self.requests[ip] = [t for t in self.requests[ip] if now - t < 60]
            
            # Check per-second rate
            recent_second = sum(1 for t in self.requests[ip] if now - t < 1)
            if recent_second >= self.requests_per_second:
                return False, "Too many requests per second"
            
            # Check per-minute rate
            if len(self.requests[ip]) >= self.requests_per_minute:
                # Block IP for 5 minutes
                self.blocked_ips[ip] = now + 300
                logger.warning(f"Rate limit exceeded, blocking IP: {ip}")
                return False, "Rate limit exceeded. IP blocked for 5 minutes"
            
            # Allow request
            self.requests[ip].append(now)
            return True, ""
    
    async def cleanup(self):
        """Remove old entries to prevent memory bloat."""
        import time
        now = time.time()
        async with self.lock:
            # Clean request history
            for ip in list(self.requests.keys()):
                self.requests[ip] = [t for t in self.requests[ip] if now - t < 60]
                if not self.requests[ip]:
                    del self.requests[ip]
            # Clean expired blocks
            for ip in list(self.blocked_ips.keys()):
                if now >= self.blocked_ips[ip]:
                    del self.blocked_ips[ip]


class BotDetector:
    """Detect and block suspicious bot traffic."""
    
    # Known bot/suspicious user agent patterns
    SUSPICIOUS_PATTERNS = [
        "curl", "wget", "python-requests", "scrapy", "bot", "spider",
        "crawler", "scan", "http", "java/", "perl", "ruby", "go-http",
        "aiohttp", "httpx", "axios", "node-fetch", "undici"
    ]
    
    # Legitimate browser patterns
    BROWSER_PATTERNS = ["mozilla", "chrome", "safari", "firefox", "edge", "opera"]
    
    @classmethod
    def is_suspicious(cls, user_agent: str, path: str) -> tuple[bool, str]:
        """Check if request appears to be from a bot. Returns (is_bot, reason)."""
        if not user_agent:
            return True, "Missing User-Agent header"
        
        ua_lower = user_agent.lower()
        
        # Skip check for API endpoints that might legitimately use non-browser clients
        if path.startswith("/api/"):
            return False, ""
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if pattern in ua_lower:
                return True, f"Suspicious User-Agent pattern: {pattern}"
        
        # For non-API routes, require browser-like user agent
        has_browser = any(pattern in ua_lower for pattern in cls.BROWSER_PATTERNS)
        if not has_browser and not path.startswith("/api/"):
            return True, "Non-browser User-Agent for browser route"
        
        return False, ""


class SpamProtection:
    """Prevent spam submissions (model requests, etc.)."""
    
    def __init__(self, max_requests_per_hour: int = 10, max_pending_per_user: int = 5):
        self.max_requests_per_hour = max_requests_per_hour
        self.max_pending_per_user = max_pending_per_user
        self.submissions: dict = {}  # user -> list of timestamps
        self.lock = asyncio.Lock()
    
    async def can_submit(self, username: str) -> tuple[bool, str]:
        """Check if user can submit a new request."""
        import time
        now = time.time()
        
        async with self.lock:
            if username not in self.submissions:
                self.submissions[username] = []
            
            # Remove submissions older than 1 hour
            self.submissions[username] = [t for t in self.submissions[username] if now - t < 3600]
            
            if len(self.submissions[username]) >= self.max_requests_per_hour:
                return False, f"Request limit reached ({self.max_requests_per_hour}/hour). Please try again later."
            
            return True, ""
    
    async def record_submission(self, username: str):
        """Record a new submission."""
        import time
        async with self.lock:
            if username not in self.submissions:
                self.submissions[username] = []
            self.submissions[username].append(time.time())
    
    def check_pending_limit(self, username: str) -> tuple[bool, str]:
        """Check if user has too many pending requests (sync DB check)."""
        conn = get_db_connection()
        count = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE requested_by = ? AND status = 'pending'",
            (username,)
        ).fetchone()[0]
        conn.close()
        
        if count >= self.max_pending_per_user:
            return False, f"Too many pending requests ({count}/{self.max_pending_per_user}). Wait for approval."
        return True, ""


# Initialize security instances
rate_limiter = RateLimiter(requests_per_minute=120, requests_per_second=15)
bot_detector = BotDetector()
spam_protection = SpamProtection(max_requests_per_hour=10, max_pending_per_user=5)


# Database Setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL,
            api_key TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            hf_repo_id TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER DEFAULT 0,
            log TEXT DEFAULT '',
            error_details TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hf_repo_id TEXT NOT NULL,
            requested_by TEXT,
            status TEXT DEFAULT 'pending',
            decline_reason TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS oauth_users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT,
            avatar_url TEXT,
            session_token TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Migration: Add decline_reason column if it doesn't exist
    try:
        c.execute("ALTER TABLE requests ADD COLUMN decline_reason TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Core Logic Managers ---

class LlamaCppManager:
    @staticmethod
    def is_installed() -> bool:
        return (LLAMA_CPP_DIR / "Makefile").exists() or (LLAMA_CPP_DIR / "CMakeLists.txt").exists()

    @staticmethod
    async def clone_repo():
        if LlamaCppManager.is_installed():
            logger.info("llama.cpp already exists. Pulling latest...")
            proc = await asyncio.create_subprocess_exec(
                "git", "pull",
                cwd=LLAMA_CPP_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.wait()
        else:
            logger.info("Cloning llama.cpp...")
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            returncode = await proc.wait()
            if returncode != 0:
                raise Exception("Failed to clone llama.cpp")

    @staticmethod
    async def build():
        logger.info("Building llama.cpp...")
        system = platform.system()
        
        try:
            if system == "Windows":
                build_dir = LLAMA_CPP_DIR / "build"
                build_dir.mkdir(exist_ok=True)
                proc = await asyncio.create_subprocess_exec(
                    "cmake", "..", "-DLLAMA_CURL=OFF",
                    cwd=build_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                returncode = await proc.wait()
                if returncode != 0:
                    raise Exception("CMake configure failed")
                
                proc = await asyncio.create_subprocess_exec(
                    "cmake", "--build", ".", "--config", "Release",
                    cwd=build_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                returncode = await proc.wait()
                if returncode != 0:
                    raise Exception("CMake build failed")
                logger.info("Built with CMake on Windows.")
            else:
                proc = await asyncio.create_subprocess_exec(
                    "make", "LLAMA_CURL=0",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                returncode = await proc.wait()
                if returncode != 0:
                    raise Exception("Make build failed")
                logger.info("Built with Make on Linux/Mac.")
        except Exception as e:
            logger.error(f"Build failed: {e}")
            raise Exception("Failed to build llama.cpp. Ensure build tools (CMake/Visual Studio/Make) are installed.")

    @staticmethod
    def get_quantize_path() -> Path:
        system = platform.system()
        if system == "Windows":
            paths = [
                LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",
                LLAMA_CPP_DIR / "build" / "Release" / "llama-quantize.exe",
                LLAMA_CPP_DIR / "llama-quantize.exe"
            ]
            for p in paths:
                if p.exists(): return p
        else:
            if (LLAMA_CPP_DIR / "llama-quantize").exists():
                return LLAMA_CPP_DIR / "llama-quantize"
        
        found = list(LLAMA_CPP_DIR.rglob("llama-quantize*"))
        if found: return found[0]
        
        raise FileNotFoundError("llama-quantize executable not found. Build might have failed.")

class HuggingFaceManager:
    def __init__(self, token: Optional[str] = None):
        self.api = HfApi(token=token)

    def search_models(self, query: str, limit: int = 10):
        models = self.api.list_models(search=query, limit=limit, sort="likes", direction=-1)
        return [{"id": m.modelId, "likes": m.likes} for m in models]

    def check_exists(self, repo_id: str) -> bool:
        try:
            self.api.model_info(repo_id)
            return True
        except:
            return False

def get_app_version() -> str:
    """Calculate app version based on git commit count * 0.1"""
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            commit_count = int(result.stdout.strip())
            version = commit_count * 0.1
            return f"v{version:.1f}"
    except Exception:
        pass
    return "v0.1"  # Fallback

class ModelWorkflow:
    def __init__(self, model_id: str, hf_repo_id: str):
        self.model_id = model_id
        self.hf_repo_id = hf_repo_id
        self.log_buffer = []
        self.model_dir = None
        self.fp16_path = None
        self.quant_paths = []
        # Time tracking
        self.start_time = None
        self.step_times = {}  # step_name -> (start, end)
        self.quant_times = []  # list of (q_type, duration_seconds)

    def log(self, message: str):
        print(f"[{self.hf_repo_id}] {message}")
        self.log_buffer.append(message)
        # Keep last 8k chars for better visibility in UI
        self._update_db(log="\n".join(self.log_buffer)[-8000:])

    def progress(self, percent: int):
        self._update_db(progress=percent)

    def status(self, status_msg: str):
        self._update_db(status=status_msg)

    def _update_db(self, **kwargs):
        conn = get_db_connection()
        try:
            updates = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [self.model_id]
            conn.execute(f"UPDATE models SET {updates} WHERE id = ?", values)
            conn.commit()
        finally:
            conn.close()

    def start_step(self, step_name: str):
        """Start timing a step."""
        import time
        self.step_times[step_name] = {"start": time.time(), "end": None}
    
    def end_step(self, step_name: str):
        """End timing a step."""
        import time
        if step_name in self.step_times:
            self.step_times[step_name]["end"] = time.time()
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = seconds / 60
            return f"{mins:.1f}min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_timing_summary(self) -> dict:
        """Get timing summary for the job."""
        import time
        summary = {
            "total_time": 0,
            "avg_quant_time": 0,
            "step_times": {}
        }
        
        if self.start_time:
            summary["total_time"] = time.time() - self.start_time
        
        for step, times in self.step_times.items():
            if times["start"] and times["end"]:
                duration = times["end"] - times["start"]
                summary["step_times"][step] = duration
        
        if self.quant_times:
            avg_time = sum(t for _, t in self.quant_times) / len(self.quant_times)
            summary["avg_quant_time"] = avg_time
        
        return summary

    def check_disk_space(self, required_gb: float):
        total, used, free = shutil.disk_usage(BASE_DIR)
        free_gb = free / (2**30)
        self.log(f"  Disk space check: Need {required_gb:.1f}GB, Available {free_gb:.1f}GB")
        if free_gb < required_gb:
            raise Exception(f"Insufficient disk space. Required: {required_gb:.1f}GB, Available: {free_gb:.1f}GB")
        self.log(f"  ✓ Sufficient disk space")

    def get_model_size_gb(self) -> float:
        """Get model size from HuggingFace API in GB."""
        try:
            hf_token = os.getenv("HF_TOKEN")
            api = HfApi(token=hf_token)
            model_info = api.model_info(self.hf_repo_id, files_metadata=True)
            
            total_bytes = 0
            if model_info.siblings:
                for sibling in model_info.siblings:
                    if hasattr(sibling, 'size') and sibling.size:
                        total_bytes += sibling.size
            
            size_gb = total_bytes / (2**30)
            return size_gb
        except Exception as e:
            self.log(f"  ⚠ Could not fetch model size: {e}")
            return 10.0  # Default fallback

    def cleanup(self):
        """Remove all downloaded and generated files."""
        self.log("Starting cleanup...")
        try:
            # Remove downloaded model directory
            if self.model_dir and Path(self.model_dir).exists():
                self.log(f"Removing downloaded model: {self.model_dir}")
                shutil.rmtree(self.model_dir, ignore_errors=True)
            
            # Remove FP16 file
            if self.fp16_path and self.fp16_path.exists():
                self.log(f"Removing FP16 file: {self.fp16_path}")
                self.fp16_path.unlink(missing_ok=True)
            
            # Remove all quantized files
            for q_path in self.quant_paths:
                if q_path.exists():
                    self.log(f"Removing quant file: {q_path}")
                    q_path.unlink(missing_ok=True)
            
            self.log("Cleanup completed.")
        except Exception as e:
            self.log(f"Cleanup error (non-fatal): {e}")

    async def run_pipeline(self):
        import time
        error_details = ""
        new_repo_id = None
        try:
            self.start_time = time.time()
            self.status("initializing")
            self.progress(0)
            self.log("━━━ GGUF Forge Pipeline Started ━━━")
            self.log(f"Job ID: {self.model_id}")
            self.log(f"Model: {self.hf_repo_id}")
            self.log(f"Version: {get_app_version()}")
            self.log("")
            
            # 1. Setup Llama
            self.start_step("setup")
            self.log("▶ STEP 1: Setting up llama.cpp...")
            self.log("  Checking llama.cpp installation...")
            await LlamaCppManager.clone_repo()
            self.log("  Building llama.cpp (this may take a while)...")
            await LlamaCppManager.build()
            quantize_bin = LlamaCppManager.get_quantize_path()
            self.log(f"  ✓ llama-quantize ready: {quantize_bin.name}")
            self.end_step("setup")
            self.progress(10)
            self.log("")

            # 2. Download
            self.start_step("download")
            self.status("downloading")
            self.log("▶ STEP 2: Downloading model from HuggingFace...")
            self.log(f"  Source: https://huggingface.co/{self.hf_repo_id}")
            
            # Get actual model size and calculate required space
            model_size_gb = self.get_model_size_gb()
            self.log(f"  Model size: {model_size_gb:.2f}GB")
            # Need: original model + FP16 GGUF (~same size) + quants (~0.5x each, 12 quants)
            # Conservative estimate: model_size * 3 (original + fp16 + some quants at a time)
            required_gb = max(5.0, model_size_gb * 3)
            self.check_disk_space(required_gb) 
            
            # Run download in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model_dir = await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=self.hf_repo_id, 
                    local_dir=CACHE_DIR / self.hf_repo_id, 
                    local_dir_use_symlinks=False
                )
            )
            self.log(f"  ✓ Downloaded to {self.model_dir}")
            self.end_step("download")
            self.progress(30)
            self.log("")

            # 3. Convert to FP16
            self.start_step("convert")
            self.status("converting")
            self.log("▶ STEP 3: Converting to GGUF format (FP16)...")
            convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
            self.fp16_path = CACHE_DIR / f"{self.hf_repo_id.replace('/', '-')}-f16.gguf"
            
            cmd = [sys.executable, str(convert_script), str(self.model_dir), "--outfile", str(self.fp16_path), "--outtype", "f16"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            async for line in process.stdout:
                decoded = line.decode().strip()
                if decoded:
                    self.log(f"  {decoded}")
            
            returncode = await process.wait()
            
            if returncode != 0:
                raise Exception("Conversion to GGUF failed. Check logs for details.")
            
            self.log(f"  ✓ FP16 conversion complete: {self.fp16_path.name}")
            self.end_step("convert")
            self.progress(50)
            self.log("")

            # 4. Quantize
            self.start_step("quantize")
            self.status("quantizing")
            self.log("▶ STEP 4: Quantizing to all formats...")
            quant_base_name = self.hf_repo_id.split("/")[-1]
            hf_token = os.getenv("HF_TOKEN")
            
            # Get current user's HuggingFace username to create repo under their account
            api = HfApi(token=hf_token)
            new_repo_id = None
            
            if hf_token:
                try:
                    user_info = api.whoami()
                    hf_username = user_info.get("name") or user_info.get("user")
                    new_repo_id = f"{hf_username}/{quant_base_name}-GGUF"
                    self.log(f"  Target repo: {new_repo_id}")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: create_repo(new_repo_id, repo_type="model", token=hf_token, exist_ok=True)
                    )
                    self.log(f"  ✓ Repo ready: https://huggingface.co/{new_repo_id}")
                except Exception as e:
                    self.log(f"  ⚠ Could not create repo: {e}")
                    new_repo_id = None
            else:
                self.log("  ⚠ No HF_TOKEN set - files will be quantized but not uploaded")

            self.log("")
            successful_quants = []  # List of (q_type, q_path) tuples
            
            total_quants = len(QUANTS)
            for idx, q_type in enumerate(QUANTS):
                self.log(f"  [{idx+1}/{total_quants}] Quantizing {q_type}...")
                q_path = CACHE_DIR / f"{quant_base_name}.{q_type}.gguf"
                self.quant_paths.append(q_path)
                
                # Track time for this quant
                quant_start = time.time()
                
                # Run quantization asynchronously
                process = await asyncio.create_subprocess_exec(
                    str(quantize_bin), str(self.fp16_path), str(q_path), q_type,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                quant_duration = time.time() - quant_start
                
                if process.returncode != 0:
                    self.log(f"      ⚠ {q_type} failed: {stderr.decode()[:100]}")
                    continue
                
                self.quant_times.append((q_type, quant_duration))
                self.log(f"      ✓ {q_type} quantized successfully ({self.format_duration(quant_duration)})")
                successful_quants.append((q_type, q_path))
                
                step_progress = 50 + int((idx + 1) / total_quants * 30)
                self.progress(step_progress)
            
            self.end_step("quantize")
            self.log("")
            
            # 4b. Upload all quants at once
            uploaded_files = []
            if hf_token and new_repo_id and successful_quants:
                self.status("uploading")
                self.log("▶ STEP 4b: Uploading all quantized files...")
                self.log(f"  Uploading {len(successful_quants)} files concurrently...")
                
                def upload_single_file(q_type, q_path):
                    """Upload a single file - runs in thread pool"""
                    try:
                        api.upload_file(
                            path_or_fileobj=q_path,
                            path_in_repo=f"{quant_base_name}.{q_type}.gguf",
                            repo_id=new_repo_id,
                            repo_type="model"
                        )
                        return (q_type, True, None)
                    except Exception as e:
                        return (q_type, False, str(e))
                
                # Upload all files concurrently using ThreadPoolExecutor
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=4) as executor:
                    upload_tasks = [
                        loop.run_in_executor(executor, upload_single_file, q_type, q_path)
                        for q_type, q_path in successful_quants
                    ]
                    results = await asyncio.gather(*upload_tasks)
                
                for q_type, success, error in results:
                    if success:
                        uploaded_files.append(q_type)
                        self.log(f"      ✓ {q_type} uploaded")
                    else:
                        self.log(f"      ⚠ {q_type} upload failed: {error}")
                
                self.log(f"  ✓ Uploaded {len(uploaded_files)}/{len(successful_quants)} files")
            elif successful_quants:
                self.log(f"  ✓ {len(successful_quants)} quants saved locally (no upload)")
            
            self.progress(90)
            
            self.log("")

            # 5. Readme
            if hf_token and uploaded_files and new_repo_id:
                self.log("▶ STEP 5: Generating README...")
                
                # Get timing summary
                timing = self.get_timing_summary()
                total_time_str = self.format_duration(timing["total_time"])
                avg_quant_str = self.format_duration(timing["avg_quant_time"]) if timing["avg_quant_time"] > 0 else "N/A"
                
                # Build timing details
                timing_details = []
                if "download" in timing["step_times"]:
                    timing_details.append(f"- Download: {self.format_duration(timing['step_times']['download'])}")
                if "convert" in timing["step_times"]:
                    timing_details.append(f"- FP16 Conversion: {self.format_duration(timing['step_times']['convert'])}")
                if "quantize" in timing["step_times"]:
                    timing_details.append(f"- Quantization: {self.format_duration(timing['step_times']['quantize'])}")
                
                timing_section = "\n".join(timing_details)
                
                readme_content = f"""---
tags:
- gguf
- llama.cpp
- quantization
base_model: {self.hf_repo_id}
---

# {quant_base_name}-GGUF

This model was converted to GGUF format from [`{self.hf_repo_id}`](https://huggingface.co/{self.hf_repo_id}) using GGUF Forge.

## Quants
The following quants are available:
{', '.join(uploaded_files)}

## Conversion Stats

| Metric | Value |
|--------|-------|
| Job ID | `{self.model_id}` |
| GGUF Forge Version | {get_app_version()} |
| Total Time | {total_time_str} |
| Avg Time per Quant | {avg_quant_str} |

### Step Breakdown
{timing_section}

## 🚀 Convert Your Own Models

**Want to convert more models to GGUF?**

👉 **[gguforge.com](https://gguforge.com)** — Free hosted GGUF conversion service. Login with HuggingFace and request conversions instantly!

## Links

 - 🌐 **Free Hosted Service**: [gguforge.com](https://gguforge.com)
 - 🛠️ Self-host GGUF Forge: [GitHub](https://github.com/Akicuo/automaticConversion)
 - 📦 llama.cpp (quantization engine): [GitHub](https://github.com/ggerganov/llama.cpp)
 - 💬 Community & Support: [Discord](https://discord.gg/4vafUgVX3a)


---
*Converted automatically by [GGUF Forge](https://gguforge.com) {get_app_version()}*

"""
                api.upload_file(
                    path_or_fileobj=readme_content.encode('utf-8'),
                    path_in_repo="README.md",
                    repo_id=new_repo_id,
                    repo_type="model"
                )
                self.log(f"  ✓ README uploaded")
                self.log("")

            # Log timing summary
            timing = self.get_timing_summary()
            self.status("complete")
            self.progress(100)
            self.log("━━━ Pipeline Complete ━━━")
            self.log(f"✓ Successfully converted {self.hf_repo_id}")
            self.log(f"✓ Job ID: {self.model_id}")
            self.log(f"✓ Total Time: {self.format_duration(timing['total_time'])}")
            if timing["avg_quant_time"] > 0:
                self.log(f"✓ Avg Time per Quant: {self.format_duration(timing['avg_quant_time'])}")
            if new_repo_id:
                self.log(f"✓ Uploaded to: https://huggingface.co/{new_repo_id}")
            self._update_db(completed_at=datetime.now().isoformat())

        except Exception as e:
            error_details = traceback.format_exc()
            self.log("")
            self.log("━━━ Pipeline Failed ━━━")
            self.log(f"✗ ERROR: {str(e)}")
            self._update_db(error_details=error_details, status="error")
            logger.exception("Pipeline failed")
        
        finally:
            # Always cleanup files
            self.log("")
            self.log("▶ Cleanup...")
            self.cleanup()

# --- FastAPI App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    conn = get_db_connection()
    admin = conn.execute("SELECT * FROM users WHERE role = 'admin'").fetchone()
    if not admin:
        key = secrets.token_urlsafe(16)
        pwd = secrets.token_urlsafe(8)
        hashed = pwd_context.hash(pwd)
        conn.execute("INSERT INTO users (username, hashed_password, role, api_key) VALUES (?, ?, ?, ?)",
                     ("admin", hashed, "admin", key))
        conn.commit()
        
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
            
    conn.close()
    yield

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
    
    # Skip rate limiting for frequent polling endpoints (already filtered from logs)
    skip_rate_limit = path in ["/api/status/all", "/api/requests/all", "/api/requests/my"]
    
    # Rate limiting check (skip for polling endpoints to avoid false positives)
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


# Models for API
class LoginRequest(BaseModel):
    username: str
    password: str

class ProcessRequest(BaseModel):
    model_id: str

class ModelRequestSubmit(BaseModel):
    hf_repo_id: str

class RejectRequest(BaseModel):
    reason: Optional[str] = ""

# Dependencies
def get_current_user(request: Request):
    """Get current user - checks both admin users and OAuth users."""
    token = request.cookies.get("session_token")
    if not token: 
        return None
    conn = get_db_connection()
    # Check admin users first
    user = conn.execute("SELECT *, 'admin' as user_type FROM users WHERE api_key = ?", (token,)).fetchone()
    if user:
        conn.close()
        return user
    # Check OAuth users
    oauth_user = conn.execute("SELECT *, 'oauth' as user_type, 'user' as role FROM oauth_users WHERE session_token = ?", (token,)).fetchone()
    conn.close()
    return oauth_user

def get_oauth_user(request: Request):
    """Get OAuth user only (not admin)."""
    token = request.cookies.get("session_token")
    if not token: 
        return None
    conn = get_db_connection()
    oauth_user = conn.execute("SELECT * FROM oauth_users WHERE session_token = ?", (token,)).fetchone()
    conn.close()
    return oauth_user

def require_admin(user = Depends(get_current_user)):
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    oauth_user = get_oauth_user(request)
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "user": user['username'] if user else None,
        "role": user['role'] if user else 'guest',
        "oauth_avatar": oauth_user['avatar_url'] if oauth_user else None,
        "is_oauth": oauth_user is not None
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    
    if user and pwd_context.verify(password, user['hashed_password']):
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(key="session_token", value=user['api_key'])
        return response
    
    return templates.TemplateResponse("login.html", {"request": {}, "error": "Invalid credentials"})

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("session_token")
    return response

# --- HuggingFace OAuth Routes ---
@app.get("/auth/login")
async def oauth_login():
    """Redirect to HuggingFace OAuth authorization page."""
    if not OAUTH_CLIENT_ID:
        raise HTTPException(status_code=500, detail="OAuth not configured. Set OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET in .env")
    
    params = {
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "scope": "openid profile email",
        "response_type": "code",
        "state": secrets.token_urlsafe(16)
    }
    auth_url = f"{OAUTH_AUTHORIZE_URL}?{urlencode(params)}"
    return RedirectResponse(url=auth_url)

@app.get("/auth/callback")
async def oauth_callback(code: str, state: str = None):
    """Handle HuggingFace OAuth callback."""
    if not OAUTH_CLIENT_ID or not OAUTH_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="OAuth not configured")
    
    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            OAUTH_TOKEN_URL,
            data={
                "client_id": OAUTH_CLIENT_ID,
                "client_secret": OAUTH_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": OAUTH_REDIRECT_URI
            }
        )
        
        if token_response.status_code != 200:
            logger.error(f"OAuth token error: {token_response.text}")
            raise HTTPException(status_code=400, detail="Failed to authenticate with HuggingFace")
        
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        
        # Get user info
        userinfo_response = await client.get(
            OAUTH_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if userinfo_response.status_code != 200:
            logger.error(f"OAuth userinfo error: {userinfo_response.text}")
            raise HTTPException(status_code=400, detail="Failed to get user info from HuggingFace")
        
        userinfo = userinfo_response.json()
    
    # Extract user data
    hf_user_id = userinfo.get("sub")
    username = userinfo.get("preferred_username") or userinfo.get("name") or f"user_{hf_user_id[:8]}"
    email = userinfo.get("email")
    avatar_url = userinfo.get("picture")
    
    # Create session token
    session_token = secrets.token_urlsafe(32)
    
    # Save or update user in database
    conn = get_db_connection()
    existing = conn.execute("SELECT * FROM oauth_users WHERE id = ?", (hf_user_id,)).fetchone()
    
    if existing:
        conn.execute(
            "UPDATE oauth_users SET username = ?, email = ?, avatar_url = ?, session_token = ? WHERE id = ?",
            (username, email, avatar_url, session_token, hf_user_id)
        )
    else:
        conn.execute(
            "INSERT INTO oauth_users (id, username, email, avatar_url, session_token) VALUES (?, ?, ?, ?, ?)",
            (hf_user_id, username, email, avatar_url, session_token)
        )
    
    conn.commit()
    conn.close()
    
    # Set session cookie and redirect
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(key="session_token", value=session_token, httponly=True, max_age=86400*7)  # 7 days
    return response

@app.get("/auth/logout")
async def oauth_logout(request: Request):
    """Clear OAuth session."""
    token = request.cookies.get("session_token")
    if token:
        conn = get_db_connection()
        conn.execute("UPDATE oauth_users SET session_token = NULL WHERE session_token = ?", (token,))
        conn.commit()
        conn.close()
    
    response = RedirectResponse(url="/")
    response.delete_cookie("session_token")
    return response


@app.get("/api/hf/search")
async def search_hf(q: str):
    mgr = HuggingFaceManager(token=os.getenv("HF_TOKEN"))
    try:
        return mgr.search_models(q)
    except Exception as e:
        return []

@app.post("/api/models/process")
async def process_model(req: ProcessRequest, background_tasks: BackgroundTasks, user = Depends(require_admin)):
    conn = get_db_connection()
    existing = conn.execute("SELECT * FROM models WHERE hf_repo_id = ?", (req.model_id,)).fetchone()
    
    if existing and existing['status'] in ['pending', 'downloading', 'converting', 'quantizing', 'initializing']:
         conn.close()
         raise HTTPException(status_code=400, detail="Model already processing")
    
    new_id = str(uuid.uuid4())
    conn.execute(
        "INSERT OR REPLACE INTO models (id, hf_repo_id, status, progress, log, error_details) VALUES (?, ?, ?, ?, ?, ?)",
        (new_id, req.model_id, "pending", 0, "Queued...", "")
    )
    conn.commit()
    conn.close()
    
    workflow = ModelWorkflow(new_id, req.model_id)
    background_tasks.add_task(workflow.run_pipeline)
    
    return {"status": "started", "id": new_id}

@app.get("/api/status/all")
async def get_all_status():
    conn = get_db_connection()
    models = conn.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT 50").fetchall()
    conn.close()
    return [dict(m) for m in models]

@app.get("/api/status/model/{model_id}")
async def get_model_status(model_id: str):
    conn = get_db_connection()
    model = conn.execute("SELECT * FROM models WHERE hf_repo_id = ?", (model_id,)).fetchone()
    conn.close()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return dict(model)

# --- Request System ---
@app.post("/api/requests/submit")
async def submit_request(req: ModelRequestSubmit, request: Request):
    """Submit a request for a model to be converted (requires login)."""
    user = get_current_user(request)
    
    # Require login for submissions (anti-spam)
    if not user:
        raise HTTPException(status_code=401, detail="Login required to submit requests")
    
    requester = user['username']
    
    # Check spam protection - pending limit
    allowed, reason = spam_protection.check_pending_limit(requester)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    
    # Check spam protection - hourly rate
    allowed, reason = await spam_protection.can_submit(requester)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    
    conn = get_db_connection()
    # Check if already requested
    existing = conn.execute("SELECT * FROM requests WHERE hf_repo_id = ? AND status = 'pending'", (req.hf_repo_id,)).fetchone()
    if existing:
        conn.close()
        return {"status": "already_requested", "message": "This model has already been requested."}
    
    conn.execute(
        "INSERT INTO requests (hf_repo_id, requested_by, status) VALUES (?, ?, ?)",
        (req.hf_repo_id, requester, "pending")
    )
    conn.commit()
    conn.close()
    
    # Record submission for rate limiting
    await spam_protection.record_submission(requester)
    
    return {"status": "submitted", "message": "Your request has been submitted for admin review."}

@app.get("/api/requests/all")
async def get_all_requests(user = Depends(require_admin)):
    """Admin only: View all pending requests."""
    conn = get_db_connection()
    requests = conn.execute("SELECT * FROM requests ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in requests]

@app.post("/api/requests/{request_id}/approve")
async def approve_request(request_id: int, background_tasks: BackgroundTasks, user = Depends(require_admin)):
    """Admin only: Approve a request and start conversion."""
    conn = get_db_connection()
    req = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
    
    if not req:
        conn.close()
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Update request status
    conn.execute("UPDATE requests SET status = 'approved' WHERE id = ?", (request_id,))
    conn.commit()
    
    # Start the conversion
    hf_repo_id = req['hf_repo_id']
    new_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO models (id, hf_repo_id, status, progress, log, error_details) VALUES (?, ?, ?, ?, ?, ?)",
        (new_id, hf_repo_id, "pending", 0, "Queued from approved request...", "")
    )
    conn.commit()
    conn.close()
    
    workflow = ModelWorkflow(new_id, hf_repo_id)
    background_tasks.add_task(workflow.run_pipeline)
    
    return {"status": "approved", "model_id": new_id}

@app.post("/api/requests/{request_id}/reject")
async def reject_request(request_id: int, body: RejectRequest = None, user = Depends(require_admin)):
    """Admin only: Reject a request with optional reason."""
    reason = body.reason if body else ""
    conn = get_db_connection()
    conn.execute("UPDATE requests SET status = 'rejected', decline_reason = ? WHERE id = ?", (reason, request_id))
    conn.commit()
    conn.close()
    return {"status": "rejected"}

@app.get("/api/requests/my")
async def get_my_requests(request: Request):
    """Get current user's request history."""
    user = get_current_user(request)
    if not user:
        return []
    
    username = user['username']
    conn = get_db_connection()
    requests = conn.execute(
        "SELECT * FROM requests WHERE requested_by = ? ORDER BY created_at DESC",
        (username,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in requests]

@app.delete("/api/models/{model_id}")
async def delete_model_record(model_id: str, user = Depends(require_admin)):
    """Admin only: Delete a model record from the database."""
    conn = get_db_connection()
    conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

if __name__ == "__main__":
    import uvicorn
    print("Starting GGUF Forge...")
    uvicorn.run("app_gguf:app", host="0.0.0.0", port=8000, reload=True)
