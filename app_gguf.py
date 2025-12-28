
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

# Handle paths for PyInstaller (Frozen) vs Dev
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
    BUNDLE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent.absolute()
    BUNDLE_DIR = BASE_DIR

TEMPLATES_DIR = BUNDLE_DIR / "templates"
CACHE_DIR = BASE_DIR / ".cache"
LLAMA_CPP_DIR = BASE_DIR / "llama.cpp"
DB_PATH = BASE_DIR / "gguf_app.db"

CACHE_DIR.mkdir(exist_ok=True)

# Llama.cpp Constants
QUANTS = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]

# Security
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
cookie_sec = APIKeyCookie(name="session_token", auto_error=False)

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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
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
    def clone_repo():
        if LlamaCppManager.is_installed():
            logger.info("llama.cpp already exists. Pulling latest...")
            subprocess.run(["git", "pull"], cwd=LLAMA_CPP_DIR, check=False)
        else:
            logger.info("Cloning llama.cpp...")
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP_DIR)], check=True)

    @staticmethod
    def build():
        logger.info("Building llama.cpp...")
        system = platform.system()
        
        try:
            if system == "Windows":
                build_dir = LLAMA_CPP_DIR / "build"
                build_dir.mkdir(exist_ok=True)
                subprocess.run(["cmake", "..", "-DLLAMA_CURL=OFF"], cwd=build_dir, check=True)
                subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, check=True)
                logger.info("Built with CMake on Windows.")
            else:
                subprocess.run(["make", "LLAMA_CURL=0"], cwd=LLAMA_CPP_DIR, check=True)
                logger.info("Built with Make on Linux/Mac.")
        except subprocess.CalledProcessError as e:
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

class ModelWorkflow:
    def __init__(self, model_id: str, hf_repo_id: str):
        self.model_id = model_id
        self.hf_repo_id = hf_repo_id
        self.log_buffer = []
        self.model_dir = None
        self.fp16_path = None
        self.quant_paths = []

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
        error_details = ""
        try:
            self.status("initializing")
            self.progress(0)
            self.log("━━━ GGUF Forge Pipeline Started ━━━")
            self.log(f"Model: {self.hf_repo_id}")
            self.log("")
            
            # 1. Setup Llama
            self.log("▶ STEP 1: Setting up llama.cpp...")
            self.log("  Checking llama.cpp installation...")
            LlamaCppManager.clone_repo()
            self.log("  Building llama.cpp (this may take a while)...")
            LlamaCppManager.build()
            quantize_bin = LlamaCppManager.get_quantize_path()
            self.log(f"  ✓ llama-quantize ready: {quantize_bin.name}")
            self.progress(10)
            self.log("")

            # 2. Download
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
            
            self.model_dir = snapshot_download(
                repo_id=self.hf_repo_id, 
                local_dir=CACHE_DIR / self.hf_repo_id, 
                local_dir_use_symlinks=False
            )
            self.log(f"  ✓ Downloaded to {self.model_dir}")
            self.progress(30)
            self.log("")

            # 3. Convert to FP16
            self.status("converting")
            self.log("▶ STEP 3: Converting to GGUF format (FP16)...")
            convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
            self.fp16_path = CACHE_DIR / f"{self.hf_repo_id.replace('/', '-')}-f16.gguf"
            
            cmd = [sys.executable, str(convert_script), str(self.model_dir), "--outfile", str(self.fp16_path), "--outtype", "f16"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                if line.strip():
                    self.log(f"  {line.strip()}")
            process.wait()
            
            if process.returncode != 0:
                raise Exception("Conversion to GGUF failed. Check logs for details.")
            
            self.log(f"  ✓ FP16 conversion complete: {self.fp16_path.name}")
            self.progress(50)
            self.log("")

            # 4. Quantize
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
                    create_repo(new_repo_id, repo_type="model", token=hf_token, exist_ok=True)
                    self.log(f"  ✓ Repo ready: https://huggingface.co/{new_repo_id}")
                except Exception as e:
                    self.log(f"  ⚠ Could not create repo: {e}")
                    new_repo_id = None
            else:
                self.log("  ⚠ No HF_TOKEN set - files will be quantized but not uploaded")

            self.log("")
            uploaded_files = []
            
            total_quants = len(QUANTS)
            for idx, q_type in enumerate(QUANTS):
                self.log(f"  [{idx+1}/{total_quants}] Quantizing {q_type}...")
                q_path = CACHE_DIR / f"{quant_base_name}.{q_type}.gguf"
                self.quant_paths.append(q_path)
                
                cmd = [str(quantize_bin), str(self.fp16_path), str(q_path), q_type]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.log(f"      ⚠ {q_type} failed: {result.stderr[:100]}")
                    continue
                
                if hf_token:
                    if new_repo_id:
                        self.log(f"      Uploading {q_type}...")
                        api.upload_file(
                            path_or_fileobj=q_path,
                            path_in_repo=f"{quant_base_name}.{q_type}.gguf",
                            repo_id=new_repo_id,
                            repo_type="model"
                        )
                        uploaded_files.append(q_type)
                        self.log(f"      ✓ {q_type} uploaded")
                    else:
                         self.log(f"      ✓ {q_type} saved locally (no upload)")
                
                step_progress = 50 + int((idx + 1) / total_quants * 40)
                self.progress(step_progress)
            
            self.log("")

            # 5. Readme
            if hf_token and uploaded_files and new_repo_id:
                self.log("▶ STEP 5: Generating README...")
                readme_content = f"""
---
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

### Automatic Conversion by GGUF Forge
"""
                api.upload_file(
                    path_or_fileobj=readme_content.encode('utf-8'),
                    path_in_repo="README.md",
                    repo_id=new_repo_id,
                    repo_type="model"
                )
                self.log(f"  ✓ README uploaded")
                self.log("")

            self.status("complete")
            self.progress(100)
            self.log("━━━ Pipeline Complete ━━━")
            self.log(f"✓ Successfully converted {self.hf_repo_id}")
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
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Models for API
class LoginRequest(BaseModel):
    username: str
    password: str

class ProcessRequest(BaseModel):
    model_id: str

class ModelRequestSubmit(BaseModel):
    hf_repo_id: str

# Dependencies
def get_current_user(request: Request):
    token = request.cookies.get("session_token")
    if not token: return None
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE api_key = ?", (token,)).fetchone()
    conn.close()
    return user

def require_admin(user = Depends(get_current_user)):
    if not user or user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "user": user['username'] if user else None,
        "role": user['role'] if user else 'guest'
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
    """Anyone can submit a request for a model to be converted."""
    user = get_current_user(request)
    requester = user['username'] if user else "anonymous"
    
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
async def reject_request(request_id: int, user = Depends(require_admin)):
    """Admin only: Reject a request."""
    conn = get_db_connection()
    conn.execute("UPDATE requests SET status = 'rejected' WHERE id = ?", (request_id,))
    conn.commit()
    conn.close()
    return {"status": "rejected"}

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
