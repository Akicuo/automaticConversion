"""
Model conversion workflow for GGUF Forge.
"""
import os
import sys
import json
import shutil
import asyncio
import traceback
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, snapshot_download, create_repo, hf_hub_download
from huggingface_hub.utils import tqdm as hf_tqdm

from database import get_db_connection
from managers import LlamaCppManager, get_app_version
from websocket_manager import broadcast_model_update, broadcast_transfer_progress

# These will be set by main app
CACHE_DIR = None
LLAMA_CPP_DIR = None
QUANTS = None
PARALLEL_QUANT_JOBS = None

# Global registry for running workflows (for termination support)
running_workflows: dict = {}  # model_id -> ModelWorkflow instance


def set_workflow_config(cache_dir: Path, llama_cpp_dir: Path, quants: list, parallel_jobs: int):
    """Set configuration for workflow module."""
    global CACHE_DIR, LLAMA_CPP_DIR, QUANTS, PARALLEL_QUANT_JOBS
    CACHE_DIR = cache_dir
    LLAMA_CPP_DIR = llama_cpp_dir
    QUANTS = quants
    PARALLEL_QUANT_JOBS = parallel_jobs


def get_quants_list():
    """Get the list of quants to process."""
    return QUANTS


class ModelWorkflow:
    def __init__(self, model_id: str, hf_repo_id: str, resume_mode: bool = False, completed_quants: Optional[List[str]] = None):
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
        # Transfer progress tracking
        self.transfer_files = {}  # filename -> {"progress": 0, "size": "", "speed": ""}
        # Termination support
        self.terminated = False
        self.running_processes: List[asyncio.subprocess.Process] = []
        # Resume support
        self.resume_mode = resume_mode
        self.completed_quants: List[str] = completed_quants or []  # Quants that have been uploaded already
        # For tracking the HF repo (needed for resume)
        self.new_repo_id = None
        self.hf_token = None
        self.api = None
    
    async def terminate(self):
        """Request termination of this workflow."""
        self.terminated = True
        await self.log("⚠ TERMINATION REQUESTED - Stopping workflow...")
        # Kill any running processes
        for proc in list(self.running_processes):
            try:
                proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    
    def check_terminated(self):
        """Check if terminated and raise exception if so."""
        if self.terminated:
            raise Exception("Workflow terminated by admin")

    async def log(self, message: str):
        print(f"[{self.hf_repo_id}] {message}")
        self.log_buffer.append(message)
        # Keep last 8k chars for better visibility in UI
        await self._update_db(log="\n".join(self.log_buffer)[-8000:])

    async def progress(self, percent: int):
        await self._update_db(progress=percent)

    async def status(self, status_msg: str):
        await self._update_db(status=status_msg)

    async def _update_db(self, **kwargs):
        conn = await get_db_connection()
        try:
            updates = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [self.model_id]
            await conn.execute(f"UPDATE models SET {updates} WHERE id = ?", values)
            await conn.commit()
            
            # Fetch updated model data and broadcast via WebSocket
            await conn.execute("SELECT * FROM models WHERE id = ?", (self.model_id,))
            model_data = await conn.fetchone()
            if model_data:
                await broadcast_model_update(model_data.to_dict())
        finally:
            await conn.close()
    
    async def save_completed_quant(self, q_type: str):
        """Save a completed quant to the database for resume capability."""
        if q_type not in self.completed_quants:
            self.completed_quants.append(q_type)
        await self._update_db(completed_quants=json.dumps(self.completed_quants))
    
    async def cleanup_safetensors(self):
        """Remove downloaded safetensors model directory to free up space."""
        if self.model_dir and Path(self.model_dir).exists():
            await self.log("  Cleaning up safetensors model to free disk space...")
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, lambda: shutil.rmtree(self.model_dir, ignore_errors=True))
                await self.log("  ✓ Safetensors model cleaned up")
                self.model_dir = None
            except Exception as e:
                await self.log(f"  ⚠ Failed to cleanup safetensors: {e}")

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
    
    async def update_transfer_progress(self, filename: str, progress: int, size: str = "", speed: str = "", transfer_type: str = "download"):
        """Update and broadcast transfer progress for a file."""
        self.transfer_files[filename] = {
            "name": filename,
            "progress": progress,
            "size": size,
            "speed": speed
        }
        
        # Broadcast the current transfer state
        files_list = list(self.transfer_files.values())
        await broadcast_transfer_progress(self.model_id, transfer_type, files_list)
    
    def clear_transfer_progress(self):
        """Clear transfer progress tracking."""
        self.transfer_files = {}
        return

    async def check_disk_space(self, required_gb: float):
        loop = asyncio.get_event_loop()
        total, used, free = await loop.run_in_executor(None, shutil.disk_usage, CACHE_DIR)
        free_gb = free / (2**30)
        await self.log(f"  Disk space check: Need {required_gb:.1f}GB, Available {free_gb:.1f}GB")
        if free_gb < required_gb:
            raise Exception(f"Insufficient disk space. Required: {required_gb:.1f}GB, Available: {free_gb:.1f}GB")
        await self.log(f"  ✓ Sufficient disk space")

    async def get_model_size_gb(self) -> float:
        """Get model size from HuggingFace API in GB."""
        try:
            hf_token = os.getenv("HF_TOKEN")
            api = HfApi(token=hf_token)
            
            # Run blocking API call in executor
            loop = asyncio.get_event_loop()
            model_info = await loop.run_in_executor(
                None,
                lambda: api.model_info(self.hf_repo_id, files_metadata=True)
            )
            
            total_bytes = 0
            if model_info.siblings:
                for sibling in model_info.siblings:
                    if hasattr(sibling, 'size') and sibling.size:
                        total_bytes += sibling.size
            
            size_gb = total_bytes / (2**30)
            return size_gb
        except Exception as e:
            await self.log(f"  ⚠ Could not fetch model size: {e}")
            return 10.0  # Default fallback

    async def cleanup(self):
        """Remove all downloaded and generated files."""
        await self.log("Starting cleanup...")
        loop = asyncio.get_event_loop()
        try:
            # Remove downloaded model directory
            if self.model_dir and Path(self.model_dir).exists():
                await self.log(f"Removing downloaded model: {self.model_dir}")
                await loop.run_in_executor(None, lambda: shutil.rmtree(self.model_dir, ignore_errors=True))
            
            # Remove FP16 file
            if self.fp16_path and self.fp16_path.exists():
                await self.log(f"Removing FP16 file: {self.fp16_path}")
                await loop.run_in_executor(None, lambda: self.fp16_path.unlink(missing_ok=True))
            
            # Remove all quantized files
            for q_path in self.quant_paths:
                if q_path.exists():
                    await self.log(f"Removing quant file: {q_path}")
                    await loop.run_in_executor(None, lambda p=q_path: p.unlink(missing_ok=True))
            
            await self.log("Cleanup completed.")
        except Exception as e:
            await self.log(f"Cleanup error (non-fatal): {e}")

    async def run_pipeline(self):
        import time
        import multiprocessing
        error_details = ""
        try:
            # Register in global registry for termination support
            running_workflows[self.model_id] = self
            
            self.start_time = time.time()
            await self.status("initializing")
            await self.progress(0)
            await self.log("━━━ GGUF Forge Pipeline Started ━━━")
            await self.log(f"Job ID: {self.model_id}")
            await self.log(f"Model: {self.hf_repo_id}")
            await self.log(f"Version: {await get_app_version()}")
            await self.log("")
            
            # 1. Setup Llama
            self.check_terminated()
            self.start_step("setup")
            await self.log("▶ STEP 1: Setting up llama.cpp...")
            await self.log("  Checking llama.cpp installation...")
            await LlamaCppManager.clone_repo()
            self.check_terminated()
            await self.log("  Building llama.cpp (this may take a while)...")
            await LlamaCppManager.build()
            quantize_bin = LlamaCppManager.get_quantize_path()
            await self.log(f"  ✓ llama-quantize ready: {quantize_bin.name}")
            self.end_step("setup")
            await self.progress(10)
            await self.log("")

            # 2. Download
            self.check_terminated()
            self.start_step("download")
            await self.status("downloading")
            await self.log("▶ STEP 2: Downloading model from HuggingFace...")
            await self.log(f"  Source: https://huggingface.co/{self.hf_repo_id}")
            
            # Get actual model size and calculate required space
            model_size_gb = await self.get_model_size_gb()
            await self.log(f"  Model size: {model_size_gb:.2f}GB")
            required_gb = max(5.0, model_size_gb * 3)
            await self.check_disk_space(required_gb)
            
            # Clear any previous transfer progress
            self.clear_transfer_progress()
            
            # Get list of files to download
            api = HfApi()
            loop = asyncio.get_event_loop()
            try:
                repo_files = await loop.run_in_executor(
                    None,
                    lambda: api.list_repo_files(self.hf_repo_id)
                )
                # Filter for model files (safetensors, bin, json, etc.)
                download_files = [f for f in repo_files if any(f.endswith(ext) for ext in 
                    ['.safetensors', '.bin', '.pt', '.pth', '.json', '.txt', '.model', '.tiktoken', '.py'])]
                
                await self.log(f"  Found {len(download_files)} files to download")
                
                # Download files with progress tracking
                local_dir = CACHE_DIR / self.hf_repo_id
                local_dir.mkdir(parents=True, exist_ok=True)
                
                total_files = len(download_files)
                for idx, filename in enumerate(download_files):
                    self.check_terminated()
                    short_name = filename.split('/')[-1] if '/' in filename else filename
                    
                    # Initialize progress for this file
                    await self.update_transfer_progress(short_name, 0, "", "Starting...", "download")
                    
                    # Download file in thread pool
                    try:
                        await loop.run_in_executor(
                            None,
                            lambda f=filename: hf_hub_download(
                                repo_id=self.hf_repo_id,
                                filename=f,
                                local_dir=local_dir,
                                local_dir_use_symlinks=False
                            )
                        )
                        # Mark as complete
                        await self.update_transfer_progress(short_name, 100, "", "Complete", "download")
                    except Exception as e:
                        await self.log(f"  ⚠ Failed to download {short_name}: {e}")
                        await self.update_transfer_progress(short_name, -1, "", "Failed", "download")
                    
                    # Update overall progress (10-30% for download step)
                    step_progress = 10 + int((idx + 1) / total_files * 20)
                    await self.progress(step_progress)
                
                self.model_dir = str(local_dir)
                
            except Exception as e:
                # Fallback to snapshot_download if file listing fails
                await self.log(f"  Using batch download...")
                self.model_dir = await loop.run_in_executor(
                    None,
                    lambda: snapshot_download(
                        repo_id=self.hf_repo_id, 
                        local_dir=CACHE_DIR / self.hf_repo_id, 
                        local_dir_use_symlinks=False
                    )
                )
            
            # Clear download progress display
            self.clear_transfer_progress()
            await broadcast_transfer_progress(self.model_id, "download", [])
            
            await self.log(f"  ✓ Downloaded to {self.model_dir}")
            self.end_step("download")
            await self.progress(30)
            await self.log("")

            # 3. Convert to FP16
            self.check_terminated()
            self.start_step("convert")
            await self.status("converting")
            await self.log("▶ STEP 3: Converting to GGUF format (FP16)...")
            convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
            self.fp16_path = CACHE_DIR / f"{self.hf_repo_id.replace('/', '-')}-f16.gguf"
            
            cmd = [sys.executable, str(convert_script), str(self.model_dir), "--outfile", str(self.fp16_path), "--outtype", "f16"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            self.running_processes.append(process)
            
            async for line in process.stdout:
                decoded = line.decode().strip()
                if decoded:
                    await self.log(f"  {decoded}")
            
            returncode = await process.wait()
            try:
                self.running_processes.remove(process)
            except ValueError:
                pass
            
            if returncode != 0:
                raise Exception("Conversion to GGUF failed. Check logs for details.")
            
            await self.log(f"  ✓ FP16 conversion complete: {self.fp16_path.name}")
            self.end_step("convert")
            await self.progress(50)
            
            # Clean up safetensors immediately - only the GGUF file is needed for quantization
            await self.cleanup_safetensors()
            await self.log("")

            # 4. Quantize and Upload (each quant is uploaded immediately after creation, then deleted)
            self.check_terminated()
            self.start_step("quantize")
            await self.status("quantizing")
            await self.log("▶ STEP 4: Quantizing and uploading each format...")
            quant_base_name = self.hf_repo_id.split("/")[-1]
            self.hf_token = os.getenv("HF_TOKEN")
            
            # Get current user's HuggingFace username to create repo under their account
            self.api = HfApi(token=self.hf_token)
            
            if self.hf_token:
                try:
                    loop = asyncio.get_event_loop()
                    user_info = await loop.run_in_executor(None, self.api.whoami)
                    hf_username = user_info.get("name") or user_info.get("user")
                    self.new_repo_id = f"{hf_username}/{quant_base_name}-GGUF"
                    await self.log(f"  Target repo: {self.new_repo_id}")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: create_repo(self.new_repo_id, repo_type="model", token=self.hf_token, exist_ok=True)
                    )
                    await self.log(f"  ✓ Repo ready: https://huggingface.co/{self.new_repo_id}")
                except Exception as e:
                    await self.log(f"  ⚠ Could not create repo: {e}")
                    self.new_repo_id = None
            else:
                await self.log("  ⚠ No HF_TOKEN set - files will be quantized but not uploaded")

            await self.log("")
            uploaded_files = []  # List of quant types that were uploaded
            
            # Determine which quants to process (skip already completed ones in resume mode)
            quants_to_process = [q for q in QUANTS if q not in self.completed_quants]
            
            if self.resume_mode and self.completed_quants:
                await self.log(f"  📋 Resume mode: {len(self.completed_quants)} quants already completed")
                await self.log(f"     Already done: {', '.join(self.completed_quants)}")
                await self.log(f"     Remaining: {len(quants_to_process)} quants to process")
                uploaded_files = list(self.completed_quants)  # Count already uploaded as successful
                await self.log("")
            
            total_quants = len(QUANTS)
            completed_count = len(self.completed_quants)
            
            # Detect CPU cores
            total_cores = multiprocessing.cpu_count()
            await self.log(f"  CPU cores: {total_cores} total")
            await self.log(f"  Mode: Sequential quantize → upload → delete (saves disk space)")
            await self.log("")
            
            # Process quants one at a time: quantize → upload → delete
            for idx, q_type in enumerate(quants_to_process):
                self.check_terminated()
                
                overall_idx = QUANTS.index(q_type) + 1
                await self.log(f"  [{overall_idx}/{total_quants}] Processing {q_type}...")
                
                q_path = CACHE_DIR / f"{quant_base_name}.{q_type}.gguf"
                
                quant_start = time.time()
                
                try:
                    # === QUANTIZE ===
                    env = os.environ.copy()
                    if quantize_bin and quantize_bin.parent:
                        current_ld = env.get('LD_LIBRARY_PATH', '')
                        env['LD_LIBRARY_PATH'] = f"{quantize_bin.parent}:{current_ld}"
                    env['OMP_NUM_THREADS'] = str(total_cores)
                    env['MKL_NUM_THREADS'] = str(total_cores)
                    env['OPENBLAS_NUM_THREADS'] = str(total_cores)
                    
                    process = await asyncio.create_subprocess_exec(
                        str(quantize_bin), str(self.fp16_path), str(q_path), q_type,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env
                    )
                    self.running_processes.append(process)
                    stdout, stderr = await process.communicate()
                    try:
                        self.running_processes.remove(process)
                    except ValueError:
                        pass
                    
                    quant_duration = time.time() - quant_start
                    
                    if process.returncode != 0:
                        await self.log(f"      ⚠ {q_type} quantization failed: {stderr.decode()[:200]}")
                        continue
                    
                    self.quant_times.append((q_type, quant_duration))
                    await self.log(f"      ✓ Quantized ({self.format_duration(quant_duration)})")
                    
                    # === UPLOAD ===
                    if self.hf_token and self.new_repo_id:
                        self.check_terminated()
                        
                        filename = f"{quant_base_name}.{q_type}.gguf"
                        file_size = q_path.stat().st_size if q_path.exists() else 0
                        size_str = f"{file_size / (1024**3):.2f}GB" if file_size > 0 else ""
                        
                        await self.update_transfer_progress(filename, 0, size_str, "Uploading...", "upload")
                        
                        try:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(
                                None,
                                lambda: self.api.upload_file(
                                    path_or_fileobj=q_path,
                                    path_in_repo=filename,
                                    repo_id=self.new_repo_id,
                                    repo_type="model"
                                )
                            )
                            
                            await self.update_transfer_progress(filename, 100, size_str, "Complete", "upload")
                            await self.log(f"      ✓ Uploaded to HuggingFace")
                            uploaded_files.append(q_type)
                            
                            # Save progress to DB for resume capability
                            await self.save_completed_quant(q_type)
                            
                        except Exception as e:
                            await self.update_transfer_progress(filename, -1, size_str, "Failed", "upload")
                            await self.log(f"      ⚠ Upload failed: {e}")
                            # Don't delete the file if upload failed - keep for retry
                            continue
                    else:
                        await self.log(f"      ℹ Skipping upload (no HF token)")
                        uploaded_files.append(q_type)  # Count as "done" for progress
                    
                    # === DELETE QUANT FILE ===
                    try:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, lambda: q_path.unlink(missing_ok=True))
                        await self.log(f"      ✓ Deleted local file")
                    except Exception as e:
                        await self.log(f"      ⚠ Failed to delete: {e}")
                    
                    # Clear transfer progress
                    self.clear_transfer_progress()
                    await broadcast_transfer_progress(self.model_id, "upload", [])
                    
                except Exception as e:
                    await self.log(f"      ⚠ {q_type} error: {e}")
                
                # Update progress
                completed_count = len(uploaded_files)
                step_progress = 50 + int(completed_count / total_quants * 40)
                await self.progress(step_progress)
            
            self.end_step("quantize")
            await self.log("")
            await self.log(f"  ✓ Completed {len(uploaded_files)}/{total_quants} quants")
            
            await self.progress(90)
            
            await self.log("")

            # 5. Readme
            if self.hf_token and uploaded_files and self.new_repo_id:
                await self.log("▶ STEP 5: Generating README...")
                
                # Get app version (async)
                app_version = await get_app_version()
                
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
| GGUF Forge Version | {app_version} |
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
*Converted automatically by [GGUF Forge](https://gguforge.com) {app_version}*

"""
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.api.upload_file(
                        path_or_fileobj=readme_content.encode('utf-8'),
                        path_in_repo="README.md",
                        repo_id=self.new_repo_id,
                        repo_type="model"
                    )
                )
                await self.log(f"  ✓ README uploaded")
                await self.log("")

            # Log timing summary
            timing = self.get_timing_summary()
            await self.status("complete")
            await self.progress(100)
            await self.log("━━━ Pipeline Complete ━━━")
            await self.log(f"✓ Successfully converted {self.hf_repo_id}")
            await self.log(f"✓ Job ID: {self.model_id}")
            await self.log(f"✓ Total Time: {self.format_duration(timing['total_time'])}")
            if timing["avg_quant_time"] > 0:
                await self.log(f"✓ Avg Time per Quant: {self.format_duration(timing['avg_quant_time'])}")
            if self.new_repo_id:
                await self.log(f"✓ Uploaded to: https://huggingface.co/{self.new_repo_id}")
            await self._update_db(completed_at=datetime.now())

        except Exception as e:
            error_details = traceback.format_exc()
            await self.log("")
            if self.terminated:
                await self.log("━━━ Pipeline Terminated ━━━")
                await self.log("⚠ Job was terminated by administrator")
                await self._update_db(error_details="Terminated by administrator", status="terminated")
            else:
                await self.log("━━━ Pipeline Failed ━━━")
                await self.log(f"✗ ERROR: {str(e)}")
                await self._update_db(error_details=error_details, status="error")
            import logging
            logging.getLogger("GGUF_Forge").exception("Pipeline failed")
        
        finally:
            # Remove from global registry
            running_workflows.pop(self.model_id, None)
            
            # Always cleanup files
            await self.log("")
            await self.log("▶ Cleanup...")
            await self.cleanup()
