"""
Model conversion workflow for GGUF Forge.
"""
import os
import sys
import shutil
import asyncio
import traceback
from typing import List
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, snapshot_download, create_repo

from database import get_db_connection
from managers import LlamaCppManager, get_app_version

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
        # Termination support
        self.terminated = False
        self.running_processes: List[asyncio.subprocess.Process] = []
    
    def terminate(self):
        """Request termination of this workflow."""
        self.terminated = True
        self.log("⚠ TERMINATION REQUESTED - Stopping workflow...")
        # Kill any running processes
        for proc in self.running_processes:
            try:
                proc.terminate()
            except Exception:
                pass
    
    def check_terminated(self):
        """Check if terminated and raise exception if so."""
        if self.terminated:
            raise Exception("Workflow terminated by admin")

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
        total, used, free = shutil.disk_usage(CACHE_DIR)
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
        import multiprocessing
        error_details = ""
        new_repo_id = None
        try:
            # Register in global registry for termination support
            running_workflows[self.model_id] = self
            
            self.start_time = time.time()
            self.status("initializing")
            self.progress(0)
            self.log("━━━ GGUF Forge Pipeline Started ━━━")
            self.log(f"Job ID: {self.model_id}")
            self.log(f"Model: {self.hf_repo_id}")
            self.log(f"Version: {get_app_version()}")
            self.log("")
            
            # 1. Setup Llama
            self.check_terminated()
            self.start_step("setup")
            self.log("▶ STEP 1: Setting up llama.cpp...")
            self.log("  Checking llama.cpp installation...")
            await LlamaCppManager.clone_repo()
            self.check_terminated()
            self.log("  Building llama.cpp (this may take a while)...")
            await LlamaCppManager.build()
            quantize_bin = LlamaCppManager.get_quantize_path()
            self.log(f"  ✓ llama-quantize ready: {quantize_bin.name}")
            self.end_step("setup")
            self.progress(10)
            self.log("")

            # 2. Download
            self.check_terminated()
            self.start_step("download")
            self.status("downloading")
            self.log("▶ STEP 2: Downloading model from HuggingFace...")
            self.log(f"  Source: https://huggingface.co/{self.hf_repo_id}")
            
            # Get actual model size and calculate required space
            model_size_gb = self.get_model_size_gb()
            self.log(f"  Model size: {model_size_gb:.2f}GB")
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
            self.check_terminated()
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
            self.check_terminated()
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
            
            # Detect CPU cores and split for parallel jobs
            total_cores = multiprocessing.cpu_count()
            parallel_jobs = max(1, min(PARALLEL_QUANT_JOBS, total_quants))
            cores_per_job = max(1, total_cores // parallel_jobs)
            
            if parallel_jobs == 1:
                self.log(f"  CPU cores: {total_cores} total (sequential mode)")
            else:
                self.log(f"  CPU cores: {total_cores} total, {cores_per_job} per quantization job")
                self.log(f"  Running {parallel_jobs} quantizations in parallel")
            self.log("")
            
            # Run quantizations with configured parallelism
            sem = asyncio.Semaphore(parallel_jobs)
            
            async def quantize_single(idx, q_type):
                async with sem:
                    self.log(f"  [{idx+1}/{total_quants}] Starting {q_type}...")
                    q_path = CACHE_DIR / f"{quant_base_name}.{q_type}.gguf"
                    self.quant_paths.append(q_path)
                    
                    quant_start = time.time()
                    
                    try:
                        # Set environment to limit thread usage
                        env = os.environ.copy()
                        env['OMP_NUM_THREADS'] = str(cores_per_job)
                        env['MKL_NUM_THREADS'] = str(cores_per_job)
                        env['OPENBLAS_NUM_THREADS'] = str(cores_per_job)
                        
                        process = await asyncio.create_subprocess_exec(
                            str(quantize_bin), str(self.fp16_path), str(q_path), q_type,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env
                        )
                        stdout, stderr = await process.communicate()
                        
                        quant_duration = time.time() - quant_start
                        
                        if process.returncode != 0:
                            self.log(f"      ⚠ {q_type} failed: {stderr.decode()[:100]}")
                        else:
                            self.quant_times.append((q_type, quant_duration))
                            self.log(f"      ✓ {q_type} ready ({self.format_duration(quant_duration)})")
                            successful_quants.append((q_type, q_path))
                            # Update progress
                            step_progress = 50 + int(len(successful_quants) / total_quants * 30)
                            self.progress(step_progress)
                    except Exception as e:
                        self.log(f"      ⚠ {q_type} error: {e}")

            # Launch all quantization jobs (controlled by PARALLEL_QUANT_JOBS semaphore)
            await asyncio.gather(*(quantize_single(i, q) for i, q in enumerate(QUANTS)))
            
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
            if self.terminated:
                self.log("━━━ Pipeline Terminated ━━━")
                self.log("⚠ Job was terminated by administrator")
                self._update_db(error_details="Terminated by administrator", status="terminated")
            else:
                self.log("━━━ Pipeline Failed ━━━")
                self.log(f"✗ ERROR: {str(e)}")
                self._update_db(error_details=error_details, status="error")
            import logging
            logging.getLogger("GGUF_Forge").exception("Pipeline failed")
        
        finally:
            # Remove from global registry
            running_workflows.pop(self.model_id, None)
            
            # Always cleanup files
            self.log("")
            self.log("▶ Cleanup...")
            self.cleanup()
