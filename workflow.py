"""
Model conversion workflow for GGUF Forge.
"""
import os
import sys
import json
import re
import shutil
import asyncio
import traceback
from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Prefer HuggingFace's accelerated transfer backends for large GGUF uploads.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi, snapshot_download, create_repo, hf_hub_download
from huggingface_hub.utils import tqdm as hf_tqdm

from database import get_db_connection
import managers
from managers import LlamaCppManager, get_app_version
from websocket_manager import broadcast_model_update, broadcast_transfer_progress

# These will be set by main app
CACHE_DIR = None
LLAMA_CPP_DIR = None
QUANTS = None
PARALLEL_QUANT_JOBS = None

# Global registry for running workflows (for termination support)
running_workflows: dict = {}  # model_id -> ModelWorkflow instance

# Global model queue instance
model_queue = None


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


def get_cache_dir():
    """Return the configured CACHE_DIR (set via set_workflow_config)."""
    return CACHE_DIR


# File extensions counted as model weights for size/detection purposes.
_MODEL_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth")


def _humanize_bytes(n: float) -> str:
    """Format a byte count as e.g. '1.23 GB', '450 MB', '2.1 KB'. Negative/None → '—'."""
    if n is None or n < 0:
        return "—"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.2f} {unit}" if n < 10 else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _has_safetensors(directory: Path) -> bool:
    try:
        return any(directory.glob("*.safetensors"))
    except OSError:
        return False


def _is_hf_model_dir(directory: Path) -> bool:
    """A directory looks like an HF model snapshot if it has config.json + safetensors."""
    try:
        return (directory / "config.json").is_file() and _has_safetensors(directory)
    except OSError:
        return False


def _model_dir_stats(directory: Path) -> Tuple[int, int]:
    """Sum size of weight files (bytes) and total file count in directory."""
    total_size = 0
    file_count = 0
    try:
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            file_count += 1
            if entry.suffix.lower() in _MODEL_WEIGHT_EXTS:
                try:
                    total_size += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total_size, file_count


def scan_local_models(cache_dir: Optional[Path] = None) -> list:
    """Scan CACHE_DIR for already-downloaded HuggingFace models.

    Detects two layouts:
      - HF org/repo: cache_dir/{org}/{repo}/  (config.json + *.safetensors)
      - Flat:       cache_dir/{name}/        (config.json + *.safetensors)

    Returns a list of dicts: {repo_id, path, size_bytes, file_count, layout}.
    """
    base = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    if base is None or not base.exists():
        return []

    results = []
    try:
        top_entries = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return []

    for top in top_entries:
        # HF org/repo layout: top is the org, subdirs are repos.
        hf_matches = []
        try:
            for sub in sorted(p for p in top.iterdir() if p.is_dir()):
                if _is_hf_model_dir(sub):
                    hf_matches.append(sub)
        except OSError:
            pass

        if hf_matches:
            for sub in hf_matches:
                size, count = _model_dir_stats(sub)
                results.append({
                    "repo_id": f"{top.name}/{sub.name}",
                    "path": str(sub.resolve()),
                    "size_bytes": size,
                    "file_count": count,
                    "layout": "hf",
                })
            continue

        # Flat layout: the top-level dir itself is the model.
        if _is_hf_model_dir(top):
            size, count = _model_dir_stats(top)
            results.append({
                "repo_id": f"local/{top.name}",
                "path": str(top.resolve()),
                "size_bytes": size,
                "file_count": count,
                "layout": "flat",
            })

    return results


async def get_quant_priority_order():
    """Get the priority order for quants from database. Returns default if not configured."""
    from database import get_db_connection

    try:
        conn = await get_db_connection()
        await conn.execute("SELECT priority_order FROM quant_priority WHERE id = 1")
        row = await conn.fetchone()
        await conn.close()

        if row and row.get('priority_order'):
            try:
                custom_order = json.loads(row['priority_order'])
                # Validate and filter
                valid_order = [q for q in custom_order if q in QUANTS]
                # Add missing quants at the end
                missing = [q for q in QUANTS if q not in valid_order]
                return valid_order + missing
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception:
        pass

    return list(QUANTS)


class ModelQueue:
    """Queue system to process one model at a time.
    
    Ensures only one model workflow runs at any given time, preventing
    resource contention from multiple simultaneous downloads/quantizations.
    """
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.current_workflow = None
        self.worker_task = None
        self._queue_list = []  # Track queue for status reporting
        
    async def add(self, workflow: "ModelWorkflow"):
        """Add a workflow to the queue."""
        import logging
        logger = logging.getLogger("GGUF_Forge")
        
        self._queue_list.append({
            "model_id": workflow.model_id,
            "hf_repo_id": workflow.hf_repo_id,
            "added_at": asyncio.get_event_loop().time()
        })
        
        queue_position = len(self._queue_list)
        logger.info(f"Model {workflow.hf_repo_id} added to queue (position {queue_position})")
        
        # Update model status in database
        from database import get_db_connection
        conn = await get_db_connection()
        try:
            if queue_position == 1 and self.current_workflow is None:
                # First in queue and nothing processing - will start immediately
                await conn.execute(
                    "UPDATE models SET log = ? WHERE id = ?",
                    (f"In queue (position 1) - starting immediately...\nQuants: {', '.join(workflow.quants_to_run) if hasattr(workflow, 'quants_to_run') and workflow.quants_to_run else 'all'}", workflow.model_id)
                )
            else:
                await conn.execute(
                    "UPDATE models SET log = ? WHERE id = ?",
                    (f"In queue (position {queue_position}) - waiting for other models to complete...\nQuants: {', '.join(workflow.quants_to_run) if hasattr(workflow, 'quants_to_run') and workflow.quants_to_run else 'all'}", workflow.model_id)
                )
            await conn.commit()
        finally:
            await conn.close()
        
        await self.queue.put(workflow)
        
        # Broadcast queue update via WebSocket
        from websocket_manager import manager as ws_manager
        await ws_manager.broadcast("models", {
            "type": "queue_update",
            "queue_size": self.queue.qsize(),
            "current_model": self.current_workflow.model_id if self.current_workflow else None
        })
        
    def start_worker(self):
        """Start the background worker that processes the queue."""
        import logging
        logger = logging.getLogger("GGUF_Forge")
        logger.info("Starting model queue worker...")
        self.worker_task = asyncio.create_task(self._worker())
        
    async def _worker(self):
        """Background worker that processes workflows one at a time."""
        import logging
        logger = logging.getLogger("GGUF_Forge")
        logger.info("Model queue worker started")
        
        while True:
            try:
                # Wait for next workflow in queue
                workflow = await self.queue.get()
                self.current_workflow = workflow
                
                # Remove from tracking list
                self._queue_list = [item for item in self._queue_list if item["model_id"] != workflow.model_id]
                
                # Update queue positions for waiting models
                await self._update_queue_positions()
                
                logger.info(f"Processing model: {workflow.hf_repo_id} (ID: {workflow.model_id})")
                
                try:
                    # Run the workflow pipeline
                    await workflow.run_pipeline()
                    logger.info(f"Model {workflow.hf_repo_id} completed successfully")
                except Exception as e:
                    logger.error(f"Model {workflow.hf_repo_id} failed: {e}")
                finally:
                    self.current_workflow = None
                    self.queue.task_done()
                    
                    # Broadcast queue update
                    from websocket_manager import manager as ws_manager
                    await ws_manager.broadcast("models", {
                        "type": "queue_update",
                        "queue_size": self.queue.qsize(),
                        "current_model": None
                    })
                    
            except Exception as e:
                logger.error(f"Queue worker error: {e}")
                self.current_workflow = None
                
    async def _update_queue_positions(self):
        """Update database with current queue positions for waiting models."""
        from database import get_db_connection
        
        position = 1
        for item in self._queue_list:
            try:
                conn = await get_db_connection()
                await conn.execute(
                    "UPDATE models SET log = REPLACE(log, 'position ' || ?, 'position ' || ?) WHERE id = ?",
                    (str(position + 1), str(position), item["model_id"])
                )
                await conn.commit()
                await conn.close()
                position += 1
            except Exception:
                pass  # Non-critical, continue
                
    def get_status(self):
        """Get current queue status."""
        return {
            "current_model_id": self.current_workflow.model_id if self.current_workflow else None,
            "current_hf_repo": self.current_workflow.hf_repo_id if self.current_workflow else None,
            "waiting_count": self.queue.qsize(),
            "queue": [
                {
                    "model_id": item["model_id"],
                    "hf_repo_id": item["hf_repo_id"],
                    "position": idx + 1
                }
                for idx, item in enumerate(self._queue_list)
            ]
        }
        
    async def clear(self):
        """Clear the queue (admin function)."""
        import logging
        logger = logging.getLogger("GGUF_Forge")
        
        # Clear the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        self._queue_list.clear()
        logger.info("Queue cleared")


def get_model_queue():
    """Get the global model queue instance."""
    global model_queue
    return model_queue


def set_model_queue(queue: ModelQueue):
    """Set the global model queue instance."""
    global model_queue
    model_queue = queue


class ModelWorkflow:
    def __init__(self, model_id: str, hf_repo_id: str, resume_mode: bool = False,
                 completed_quants: Optional[List[str]] = None, quants_to_run: Optional[List[str]] = None,
                 ignore_space_check: bool = False, force_llama_update: bool = False,
                 enable_shard_merging: bool = True, requested_by: Optional[str] = None,
                 local_source_path: Optional[str] = None, keep_local_only: bool = False,
                 convert_outtype: Optional[str] = None):
        self.model_id = model_id
        self.hf_repo_id = hf_repo_id
        self.log_buffer = []
        self.model_dir = None
        self.fp16_path = None
        self.quant_paths = []
        self.current_progress = 0
        # Time tracking
        self.start_time = None
        self.step_times = {}  # step_name -> (start, end)
        self.quant_times = []  # list of (q_type, duration_seconds)
        # Transfer progress tracking
        self.transfer_files = {}  # filename -> {"progress": 0, "size": "", "speed": ""}
        self._last_transfer_persist = 0.0  # monotonic seconds; throttles DB writes
        self._last_transfer_type = "download"  # remembered for restore-on-reload
        # Termination support
        self.terminated = False
        self.running_processes: List[asyncio.subprocess.Process] = []
        # Resume support
        self.resume_mode = resume_mode
        self.completed_quants: List[str] = completed_quants or []  # Quants that have been uploaded already
        # Custom quants - if specified, only these quants will be processed
        self.quants_to_run: List[str] = quants_to_run if quants_to_run else QUANTS
        # For tracking the HF repo (needed for resume)
        self.new_repo_id = None
        self.hf_token = None
        self.api = None
        # Admin override - skip conservative disk space checks
        self.ignore_space_check = ignore_space_check
        # Force llama.cpp update flag
        self.force_llama_update = force_llama_update
        # Admin override - enable/disable shard merging
        self.enable_shard_merging = enable_shard_merging
        # Who requested this conversion (HuggingFace username)
        self.requested_by = requested_by
        # Local-source mode: skip download, never delete the source dir
        self.local_source_path = local_source_path
        # If True, skip HF upload and move quantized GGUFs into <local_source_path>/gguf/
        self.keep_local_only = keep_local_only
        # Vision/multimodal support: detected during convert step, set if model has a vision tower
        self.is_vision_model = False
        self.mmproj_path: Optional[Path] = None
        self.mmproj_uploaded_name: Optional[str] = None
        # Fork-specific compact convert outtype (e.g. "q8_0", "iq2_xxs"). When set
        # to a value outside {f16, bf16, f32}, convert_hf_to_gguf.py emits the
        # final artifact directly and the llama-quantize loop is skipped.
        self.convert_outtype: Optional[str] = (convert_outtype or "").strip().lower() or None
        self.is_direct_outtype: bool = bool(
            self.convert_outtype
            and self.convert_outtype not in {"f16", "bf16", "f32"}
        )

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
        # Keep UI progress monotonic during resume/parallel phases.
        percent = max(0, min(100, int(percent)))
        if percent < self.current_progress:
            percent = self.current_progress
        self.current_progress = percent
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
        if self.local_source_path:
            await self.log("  ℹ Local source detected — keeping original safetensors")
            return
        if self.model_dir and Path(self.model_dir).exists():
            await self.log("  Cleaning up safetensors model to free disk space...")
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, lambda: shutil.rmtree(self.model_dir, ignore_errors=True))
                await self.log("  ✓ Safetensors model cleaned up")
                self.model_dir = None
            except Exception as e:
                await self.log(f"  ⚠ Failed to cleanup safetensors: {e}")

    async def _upload_with_retry(self, do_upload, description: str, max_attempts: int = 3) -> bool:
        """Run a HuggingFace upload callable with retries on failure.

        do_upload: zero-arg callable that performs the upload synchronously.
        Returns True on success, False after all attempts fail.
        """
        loop = asyncio.get_event_loop()
        for attempt in range(1, max_attempts + 1):
            self.check_terminated()
            try:
                await loop.run_in_executor(None, do_upload)
                if attempt > 1:
                    await self.log(f"      ✓ {description} succeeded on attempt {attempt}")
                return True
            except Exception as e:
                if attempt < max_attempts:
                    wait_s = 2 ** attempt
                    await self.log(f"      ⚠ {description} attempt {attempt}/{max_attempts} failed: {e} — retrying in {wait_s}s")
                    try:
                        await asyncio.sleep(wait_s)
                    except asyncio.CancelledError:
                        raise
                else:
                    await self.log(f"      ✗ {description} failed after {max_attempts} attempts: {e}")
        return False

    def _detect_vision_model(self) -> bool:
        """Inspect config.json to decide if this is a multimodal/vision model with a projector."""
        if not self.model_dir:
            return False
        config_path = Path(self.model_dir) / "config.json"
        if not config_path.is_file():
            return False
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        # Most multimodal HF configs expose a nested vision config or tower reference.
        vision_keys = ("vision_config", "vision_tower", "mm_vision_tower", "image_token_index", "vision_feature_layer")
        if any(k in config for k in vision_keys):
            return True

        # Fallback: match known multimodal architecture class names.
        known_vision_archs = {
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
            "LlavaNextVideoForConditionalGeneration",
            "LlavaOnevisionForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Gemma3ForConditionalGeneration",
            "Llama4ForConditionalGeneration",
            "MllamaForConditionalGeneration",
            "InternVLChatModel",
            "Idefics2ForConditionalGeneration",
            "Idefics3ForConditionalGeneration",
            "MiniCPMV",
            "MiniCPMVForConditionalGeneration",
            "PaliGemmaForConditionalGeneration",
            "Phi3VForCausalLM",
            "SmolVLMForConditionalGeneration",
        }
        architectures = config.get("architectures") or []
        if isinstance(architectures, list) and any(a in known_vision_archs for a in architectures):
            return True

        return False

    async def _generate_mmproj(self) -> Optional[Path]:
        """Run convert_hf_to_gguf.py --mmproj to emit the multimodal projector file."""
        if not self.model_dir or not Path(self.model_dir).exists():
            await self.log("  ⚠ Cannot generate mmproj: source model directory not available")
            return None

        convert_script = managers.LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            await self.log(f"  ⚠ Cannot generate mmproj: convert_hf_to_gguf.py missing at {convert_script}")
            return None

        slug = self.hf_repo_id.replace('/', '-')
        mmproj_out = CACHE_DIR / f"mmproj-{slug}-f16.gguf"
        # Remove any stale leftover before running.
        try:
            mmproj_out.unlink(missing_ok=True)
        except Exception:
            pass

        cmd = [
            sys.executable, str(convert_script), str(self.model_dir),
            "--outfile", str(mmproj_out), "--outtype", "f16", "--mmproj",
        ]
        await self.log(f"  Running: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.running_processes.append(process)
        try:
            async for line in process.stdout:
                decoded = line.decode(errors="replace").strip()
                if decoded:
                    await self.log(f"    {decoded}")
            returncode = await process.wait()
        finally:
            try:
                self.running_processes.remove(process)
            except ValueError:
                pass

        if returncode != 0:
            await self.log(f"  ⚠ mmproj generation exited with code {returncode}")
            # Some llama.cpp versions reject --mmproj on certain archs; treat as best-effort.
            return None

        # llama.cpp may rewrite the output filename (e.g. prefixing with mmproj-). Locate the result.
        if mmproj_out.exists() and mmproj_out.stat().st_size > 0:
            return mmproj_out

        candidates = sorted(CACHE_DIR.glob(f"mmproj-*{slug}*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]

        # Last resort: look inside the source model dir, where the converter sometimes writes by default.
        try:
            in_model_dir = sorted(Path(self.model_dir).glob("mmproj-*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
            if in_model_dir:
                target = CACHE_DIR / in_model_dir[0].name
                shutil.move(str(in_model_dir[0]), str(target))
                return target
        except Exception as e:
            await self.log(f"  ⚠ mmproj generation produced no output file: {e}")
            return None

        await self.log("  ⚠ mmproj generation produced no output file")
        return None

    def _get_gguf_shards(self, base_path: Path) -> List[Tuple[int, int, Path]]:
        """Find sharded GGUF files matching a base output path."""
        stem = base_path.stem
        if not stem:
            return []
        
        pattern = re.compile(rf"^{re.escape(stem)}-(\d{{5}})-of-(\d{{5}})\.gguf$")
        shard_sets = {}
        
        for file_path in base_path.parent.glob(f"{stem}-?????-of-?????.gguf"):
            match = pattern.match(file_path.name)
            if not match:
                continue
            idx = int(match.group(1))
            total = int(match.group(2))
            shard_sets.setdefault(total, []).append((idx, file_path))
        
        if not shard_sets:
            return []
        
        # Prefer the shard set with the most parts (handles stale leftovers).
        total = max(shard_sets.keys(), key=lambda t: len(shard_sets[t]))
        shards = shard_sets[total]
        shards.sort(key=lambda s: s[0])
        return [(idx, total, path) for idx, path in shards]
    
    async def _cleanup_gguf_shards(self, shard_paths: List[Path], q_type: str):
        """Delete shard files after successful merge."""
        loop = asyncio.get_event_loop()
        for shard_path in shard_paths:
            try:
                await loop.run_in_executor(None, lambda p=shard_path: p.unlink(missing_ok=True))
            except Exception as e:
                await self.log(f"      ⚠ {q_type} Failed to delete shard {shard_path.name}: {e}")
    
    async def ensure_unsharded_gguf(self, q_path: Path, q_type: str) -> Optional[Path]:
        """Merge sharded GGUF output into a single file when needed."""
        shards = self._get_gguf_shards(q_path)
        if not shards:
            if q_path.exists():
                return q_path
            await self.log(f"      ⚠ {q_type} Output file missing: {q_path.name}")
            return None

        total = shards[0][1]
        shard_paths = [path for _, _, path in shards]
        shard_indices = {idx for idx, _, _ in shards}
        missing = [i for i in range(1, total + 1) if i not in shard_indices]

        if missing:
            preview = ", ".join(f"{i:05d}" for i in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            await self.log(f"      ⚠ {q_type} Shard set incomplete (missing {preview}{suffix})")
            return None

        # Check if shard merging is disabled
        if not self.enable_shard_merging:
            await self.log(f"      ℹ {q_type} Shard merging disabled by admin - keeping sharded output")
            # Return the first shard path as the output
            return shard_paths[0]

        if q_path.exists():
            try:
                base_mtime = q_path.stat().st_mtime
                latest_shard_mtime = max(p.stat().st_mtime for p in shard_paths)
                if latest_shard_mtime < base_mtime:
                    await self.log(f"      ℹ {q_type} Shards are older than merged output - skipping merge")
                    return q_path
            except Exception:
                pass

        await self.log(f"      ℹ {q_type} Output is sharded ({total} parts). Merging...")
        
        try:
            gguf_split_bin = LlamaCppManager.get_gguf_split_path()
        except FileNotFoundError as e:
            await self.log(f"      ⚠ {q_type} Merge tool not found: {e}")
            return None
        
        merge_output = q_path
        if merge_output.exists():
            merge_output = q_path.with_suffix(".merged.gguf")
        if merge_output.exists():
            try:
                merge_output.unlink(missing_ok=True)
            except Exception:
                pass
        
        process = await asyncio.create_subprocess_exec(
            str(gguf_split_bin), "--merge", str(shard_paths[0]), str(merge_output),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        self.running_processes.append(process)
        stdout, stderr = await process.communicate()
        try:
            self.running_processes.remove(process)
        except ValueError:
            pass
        
        if process.returncode != 0:
            error_output = (stderr.decode().strip() or stdout.decode().strip() or "Unknown error")
            await self.log(f"      ⚠ {q_type} Shard merge failed: {error_output[:200]}")
            return None
        
        if merge_output != q_path:
            try:
                q_path.unlink(missing_ok=True)
            except Exception as e:
                await self.log(f"      ⚠ {q_type} Failed to remove old output: {e}")
                return None
            try:
                merge_output.replace(q_path)
            except Exception as e:
                await self.log(f"      ⚠ {q_type} Failed to finalize merged file: {e}")
                return None
        
        await self.log(f"      ✓ {q_type} Shards merged into {q_path.name}")
        await self._cleanup_gguf_shards(shard_paths, q_type)
        return q_path

    async def upload_status_readme(self, quant_base_name: str, uploaded_files: List[str]):
        """Upload a temporary README with current conversion status."""
        if not (self.hf_token and self.new_repo_id and self.api):
            return

        try:
            app_version = await get_app_version()
            updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            completed_display = ", ".join(uploaded_files) if uploaded_files else "None yet"
            remaining = [q for q in self.quants_to_run if q not in uploaded_files]
            remaining_display = ", ".join(remaining) if remaining else "None"
            progress = f"{len(uploaded_files)}/{len(self.quants_to_run)}"

            # Build requester section
            requester_section = ""
            if self.requested_by:
                requester_section = f"\n- Requested by: [@{self.requested_by}](https://huggingface.co/{self.requested_by})"

            readme_content = f"""---
tags:
- gguf
- llama.cpp
- quantization
base_model: {self.hf_repo_id}
---

# {quant_base_name}-GGUF

This repository is being generated by GGUF Forge and will update as quants finish.

## Status
- Job ID: `{self.model_id}`{requester_section}
- Stage: Quantizing
- Updated: {updated_at}
- Progress: {progress}
- Completed quants: {completed_display}
- Remaining quants: {remaining_display}

## Ollama Support
Full Ollama support is provided by merging any sharded GGUF output into a single file after quantization.

---
*This README is temporary and will be replaced when conversion completes.*
*Converted automatically by [GGUF Forge](https://gguforge.com) {app_version}*

"""
            ok = await self._upload_with_retry(
                lambda: self.api.upload_file(
                    path_or_fileobj=readme_content.encode('utf-8'),
                    path_in_repo="README.md",
                    repo_id=self.new_repo_id,
                    repo_type="model"
                ),
                description="Status README upload",
            )
            if ok:
                await self.log("  ✓ Status README uploaded")
        except Exception as e:
            await self.log(f"  ⚠ Status README upload failed: {e}")

    async def notify_requester(self, quant_base_name: str, uploaded_files: List[str]):
        """Create a HuggingFace discussion to notify the requester that their model is ready."""
        if not (self.hf_token and self.new_repo_id and self.api and self.requested_by):
            return

        try:
            await self.log(f"  Notifying @{self.requested_by} via HuggingFace...")

            # Create a discussion on the new GGUF repo to notify the user
            discussion_title = f"🎉 Your GGUF conversion is complete!"
            discussion_body = f"""Hey @{self.requested_by}! 👋

Great news! Your requested GGUF conversion is now complete!

**Model**: [`{self.hf_repo_id}`](https://huggingface.co/{self.hf_repo_id})
**GGUF Repo**: [`{self.new_repo_id}`](https://huggingface.co/{self.new_repo_id})

## Available Quantizations
{chr(10).join([f'- **{q}**' for q in uploaded_files])}

## What's Next?
- Download your preferred quantization from the [Files tab](https://huggingface.co/{self.new_repo_id}/tree/main)
- Use with [llama.cpp](https://github.com/ggml-org/llama.cpp), [Ollama](https://ollama.ai/), or any GGUF-compatible inference engine
- Star the repo if you find it useful! ⭐

---
*This notification was sent automatically by [GGUF Forge](https://gguforge.com)*
"""

            loop = asyncio.get_event_loop()
            from huggingface_hub import HfApi
            sync_api = HfApi(token=self.hf_token)

            await loop.run_in_executor(
                None,
                lambda: sync_api.create_discussion(
                    repo_id=self.new_repo_id,
                    repo_type="model",
                    title=discussion_title,
                    description=discussion_body
                )
            )
            await self.log(f"  ✓ Notification sent to @{self.requested_by}")
        except Exception as e:
            # Non-fatal - don't fail the whole job if notification fails
            await self.log(f"  ⚠ Could not notify requester: {str(e)[:100]}")

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
        """Update and broadcast transfer progress for a file.

        Always broadcasts over WebSocket (cheap). Persists the snapshot to the
        DB on ~1Hz throttle or whenever a file hits a terminal state, so a
        page reload mid-transfer can rehydrate the per-file UI.
        """
        import time as _time
        self.transfer_files[filename] = {
            "name": filename,
            "progress": progress,
            "size": size,
            "speed": speed
        }
        self._last_transfer_type = transfer_type

        files_list = list(self.transfer_files.values())
        await broadcast_transfer_progress(self.model_id, transfer_type, files_list)

        # Persist a snapshot for reload-safety. Throttled to ~1Hz; terminal
        # states (100% / failed) always flush so the UI re-renders correctly.
        terminal = progress in (100, -1)
        now = _time.monotonic()
        if terminal or (now - self._last_transfer_persist) >= 1.0:
            self._last_transfer_persist = now
            snapshot = json.dumps({"type": transfer_type, "files": files_list})
            try:
                await self._update_db(transfer_state=snapshot)
            except Exception:
                pass  # never let persistence kill the transfer

    async def clear_transfer_progress(self):
        """Clear transfer progress tracking (both memory and DB snapshot)."""
        self.transfer_files = {}
        self._last_transfer_persist = 0.0
        try:
            await self._update_db(transfer_state="")
        except Exception:
            pass

    async def check_disk_space(self, required_gb: float):
        loop = asyncio.get_event_loop()
        total, used, free = await loop.run_in_executor(None, shutil.disk_usage, CACHE_DIR)
        free_gb = free / (2**30)

        import logging
        logger = logging.getLogger("GGUF_Forge")
        logger.info(f"check_disk_space called: ignore_space_check={self.ignore_space_check}, required={required_gb:.1f}GB, available={free_gb:.1f}GB")

        if self.ignore_space_check:
            await self.log(f"  ⚠ Space check BYPASSED by admin (Available: {free_gb:.1f}GB)")
            await self.log(f"  ⚠ Original requirement was: {required_gb:.1f}GB")
            await self.log(f"  ℹ Sequential processing requires much less space than conservative estimate")
            return
        
        await self.log(f"  Disk space check: Need {required_gb:.1f}GB, Available {free_gb:.1f}GB")
        if free_gb < required_gb:
            raise Exception(f"Insufficient disk space. Required: {required_gb:.1f}GB, Available: {free_gb:.1f}GB")
        await self.log(f"  ✓ Sufficient disk space")

    async def get_repo_file_sizes(self) -> dict:
        """Fetch per-file byte sizes for the HF repo. Returns {rfilename: size_bytes}.

        Cached on the instance so we only hit the HF metadata API once per run.
        """
        if getattr(self, "_repo_file_sizes", None) is not None:
            return self._repo_file_sizes
        try:
            hf_token = os.getenv("HF_TOKEN")
            api = HfApi(token=hf_token)
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                lambda: api.model_info(self.hf_repo_id, files_metadata=True)
            )
            sizes = {}
            for sib in (info.siblings or []):
                name = getattr(sib, 'rfilename', None) or getattr(sib, 'filename', None)
                size = getattr(sib, 'size', None)
                if name and size:
                    sizes[name] = int(size)
            self._repo_file_sizes = sizes
            return sizes
        except Exception as e:
            await self.log(f"  ⚠ Could not fetch repo file sizes: {e}")
            self._repo_file_sizes = {}
            return {}

    async def get_model_size_gb(self) -> float:
        """Get total model size in GB. Reuses get_repo_file_sizes()."""
        sizes = await self.get_repo_file_sizes()
        if not sizes:
            return 10.0  # Fallback when API failed
        total_bytes = sum(sizes.values())
        return total_bytes / (2**30)

    async def _poll_file_progress(self, short_name: str, dest_path: Path, total_size: int, transfer_type: str = "download"):
        """Watch dest_path (and any *.incomplete sibling) and broadcast live %/speed.

        Runs until cancelled. Caps at 99% so the final 100% tick is the one
        emitted by the caller after the download executor returns.
        """
        import time as _time
        last_bytes = 0
        last_t = _time.monotonic()
        ema_rate = 0.0

        # hf_hub_download may write to either `<dest>` directly or a sibling
        # *.incomplete temp file depending on hub-version & symlink mode. We
        # try the final path first, then glob siblings.
        def _current_bytes() -> int:
            try:
                if dest_path.exists():
                    return dest_path.stat().st_size
            except OSError:
                pass
            try:
                parent = dest_path.parent
                stem = dest_path.name
                best = 0
                for candidate in parent.glob(stem + "*.incomplete"):
                    try:
                        sz = candidate.stat().st_size
                        if sz > best:
                            best = sz
                    except OSError:
                        continue
                return best
            except Exception:
                return 0

        try:
            while True:
                await asyncio.sleep(0.5)
                bytes_now = _current_bytes()
                now = _time.monotonic()
                dt = max(now - last_t, 1e-3)
                inst_rate = max(0.0, (bytes_now - last_bytes) / dt)
                # Exponential moving average for a less jittery speed display.
                ema_rate = inst_rate if ema_rate == 0 else (0.7 * ema_rate + 0.3 * inst_rate)
                last_bytes = bytes_now
                last_t = now

                if total_size > 0:
                    pct = min(99, int(100 * bytes_now / total_size))
                    size_str = f"{_humanize_bytes(bytes_now)} / {_humanize_bytes(total_size)}"
                else:
                    pct = 0
                    size_str = _humanize_bytes(bytes_now)

                speed_str = f"{_humanize_bytes(ema_rate)}/s" if ema_rate > 0 else ""
                await self.update_transfer_progress(short_name, pct, size_str, speed_str, transfer_type)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Don't let a poller bug kill the download.
            import logging
            logging.getLogger("GGUF_Forge").debug(f"_poll_file_progress({short_name}): {e}")

    async def cleanup(self):
        """Remove all downloaded and generated files."""
        await self.log("Starting cleanup...")
        loop = asyncio.get_event_loop()
        try:
            # Remove downloaded model directory (NEVER delete a user-provided local source)
            if self.local_source_path:
                await self.log(f"Preserving local source: {self.local_source_path}")
            elif self.model_dir and Path(self.model_dir).exists():
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

            # Remove mmproj artifact from cache (skip if it was moved into a local output dir).
            if self.mmproj_path and self.mmproj_path.exists():
                try:
                    if CACHE_DIR in self.mmproj_path.parents:
                        await self.log(f"Removing mmproj file: {self.mmproj_path}")
                        await loop.run_in_executor(None, lambda p=self.mmproj_path: p.unlink(missing_ok=True))
                except Exception as e:
                    await self.log(f"  ⚠ Failed to remove mmproj: {e}")

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
            initial_progress = 0
            if self.resume_mode and self.completed_quants and self.quants_to_run:
                initial_progress = min(89, 50 + int(len(self.completed_quants) / len(self.quants_to_run) * 40))
            await self.progress(initial_progress)
            await self.log("━━━ GGUF Forge Pipeline Started ━━━")
            await self.log(f"Job ID: {self.model_id}")
            await self.log(f"Model: {self.hf_repo_id}")
            await self.log(f"Version: {await get_app_version()}")
            await self.log("")
            
            # 1. Setup Llama
            self.check_terminated()
            self.start_step("setup")
            await self.log("▶ STEP 1: Setting up llama.cpp...")
            # Re-read config so admin UI changes apply without restart
            await managers.refresh_llama_config()
            await self.log(f"  Repo: {managers.LLAMA_CPP_REPO}")
            await self.log(f"  Dir:  {managers.LLAMA_CPP_DIR}")
            await self.log("  Checking llama.cpp installation...")
            if self.force_llama_update:
                await self.log("  Force update enabled - will fetch latest llama.cpp commit...")
            await LlamaCppManager.clone_repo(force=self.force_llama_update)
            self.check_terminated()
            await self.log("  Building llama.cpp (this may take a while)...")
            await LlamaCppManager.build()
            quantize_bin = LlamaCppManager.get_quantize_path()
            await self.log(f"  ✓ llama-quantize ready: {quantize_bin.name}")
            self.end_step("setup")
            await self.progress(10)
            await self.log("")

            # Check if the intermediate GGUF file already exists (crash recovery).
            # In direct-outtype mode, the converter emits the final artifact instead
            # of an FP16 intermediate — name it after the chosen outtype so it
            # doesn't clash with an FP16 file from a previous run.
            slug = self.hf_repo_id.replace('/', '-')
            if self.is_direct_outtype:
                self.fp16_path = CACHE_DIR / f"{slug}-{self.convert_outtype}.gguf"
            else:
                self.fp16_path = CACHE_DIR / f"{slug}-f16.gguf"
            fp16_exists = self.fp16_path.exists() and self.fp16_path.stat().st_size > 0

            # 2. Download (skip if FP16 already exists, or use local source)
            self.check_terminated()
            if fp16_exists and self.resume_mode:
                await self.log("▶ STEP 2: Download SKIPPED (FP16 file exists from previous run)")
                await self.log(f"  ✓ Using existing FP16 file: {self.fp16_path.name}")
                await self.log("")
            elif self.local_source_path:
                self.start_step("download")
                await self.status("downloading")
                await self.log("▶ STEP 2: Using local model (no download needed)")
                await self.log(f"  Source: {self.local_source_path}")
                local_path = Path(self.local_source_path)
                if not local_path.exists():
                    raise Exception(f"Local source path does not exist: {self.local_source_path}")
                if not (local_path / "config.json").is_file():
                    raise Exception(f"Local source missing config.json: {self.local_source_path}")
                self.model_dir = str(local_path)
                await self.log(f"  ✓ Local source ready (originals will be preserved)")
                self.end_step("download")
                await self.progress(30)
                await self.log("")
            else:
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
                await self.clear_transfer_progress()

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

                    # Fetch per-file sizes once for live progress + size display
                    file_sizes = await self.get_repo_file_sizes()

                    # Download files with progress tracking
                    local_dir = CACHE_DIR / self.hf_repo_id
                    local_dir.mkdir(parents=True, exist_ok=True)

                    total_files = len(download_files)
                    for idx, filename in enumerate(download_files):
                        self.check_terminated()
                        short_name = filename.split('/')[-1] if '/' in filename else filename
                        total_size = file_sizes.get(filename, 0)
                        size_label = _humanize_bytes(total_size) if total_size else ""

                        # Initialize progress for this file
                        await self.update_transfer_progress(short_name, 0, size_label, "Starting...", "download")

                        # Spawn the disk poller so % / bytes / speed tick live
                        dest_path = local_dir / filename
                        poller = asyncio.create_task(
                            self._poll_file_progress(short_name, dest_path, total_size, "download")
                        )

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
                            # Mark as complete (overrides whatever the poller last reported)
                            await self.update_transfer_progress(short_name, 100, size_label, "Complete", "download")
                        except Exception as e:
                            await self.log(f"  ⚠ Failed to download {short_name}: {e}")
                            await self.update_transfer_progress(short_name, -1, size_label, "Failed", "download")
                        finally:
                            poller.cancel()
                            try:
                                await poller
                            except (asyncio.CancelledError, Exception):
                                pass

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
                await self.clear_transfer_progress()
                await broadcast_transfer_progress(self.model_id, "download", [])

                await self.log(f"  ✓ Downloaded to {self.model_dir}")
                self.end_step("download")
                await self.progress(30)
                await self.log("")

            # 3. Convert to FP16 (skip if FP16 already exists)
            self.check_terminated()
            if fp16_exists and self.resume_mode:
                await self.log("▶ STEP 3: Conversion SKIPPED (FP16 file exists from previous run)")
                await self.log(f"  ✓ Using existing FP16 file: {self.fp16_path.name}")
                # On resume, try to recover an existing mmproj artifact from cache.
                slug = self.hf_repo_id.replace('/', '-')
                existing_mmproj = CACHE_DIR / f"mmproj-{slug}-f16.gguf"
                if existing_mmproj.exists() and existing_mmproj.stat().st_size > 0:
                    self.is_vision_model = True
                    self.mmproj_path = existing_mmproj
                    await self.log(f"  ✓ Found existing mmproj from previous run: {existing_mmproj.name}")
                elif self.model_dir and Path(self.model_dir).exists():
                    # Source still available (e.g. local_source mode) — regenerate.
                    self.is_vision_model = self._detect_vision_model()
                    if self.is_vision_model:
                        await self.log("  Vision model detected — regenerating mmproj projector...")
                        try:
                            mmproj_path = await self._generate_mmproj()
                        except Exception as e:
                            await self.log(f"  ⚠ mmproj generation raised: {e}")
                            mmproj_path = None
                        if mmproj_path and mmproj_path.exists() and mmproj_path.stat().st_size > 0:
                            self.mmproj_path = mmproj_path
                            await self.log(f"  ✓ mmproj generated: {mmproj_path.name}")
                await self.log("")
            else:
                self.start_step("convert")
                await self.status("converting")
                outtype_arg = self.convert_outtype if self.is_direct_outtype else "f16"
                if self.is_direct_outtype:
                    await self.log(
                        f"▶ STEP 3: Converting directly to GGUF format ({outtype_arg.upper()}) — llama-quantize will be skipped..."
                    )
                else:
                    await self.log("▶ STEP 3: Converting to GGUF format (FP16)...")
                convert_script = managers.LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

                cmd = [sys.executable, str(convert_script), str(self.model_dir), "--outfile", str(self.fp16_path), "--outtype", outtype_arg]
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

                if self.is_direct_outtype:
                    await self.log(
                        f"  ✓ Direct {self.convert_outtype.upper()} conversion complete: {self.fp16_path.name}"
                    )
                else:
                    await self.log(f"  ✓ FP16 conversion complete: {self.fp16_path.name}")
                self.end_step("convert")
                await self.progress(50)

                # 3b. Detect & generate multimodal projector (mmproj) BEFORE deleting safetensors
                self.check_terminated()
                self.is_vision_model = self._detect_vision_model()
                if self.is_vision_model:
                    await self.log("▶ STEP 3b: Vision model detected — generating mmproj projector...")
                    try:
                        mmproj_path = await self._generate_mmproj()
                    except Exception as e:
                        await self.log(f"  ⚠ mmproj generation raised: {e}")
                        mmproj_path = None
                    if mmproj_path and mmproj_path.exists() and mmproj_path.stat().st_size > 0:
                        self.mmproj_path = mmproj_path
                        await self.log(f"  ✓ mmproj generated: {mmproj_path.name}")
                    else:
                        await self.log("  ⚠ mmproj generation produced no usable file — continuing without it")
                    await self.log("")

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
                transfer_modes = []
                if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1":
                    transfer_modes.append("hf_transfer")
                if os.getenv("HF_XET_HIGH_PERFORMANCE") == "1":
                    transfer_modes.append("hf_xet high-performance")
                await self.log(f"  HF transfer acceleration: {', '.join(transfer_modes) if transfer_modes else 'default'}")

            # Local-only mode: prepare output directory inside the source folder, skip HF upload
            local_output_dir = None
            if self.keep_local_only:
                if not self.local_source_path:
                    await self.log("  ⚠ keep_local_only requires a local source — falling back to upload mode")
                    self.keep_local_only = False
                else:
                    local_output_dir = Path(self.local_source_path) / "gguf"
                    local_output_dir.mkdir(parents=True, exist_ok=True)
                    await self.log(f"  Local-only mode: outputs will be saved to {local_output_dir}")

            if self.keep_local_only:
                # Skip HF repo creation entirely
                self.new_repo_id = None
            elif self.hf_token:
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

            # Determine which quants to process (use custom list if set, skip already completed ones)
            quants_to_process = [q for q in self.quants_to_run if q not in self.completed_quants]

            # Sort quants by priority order
            priority_order = await get_quant_priority_order()
            priority_map = {q: i for i, q in enumerate(priority_order)}
            quants_to_process.sort(key=lambda q: priority_map.get(q, 999))

            await self.log(f"  Quant priority order: {', '.join(quants_to_process)}")

            if self.resume_mode and self.completed_quants:
                await self.log(f"  📋 Resume mode: {len(self.completed_quants)} quants already completed")
                await self.log(f"     Already done: {', '.join(self.completed_quants)}")
                await self.log(f"     Remaining: {len(quants_to_process)} quants to process")
                uploaded_files = list(self.completed_quants)  # Count already uploaded as successful
                await self.log("")
            elif len(self.quants_to_run) < len(QUANTS):
                # User requested specific quants
                await self.log(f"  📋 Custom quants requested: {', '.join(self.quants_to_run)}")
                await self.log("")

            uploaded_files_lock = asyncio.Lock()
            readme_update_lock = asyncio.Lock()

            async def remember_completed_quant(q_type: str, *, save_resume: bool = False, update_readme: bool = False):
                """Track a finished quant and optionally refresh the repo README immediately."""
                async with uploaded_files_lock:
                    if q_type not in uploaded_files:
                        uploaded_files.append(q_type)

                if save_resume:
                    await self.save_completed_quant(q_type)

                if update_readme and self.hf_token and self.new_repo_id:
                    async with readme_update_lock:
                        self.check_terminated()
                        async with uploaded_files_lock:
                            snapshot = list(uploaded_files)
                        await self.upload_status_readme(quant_base_name, snapshot)

            if self.hf_token and self.new_repo_id:
                await self.status("uploading")
                await self.upload_status_readme(quant_base_name, uploaded_files)

            # Upload (or relocate) the mmproj projector for vision models, once, before quants run.
            if self.mmproj_path and self.mmproj_path.exists():
                mmproj_name = f"mmproj-{quant_base_name}-f16.gguf"
                if self.keep_local_only and local_output_dir is not None:
                    try:
                        target_path = local_output_dir / mmproj_name
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, lambda: shutil.move(str(self.mmproj_path), str(target_path))
                        )
                        self.mmproj_uploaded_name = mmproj_name
                        self.mmproj_path = target_path
                        await self.log(f"  ✓ mmproj saved locally: {target_path}")
                    except Exception as e:
                        await self.log(f"  ⚠ Failed to move mmproj to local output: {e}")
                elif self.hf_token and self.new_repo_id:
                    await self.status("uploading")
                    file_size = self.mmproj_path.stat().st_size
                    size_str = f"{file_size / (1024**3):.2f}GB"
                    await self.update_transfer_progress(mmproj_name, 0, size_str, "Uploading...", "upload")
                    ok = await self._upload_with_retry(
                        lambda p=self.mmproj_path, n=mmproj_name: self.api.upload_file(
                            path_or_fileobj=p,
                            path_in_repo=n,
                            repo_id=self.new_repo_id,
                            repo_type="model"
                        ),
                        description="mmproj upload",
                    )
                    if ok:
                        await self.update_transfer_progress(mmproj_name, 100, size_str, "Complete", "upload")
                        await self.log(f"  ✓ mmproj uploaded to HuggingFace: {mmproj_name}")
                        self.mmproj_uploaded_name = mmproj_name
                        # mmproj is kept on disk for resume safety; cache cleanup at end of run handles it.
                    else:
                        await self.update_transfer_progress(mmproj_name, -1, size_str, "Failed", "upload")
                        await self.log(f"  ⚠ mmproj upload abandoned after retries (file kept at {self.mmproj_path})")
                    self.transfer_files.pop(mmproj_name, None)
                    await broadcast_transfer_progress(self.model_id, "upload", list(self.transfer_files.values()))

            total_quants = len(self.quants_to_run)
            completed_count = len(self.completed_quants)
            
            # Detect CPU cores
            total_cores = multiprocessing.cpu_count()
            # Calculate threads per job
            num_parallel = max(1, min(PARALLEL_QUANT_JOBS, len(quants_to_process)))
            threads_per_job = max(1, total_cores // num_parallel)
            
            await self.log(f"  CPU cores: {total_cores} total")
            await self.log(f"  Parallel jobs: {num_parallel}")
            await self.log(f"  Threads per job: {threads_per_job}")
            if self.is_direct_outtype:
                await self.log(
                    f"  Mode: Direct convert outtype ({self.convert_outtype.upper()}) — skipping llama-quantize"
                )
            else:
                await self.log(f"  Mode: Parallel quantize ({num_parallel} at a time)")
            await self.log("")
            
            # Semaphore to limit parallel quantization jobs
            semaphore = asyncio.Semaphore(num_parallel)
            
            async def process_single_quant(q_type: str, overall_idx: int):
                async with semaphore:
                    self.check_terminated()
                    await self.status("quantizing")
                    await self.log(f"  [{overall_idx}/{total_quants}] Starting {q_type}...")

                    q_path = CACHE_DIR / f"{quant_base_name}.{q_type}.gguf"
                    quant_start = time.time()

                    try:
                        if self.is_direct_outtype:
                            # Direct-outtype mode: the converter already produced the
                            # final compact GGUF as self.fp16_path. Reuse it as q_path
                            # instead of running llama-quantize on it.
                            if not (self.fp16_path and self.fp16_path.exists()
                                    and self.fp16_path.stat().st_size > 0):
                                await self.log(
                                    f"      ⚠ {q_type} expected direct-outtype file missing at {self.fp16_path}"
                                )
                                return
                            q_path = self.fp16_path
                            quant_duration = time.time() - quant_start
                            self.quant_times.append((q_type, quant_duration))
                            await self.log(
                                f"      ✓ {q_type} ready from direct convert ({self.format_duration(quant_duration)})"
                            )
                        else:
                            # === QUANTIZE ===
                            env = os.environ.copy()
                            if quantize_bin and quantize_bin.parent:
                                current_ld = env.get('LD_LIBRARY_PATH', '')
                                env['LD_LIBRARY_PATH'] = f"{quantize_bin.parent}:{current_ld}"

                            # Apply threads constraint
                            env['OMP_NUM_THREADS'] = str(threads_per_job)
                            env['MKL_NUM_THREADS'] = str(threads_per_job)
                            env['OPENBLAS_NUM_THREADS'] = str(threads_per_job)

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
                                return

                            self.quant_times.append((q_type, quant_duration))
                            await self.log(f"      ✓ {q_type} Quantized ({self.format_duration(quant_duration)})")
                        
                        # Ensure output is a single GGUF file (merge shards if needed)
                        merged_path = await self.ensure_unsharded_gguf(q_path, q_type)
                        if not merged_path:
                            return
                        q_path = merged_path
                        
                        # === UPLOAD or KEEP-LOCAL ===
                        if self.keep_local_only and local_output_dir is not None:
                            self.check_terminated()
                            filename = f"{quant_base_name}.{q_type}.gguf"
                            target_path = local_output_dir / filename
                            try:
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(
                                    None, lambda: shutil.move(str(q_path), str(target_path))
                                )
                                await self.log(f"      ✓ {q_type} Saved locally: {target_path}")
                                await remember_completed_quant(q_type, save_resume=True)
                            except Exception as e:
                                await self.log(f"      ⚠ {q_type} Failed to move to local output: {e}")
                                return
                        elif self.hf_token and self.new_repo_id:
                            self.check_terminated()
                            await self.status("uploading")

                            filename = f"{quant_base_name}.{q_type}.gguf"
                            file_size = q_path.stat().st_size if q_path.exists() else 0
                            size_str = f"{file_size / (1024**3):.2f}GB" if file_size > 0 else ""

                            await self.update_transfer_progress(filename, 0, size_str, "Uploading...", "upload")

                            ok = await self._upload_with_retry(
                                lambda p=q_path, n=filename: self.api.upload_file(
                                    path_or_fileobj=p,
                                    path_in_repo=n,
                                    repo_id=self.new_repo_id,
                                    repo_type="model"
                                ),
                                description=f"{q_type} upload",
                            )
                            if not ok:
                                await self.update_transfer_progress(filename, -1, size_str, "Failed", "upload")
                                await self.log(f"      ⚠ {q_type} Upload abandoned after retries (file kept for manual recovery: {q_path})")
                                # Don't delete the file if upload failed - keep for potential manual recovery or retry
                                return

                            await self.update_transfer_progress(filename, 100, size_str, "Complete", "upload")
                            await self.log(f"      ✓ {q_type} Uploaded to HuggingFace")

                            # Save progress for resume and refresh README so the repo shows each new quant immediately.
                            await remember_completed_quant(q_type, save_resume=True, update_readme=True)

                            # === DELETE QUANT FILE (only after successful upload) ===
                            try:
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, lambda: q_path.unlink(missing_ok=True))
                            except Exception as e:
                                await self.log(f"      ⚠ {q_type} Failed to delete: {e}")
                        else:
                            await self.log(f"      ℹ {q_type} Skipping upload (no HF token)")
                            await remember_completed_quant(q_type)
                            try:
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, lambda: q_path.unlink(missing_ok=True))
                            except Exception as e:
                                await self.log(f"      ⚠ {q_type} Failed to delete: {e}")
                        
                        # Clear transfer progress for this file
                        self.transfer_files.pop(f"{quant_base_name}.{q_type}.gguf", None)
                        await broadcast_transfer_progress(self.model_id, "upload", list(self.transfer_files.values()))
                        
                    except Exception as e:
                        await self.log(f"      ⚠ {q_type} error: {e}")
                    
                    # Update overall progress
                    current_completed = len(uploaded_files)
                    step_progress = 50 + int(current_completed / total_quants * 40)
                    await self.progress(step_progress)

            # Create tasks for all quants
            tasks = []
            for q_type in quants_to_process:
                overall_idx = self.quants_to_run.index(q_type) + 1
                tasks.append(process_single_quant(q_type, overall_idx))
            
            # Run all tasks concurrently with semaphore limit
            if tasks:
                await asyncio.gather(*tasks)
            
            self.end_step("quantize")
            await self.log("")
            await self.log(f"  ✓ Completed {len(uploaded_files)}/{total_quants} quants")
            
            await self.progress(90)
            
            await self.log("")

            # 5. Readme
            if self.hf_token and uploaded_files and self.new_repo_id:
                await self.status("uploading")
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

                # Build requester section
                requester_section = ""
                if self.requested_by:
                    requester_section = f"""
## 🙏 Requested By

This conversion was requested by [@{self.requested_by}](https://huggingface.co/{self.requested_by}).
"""

                # Build mmproj / vision section if a multimodal projector was uploaded
                mmproj_section = ""
                vision_tag = ""
                if self.mmproj_uploaded_name:
                    vision_tag = "\n- vision"
                    mmproj_section = f"""
## Vision / Multimodal (mmproj)

This is a vision-capable model. To use image input with llama.cpp / llama-server, download **both** the language-model quant and the projector file:

- Projector: `{self.mmproj_uploaded_name}`

Example with `llama-server`:

```
llama-server \\
  -m {quant_base_name}.Q4_K_M.gguf \\
  --mmproj {self.mmproj_uploaded_name}
```
"""

                readme_content = f"""---
tags:
- gguf
- llama.cpp
- quantization{vision_tag}
base_model: {self.hf_repo_id}
---

# {quant_base_name}-GGUF

This model was converted to GGUF format from [`{self.hf_repo_id}`](https://huggingface.co/{self.hf_repo_id}) using GGUF Forge.
{requester_section}
## Quants
The following quants are available:
{', '.join(uploaded_files)}
{mmproj_section}
## Ollama Support
Full Ollama support is provided by merging any sharded GGUF output into a single file after quantization.

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
 - 📦 llama.cpp (quantization engine): [GitHub](https://github.com/ggml-org/llama.cpp)
 - 💬 Community & Support: [Discord](https://discord.gg/4vafUgVX3a)


---
*Converted automatically by [GGUF Forge](https://gguforge.com) {app_version}*

"""
                ok = await self._upload_with_retry(
                    lambda: self.api.upload_file(
                        path_or_fileobj=readme_content.encode('utf-8'),
                        path_in_repo="README.md",
                        repo_id=self.new_repo_id,
                        repo_type="model"
                    ),
                    description="Final README upload",
                )
                if ok:
                    await self.log(f"  ✓ README uploaded")
                else:
                    await self.log(f"  ⚠ README upload failed after retries")
                await self.log("")

                # Notify the requester via HuggingFace discussion
                if self.requested_by and self.new_repo_id:
                    await self.notify_requester(quant_base_name, uploaded_files)

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

            # Clear any persisted transfer snapshot so a finished job's card
            # doesn't render stale file-by-file rows on reload.
            try:
                await self.clear_transfer_progress()
            except Exception:
                pass

            # Always cleanup files
            await self.log("")
            await self.log("▶ Cleanup...")
            await self.cleanup()
