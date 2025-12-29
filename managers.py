"""
Manager classes for llama.cpp and HuggingFace operations.
"""
import os
import shutil
import asyncio
import logging
import platform
import subprocess
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi

logger = logging.getLogger("GGUF_Forge")

# These will be set by main app
LLAMA_CPP_DIR = None
BASE_DIR = None

# Thread pool for blocking operations
_executor = ThreadPoolExecutor(max_workers=4)


def set_paths(base_dir: Path, llama_cpp_dir: Path):
    """Set the paths for managers."""
    global BASE_DIR, LLAMA_CPP_DIR
    BASE_DIR = base_dir
    LLAMA_CPP_DIR = llama_cpp_dir


class LlamaCppManager:
    @staticmethod
    def is_installed() -> bool:
        return (LLAMA_CPP_DIR / "CMakeLists.txt").exists()
    
    @staticmethod
    def check_tool(tool_name: str) -> bool:
        """Check if a tool is available in PATH."""
        return shutil.which(tool_name) is not None
    
    @staticmethod
    async def has_nvidia_gpu() -> bool:
        """Check if NVIDIA GPU is available (async)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            returncode = await proc.wait()
            return returncode == 0
        except FileNotFoundError:
            return False
    
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
        """Build llama.cpp using CMake with optional CUDA support."""
        logger.info("Building llama.cpp...")
        system = platform.system()
        
        # Check if already built - skip if llama-quantize exists
        try:
            existing = LlamaCppManager.get_quantize_path()
            if existing.exists():
                logger.info(f"llama.cpp already built, skipping. Found: {existing}")
                return
        except FileNotFoundError:
            pass  # Not built yet, continue with build
        
        # Check if cmake is available
        if not LlamaCppManager.check_tool("cmake"):
            raise Exception("CMake is not installed or not in PATH. Please install CMake.")
        
        # Check for CUDA support
        has_cuda = await LlamaCppManager.has_nvidia_gpu()
        if has_cuda:
            logger.info("NVIDIA GPU detected, building with CUDA support...")
        else:
            logger.info("No NVIDIA GPU detected, building CPU-only version...")
        
        build_dir = LLAMA_CPP_DIR / "build"
        build_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: CMake Configure
            cmake_args = [
                "cmake", "..",
                "-DLLAMA_CURL=OFF",
                "-DCMAKE_BUILD_TYPE=Release"
            ]
            
            # Add CUDA flag if available
            if has_cuda:
                cmake_args.append("-DGGML_CUDA=ON")
            
            # Windows-specific: use Release config
            if system == "Windows":
                cmake_args.extend(["-A", "x64"])
            
            logger.info(f"Running CMake configure: {' '.join(cmake_args)}")
            
            proc = await asyncio.create_subprocess_exec(
                *cmake_args,
                cwd=build_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode != 0:
                error_output = stdout.decode() if stdout else "No output"
                logger.error(f"CMake configure failed:\n{error_output}")
                raise Exception(f"CMake configure failed. Output:\n{error_output[:2000]}")
            
            logger.info("CMake configure successful")
            
            # Step 2: CMake Build
            build_args = ["cmake", "--build", ".", "--config", "Release"]
            
            # Use multiple cores for faster builds
            if system != "Windows":
                import multiprocessing
                cores = multiprocessing.cpu_count()
                build_args.extend(["-j", str(cores)])
            else:
                build_args.extend(["--", "/m"])  # Parallel build for MSBuild
            
            logger.info(f"Running CMake build: {' '.join(build_args)}")
            
            proc = await asyncio.create_subprocess_exec(
                *build_args,
                cwd=build_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode != 0:
                error_output = stdout.decode() if stdout else "No output"
                logger.error(f"CMake build failed:\n{error_output}")
                raise Exception(f"CMake build failed. Output:\n{error_output[:2000]}")
            
            cuda_status = "with CUDA" if has_cuda else "CPU-only"
            logger.info(f"Build successful ({cuda_status}) on {system}")
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            raise Exception(
                f"Failed to build llama.cpp: {str(e)}\n"
                "Ensure build tools are installed:\n"
                "  - Windows: Visual Studio Build Tools + CMake\n"
                "  - Linux: build-essential, cmake, (nvidia-cuda-toolkit for GPU)"
            )

    @staticmethod
    def get_quantize_path() -> Path:
        """Find the llama-quantize executable."""
        system = platform.system()
        build_dir = LLAMA_CPP_DIR / "build"
        
        # Common paths based on CMake output
        if system == "Windows":
            candidates = [
                build_dir / "bin" / "Release" / "llama-quantize.exe",
                build_dir / "Release" / "llama-quantize.exe",
                build_dir / "bin" / "llama-quantize.exe",
                LLAMA_CPP_DIR / "build" / "llama-quantize.exe",
            ]
        else:
            candidates = [
                build_dir / "bin" / "llama-quantize",
                build_dir / "llama-quantize",
                LLAMA_CPP_DIR / "llama-quantize",
            ]
        
        for path in candidates:
            if path.exists():
                logger.info(f"Found llama-quantize at: {path}")
                return path
        
        # Fallback: recursive search
        pattern = "llama-quantize.exe" if system == "Windows" else "llama-quantize"
        found = list(LLAMA_CPP_DIR.rglob(pattern))
        if found:
            # Prefer executables in build directories
            for f in found:
                if "build" in str(f):
                    logger.info(f"Found llama-quantize at: {f}")
                    return f
            logger.info(f"Found llama-quantize at: {found[0]}")
            return found[0]
        
        raise FileNotFoundError(
            "llama-quantize executable not found. Build might have failed.\n"
            f"Searched in: {LLAMA_CPP_DIR}"
        )


class HuggingFaceManager:
    def __init__(self, token: Optional[str] = None):
        self.api = HfApi(token=token)

    async def search_models(self, query: str, limit: int = 10):
        """Search for models on HuggingFace (async)."""
        loop = asyncio.get_event_loop()
        models = await loop.run_in_executor(
            _executor,
            lambda: list(self.api.list_models(search=query, limit=limit, sort="likes", direction=-1))
        )
        return [{"id": m.modelId, "likes": m.likes} for m in models]

    async def check_exists(self, repo_id: str) -> bool:
        """Check if a model exists on HuggingFace (async)."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                _executor,
                lambda: self.api.model_info(repo_id)
            )
            return True
        except:
            return False


async def get_app_version() -> str:
    """Calculate app version based on git commit count * 0.1 (async)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-list", "--count", "HEAD",
            cwd=BASE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            commit_count = int(stdout.decode().strip())
            version = commit_count * 0.1
            return f"v{version:.1f}"
    except Exception:
        pass
    return "v0.1"  # Fallback

