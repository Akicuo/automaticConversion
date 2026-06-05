"""
Manager classes for llama.cpp and HuggingFace operations.
"""
import os
import re
import json
import shutil
import asyncio
import logging
import platform
import subprocess
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

# Prefer HuggingFace's accelerated transfer backends for large GGUF uploads.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi

logger = logging.getLogger("GGUF_Forge")

# These will be set by main app
LLAMA_CPP_DIR = None
LLAMA_CPP_REPO = None
BASE_DIR = None
# Fork-specific compact GGUF outtypes available to users at conversion time.
# Empty list means only the default FP16 + standard llama-quantize flow is exposed.
LLAMA_CPP_OUTTYPES: List[str] = []

# Defaults captured from env at startup so refresh can fall back to them
_ENV_LLAMA_CPP_DIR: Optional[Path] = None
_ENV_LLAMA_CPP_REPO: Optional[str] = None
_ENV_LLAMA_CPP_OUTTYPES: List[str] = []

DEFAULT_LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"

# Outtypes that are NOT compact quantizations — selecting one of these is
# equivalent to today's "FP16 then llama-quantize" flow, not direct-output.
NON_COMPACT_OUTTYPES = frozenset({"f16", "bf16", "f32"})

_OUTTYPE_RE = re.compile(r"^[a-z0-9_]+$")


def normalize_outtypes(values) -> List[str]:
    """Normalize a list/iterable of outtype strings: trim, lowercase, dedupe, validate shape."""
    if not values:
        return []
    seen = []
    for v in values:
        if not isinstance(v, str):
            continue
        s = v.strip().lower()
        if not s or not _OUTTYPE_RE.match(s):
            continue
        if s in seen:
            continue
        seen.append(s)
    return seen

# Thread pool for blocking operations
_executor = ThreadPoolExecutor(max_workers=4)


def set_paths(base_dir: Path, llama_cpp_dir: Path, llama_cpp_repo: Optional[str] = None,
              llama_cpp_outtypes: Optional[List[str]] = None):
    """Set the paths/repo for managers. Values here are treated as the env/default fallback."""
    global BASE_DIR, LLAMA_CPP_DIR, LLAMA_CPP_REPO, LLAMA_CPP_OUTTYPES
    global _ENV_LLAMA_CPP_DIR, _ENV_LLAMA_CPP_REPO, _ENV_LLAMA_CPP_OUTTYPES
    BASE_DIR = base_dir
    LLAMA_CPP_DIR = llama_cpp_dir
    LLAMA_CPP_REPO = llama_cpp_repo or DEFAULT_LLAMA_CPP_REPO
    LLAMA_CPP_OUTTYPES = normalize_outtypes(llama_cpp_outtypes)
    _ENV_LLAMA_CPP_DIR = llama_cpp_dir
    _ENV_LLAMA_CPP_REPO = LLAMA_CPP_REPO
    _ENV_LLAMA_CPP_OUTTYPES = list(LLAMA_CPP_OUTTYPES)


async def refresh_llama_config():
    """Reload LLAMA_CPP_DIR, LLAMA_CPP_REPO, and LLAMA_CPP_OUTTYPES from DB (with env/default fallback).

    Priority: app_config DB row > env-supplied value (captured in set_paths) > default.
    Safe to call before init_db — falls back to env values if the DB read fails.
    """
    global LLAMA_CPP_DIR, LLAMA_CPP_REPO, LLAMA_CPP_OUTTYPES
    try:
        from database import get_app_config
        db_dir = await get_app_config("llama_cpp_dir")
        db_repo = await get_app_config("llama_cpp_repo")
        db_outtypes = await get_app_config("llama_cpp_outtypes")
    except Exception as e:
        logger.debug(f"refresh_llama_config: DB read skipped ({e})")
        db_dir = None
        db_repo = None
        db_outtypes = None

    LLAMA_CPP_DIR = Path(db_dir).expanduser() if db_dir else _ENV_LLAMA_CPP_DIR
    LLAMA_CPP_REPO = db_repo or _ENV_LLAMA_CPP_REPO or DEFAULT_LLAMA_CPP_REPO

    parsed_outtypes: List[str] = []
    if db_outtypes:
        try:
            parsed_outtypes = json.loads(db_outtypes)
            if not isinstance(parsed_outtypes, list):
                parsed_outtypes = []
        except (ValueError, TypeError):
            parsed_outtypes = []
    LLAMA_CPP_OUTTYPES = normalize_outtypes(parsed_outtypes) if parsed_outtypes else list(_ENV_LLAMA_CPP_OUTTYPES)


async def get_current_origin(folder: Optional[Path] = None) -> Optional[str]:
    """Return `git config --get remote.origin.url` for the given folder, or None."""
    target = folder or LLAMA_CPP_DIR
    if not target or not (Path(target) / ".git").exists():
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "config", "--get", "remote.origin.url",
            cwd=str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            url = stdout.decode().strip()
            return url or None
    except Exception:
        pass
    return None


def _normalize_repo_url(url: str) -> str:
    """Normalize a git URL for comparison (strip .git, trailing slash, lowercase host)."""
    if not url:
        return ""
    u = url.strip()
    if u.endswith(".git"):
        u = u[:-4]
    if u.endswith("/"):
        u = u[:-1]
    normalized = u.lower()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.split(":", 1)[1]
    aliases = {
        "https://github.com/ggerganov/llama.cpp": "https://github.com/ggml-org/llama.cpp",
    }
    return aliases.get(normalized, normalized)


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
    async def clone_repo(force: bool = False):
        """Clone or update llama.cpp repository.

        Args:
            force: If True, forcefully fetch and reset to latest remote commit,
                   discarding any local changes. If False, perform normal git pull.
        """
        # If the folder already has a .git, make sure its origin matches the
        # configured LLAMA_CPP_REPO. Refuse to touch unrelated checkouts.
        if (LLAMA_CPP_DIR / ".git").exists():
            current_origin = await get_current_origin(LLAMA_CPP_DIR)
            if current_origin and _normalize_repo_url(current_origin) != _normalize_repo_url(LLAMA_CPP_REPO):
                raise Exception(
                    f"Folder '{LLAMA_CPP_DIR}' is already a clone of '{current_origin}', "
                    f"but configured repo is '{LLAMA_CPP_REPO}'. Resolve manually "
                    f"(delete the folder, change the configured repo, or point LLAMA_CPP_DIR elsewhere)."
                )
            if current_origin and current_origin.strip() != LLAMA_CPP_REPO:
                logger.info(f"Updating llama.cpp origin URL from {current_origin} to {LLAMA_CPP_REPO}")
                proc = await asyncio.create_subprocess_exec(
                    "git", "remote", "set-url", "origin", LLAMA_CPP_REPO,
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    logger.warning(f"Failed to update llama.cpp origin URL: {stderr.decode(errors='replace')}")

        if LlamaCppManager.is_installed():
            if force:
                logger.info("llama.cpp already exists. Forcefully updating to latest commit...")
                
                # Step 1: Fetch all updates from origin
                logger.info("Fetching all updates from origin...")
                proc = await asyncio.create_subprocess_exec(
                    "git", "fetch", "--all",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    logger.error(f"Failed to fetch updates: {error_msg}")
                    raise Exception(f"Failed to fetch llama.cpp updates: {error_msg}")
                
                # Step 2: Get current branch name
                proc = await asyncio.create_subprocess_exec(
                    "git", "rev-parse", "--abbrev-ref", "HEAD",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    # Fallback to master/main if branch detection fails
                    branch = "master"
                    logger.warning(f"Could not detect branch, defaulting to {branch}")
                else:
                    branch = stdout.decode().strip()
                    logger.info(f"Current branch: {branch}")
                
                # Step 3: Hard reset to origin branch
                logger.info(f"Resetting to origin/{branch} (discarding local changes)...")
                proc = await asyncio.create_subprocess_exec(
                    "git", "reset", "--hard", f"origin/{branch}",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    logger.error(f"Failed to reset to origin/{branch}: {error_msg}")
                    raise Exception(f"Failed to reset llama.cpp: {error_msg}")
                
                # Step 4: Get current commit info
                proc = await asyncio.create_subprocess_exec(
                    "git", "log", "-1", "--oneline",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    commit_info = stdout.decode().strip()
                    logger.info(f"Updated to: {commit_info}")
                else:
                    logger.info("Force update complete")
            else:
                logger.info("llama.cpp already exists. Pulling latest...")
                proc = await asyncio.create_subprocess_exec(
                    "git", "pull",
                    cwd=LLAMA_CPP_DIR,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.wait()
        else:
            logger.info(f"Cloning llama.cpp from {LLAMA_CPP_REPO} into {LLAMA_CPP_DIR}...")
            LLAMA_CPP_DIR.parent.mkdir(parents=True, exist_ok=True)
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", LLAMA_CPP_REPO, str(LLAMA_CPP_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise Exception(f"Failed to clone llama.cpp from {LLAMA_CPP_REPO}: {error_msg}")

    @staticmethod
    def _decode_output(data: bytes) -> str:
        """Safely decode subprocess output, handling encoding issues."""
        if not data:
            return "No output"
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return data.decode('cp1252')  # Windows default
            except UnicodeDecodeError:
                return data.decode('utf-8', errors='replace')

    @staticmethod
    def _get_vs_env() -> dict:
        """Get Visual Studio environment variables for building."""
        # Common paths for VS Developer Command Prompt
        vs_where = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        vcvars_paths = [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        ]

        for vcvars in vcvars_paths:
            if Path(vcvars).exists():
                logger.info(f"Found VS environment: {vcvars}")
                # Run vcvars and capture environment
                try:
                    result = subprocess.run(
                        f'cmd /c ""{vcvars}" && set"',
                        capture_output=True,
                        shell=True,
                        text=True
                    )
                    if result.returncode == 0:
                        env = os.environ.copy()
                        for line in result.stdout.splitlines():
                            if '=' in line:
                                key, _, value = line.partition('=')
                                env[key] = value
                        return env
                except Exception as e:
                    logger.warning(f"Failed to get VS env from {vcvars}: {e}")

        return os.environ.copy()

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

        # Clean build directory if it exists but might be corrupted
        if build_dir.exists():
            cmake_cache = build_dir / "CMakeCache.txt"
            if cmake_cache.exists():
                logger.info("Existing build directory found, cleaning...")
                shutil.rmtree(build_dir, ignore_errors=True)

        build_dir.mkdir(exist_ok=True)

        try:
            # Step 1: CMake Configure
            cmake_args = [
                "cmake", "..",
                "-DGGML_CUDA=OFF" if not has_cuda else "-DGGML_CUDA=ON",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_NATIVE=OFF",
                "-Wno-dev",  # Suppress developer warnings
            ]

            # Platform-specific CMake generator and settings
            if system == "Windows":
                # Get Visual Studio environment
                build_env = LlamaCppManager._get_vs_env()

                # Try Visual Studio generators first (they work best with proper env)
                # If CUDA fails, fall back to CPU-only
                generators = [
                    ("Visual Studio 17 2022", "x64"),
                    ("Visual Studio 16 2019", "x64"),
                ]

                cmake_success = False
                last_error = ""

                # First try with CUDA if detected
                for generator, arch in generators:
                    logger.info(f"Trying CMake generator: {generator}")
                    test_args = cmake_args.copy()
                    test_args.extend(["-G", generator])
                    if arch:
                        test_args.extend(["-A", arch])

                    proc = await asyncio.create_subprocess_exec(
                        *test_args,
                        cwd=build_dir,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        env=build_env
                    )
                    stdout, _ = await proc.communicate()

                    if proc.returncode == 0:
                        cmake_success = True
                        logger.info(f"CMake configure successful with {generator}")
                        break
                    else:
                        last_error = LlamaCppManager._decode_output(stdout)
                        logger.warning(f"CMake failed with {generator}: {last_error[:500]}")
                        # Clean build dir for next attempt
                        if build_dir.exists():
                            shutil.rmtree(build_dir, ignore_errors=True)
                        build_dir.mkdir(exist_ok=True)

                # If CUDA build failed and we have CUDA, try CPU-only as fallback
                if not cmake_success and has_cuda:
                    logger.warning("CUDA build failed, falling back to CPU-only build...")
                    cmake_args_cpu = [
                        "cmake", "..",
                        "-DGGML_CUDA=OFF",
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_NATIVE=OFF",
                        "-Wno-dev",
                    ]

                    for generator, arch in generators:
                        logger.info(f"Retrying with CPU-only: {generator}")
                        test_args = cmake_args_cpu.copy()
                        test_args.extend(["-G", generator])
                        if arch:
                            test_args.extend(["-A", arch])

                        proc = await asyncio.create_subprocess_exec(
                            *test_args,
                            cwd=build_dir,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                            env=build_env
                        )
                        stdout, _ = await proc.communicate()

                        if proc.returncode == 0:
                            cmake_success = True
                            has_cuda = False  # Update flag since we're building CPU-only
                            logger.info(f"CMake configure successful with {generator} (CPU-only)")
                            break
                        else:
                            last_error = LlamaCppManager._decode_output(stdout)
                            logger.warning(f"CMake failed with {generator}: {last_error[:500]}")
                            if build_dir.exists():
                                shutil.rmtree(build_dir, ignore_errors=True)
                            build_dir.mkdir(exist_ok=True)

                if not cmake_success:
                    raise Exception(f"CMake configure failed with all generators. Last error:\n{last_error[:4000]}")
            else:
                # Linux/macOS - simpler approach
                build_env = os.environ.copy()
                cmake_args.extend(["-DGGML_BLAS=OFF"])

                logger.info(f"Running CMake configure: {' '.join(cmake_args)}")

                proc = await asyncio.create_subprocess_exec(
                    *cmake_args,
                    cwd=build_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
                stdout, _ = await proc.communicate()

                if proc.returncode != 0:
                    error_output = LlamaCppManager._decode_output(stdout)
                    logger.error(f"CMake configure failed:\n{error_output}")
                    raise Exception(f"CMake configure failed. Output:\n{error_output[:2000]}")

                logger.info("CMake configure successful")

            # Step 2: CMake Build
            build_args = ["cmake", "--build", ".", "--config", "Release"]

            # Use multiple cores for faster builds
            import multiprocessing
            cores = multiprocessing.cpu_count()

            if system == "Windows":
                build_args.extend(["--", "/m"])
            else:
                build_args.extend(["-j", str(cores)])

            logger.info(f"Running CMake build: {' '.join(build_args)}")

            proc = await asyncio.create_subprocess_exec(
                *build_args,
                cwd=build_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=build_env
            )
            stdout, _ = await proc.communicate()

            if proc.returncode != 0:
                error_output = LlamaCppManager._decode_output(stdout)
                logger.error(f"CMake build failed:\n{error_output}")
                raise Exception(f"CMake build failed. Output:\n{error_output[:2000]}")

            cuda_status = "with CUDA" if has_cuda else "CPU-only"
            logger.info(f"Build successful ({cuda_status}) on {system}")

        except Exception as e:
            logger.error(f"Build failed: {e}")
            raise Exception(
                f"Failed to build llama.cpp: {str(e)}\n\n"
                "Ensure build tools are installed:\n"
                "  Windows:\n"
                "    - Visual Studio Build Tools 2019 or 2022\n"
                "    - CMake (https://cmake.org/download/)\n"
                "    - Or use: winget install Microsoft.VisualStudio.2022.BuildTools\n"
                "    - Or use: winget install Kitware.CMake\n\n"
                "  Linux (Ubuntu/Debian):\n"
                "    - sudo apt install build-essential cmake\n"
                "    - For CUDA: sudo apt install nvidia-cuda-toolkit\n\n"
                "  macOS:\n"
                "    - xcode-select --install\n"
                "    - brew install cmake"
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

    @staticmethod
    def get_gguf_split_path() -> Path:
        """Find the llama-gguf-split executable."""
        system = platform.system()
        build_dir = LLAMA_CPP_DIR / "build"
        
        if system == "Windows":
            candidates = [
                build_dir / "bin" / "Release" / "llama-gguf-split.exe",
                build_dir / "Release" / "llama-gguf-split.exe",
                build_dir / "bin" / "llama-gguf-split.exe",
                build_dir / "llama-gguf-split.exe",
                build_dir / "bin" / "Release" / "gguf-split.exe",
                build_dir / "Release" / "gguf-split.exe",
                build_dir / "bin" / "gguf-split.exe",
                build_dir / "gguf-split.exe",
            ]
        else:
            candidates = [
                build_dir / "bin" / "llama-gguf-split",
                build_dir / "llama-gguf-split",
                LLAMA_CPP_DIR / "llama-gguf-split",
                build_dir / "bin" / "gguf-split",
                build_dir / "gguf-split",
                LLAMA_CPP_DIR / "gguf-split",
            ]
        
        for path in candidates:
            if path.exists():
                logger.info(f"Found llama-gguf-split at: {path}")
                return path
        
        patterns = ["llama-gguf-split.exe", "gguf-split.exe"] if system == "Windows" else ["llama-gguf-split", "gguf-split"]
        for pattern in patterns:
            found = list(LLAMA_CPP_DIR.rglob(pattern))
            if found:
                for f in found:
                    if "build" in str(f):
                        logger.info(f"Found llama-gguf-split at: {f}")
                        return f
                logger.info(f"Found llama-gguf-split at: {found[0]}")
                return found[0]
        
        raise FileNotFoundError(
            "llama-gguf-split executable not found. Build might have failed.\n"
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

