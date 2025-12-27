
import subprocess
import sys
import shutil
from pathlib import Path

def build():
    print("Building GGUF Forge executable...")
    
    # Check if pyinstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Define build command
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--name", "GGUF-Forge",
        "--add-data", "templates;templates",
        "--hidden-import", "passlib.handlers.argon2",
        "--hidden-import", "argon2-cffi",
        "--hidden-import", "uvicorn", 
        "--hidden-import", "python_multipart",
        "--clean",
        "app_gguf.py"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    print("\nBuild complete!")
    print(f"Executable is located at: {Path('dist/GGUF-Forge.exe').absolute()}")
    print("NOTE: The executable expects to be able to write to its current directory (for DB, Cache, etc).")

if __name__ == "__main__":
    build()
