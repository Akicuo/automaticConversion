# GGUF Forge

A modern, glassmorphic FastAPI application to automate the conversion of HuggingFace models to GGUF format using `llama.cpp`.

## Features
- **Dashboard**: Real-time status of model conversions.
- **Model Search**: Search HuggingFace models with autocomplete.
- **Auto-Pipeline**: Download -> Convert (FP16) -> Quantize (All levels) -> Upload.
- **Llama.cpp Management**: Automatically clones and builds `llama.cpp`.
- **Validation**: Checks disk space before processing.
- **Responsive UI**: Glassmorphism design.

## Prerequisites
1.  **Python 3.10+**
2.  **Build Tools**:
    - **Windows**: Install [CMake](https://cmake.org/download/) and ensure `cmake` is in your PATH. You might also need Visual Studio Build Tools.
    - **Linux**: `sudo apt install build-essential cmake`

## Installation
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  (Optional) Create a `.env` file to enable HuggingFace uploads:
    ```
    HF_TOKEN=your_write_token_here
    ```

## Usage
1.  Start the app:
    ```bash
    python app_gguf.py
    ```
2.  **First Run**: Check the console output! It will print your **Admin Username** and **Password**.
3.  Open [http://localhost:8000](http://localhost:8000).
4.  Login at `/login` with the credentials.
5.  Search for a model (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) and click **Process Model**.

## Notes
- The first conversion will take longer as it clones and builds `llama.cpp`.
- Ensure you have enough disk space. A 7B model requires ~15GB for download + ~15GB for FP16 + ~5-10GB per quant.
