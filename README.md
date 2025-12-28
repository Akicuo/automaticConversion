# GGUF Forge

A modern, glassmorphic FastAPI application to automate the conversion of HuggingFace models to GGUF format using `llama.cpp`.

> 🚀 **Don't want to self-host?** Use our free hosted service at **[gguforge.com](https://gguforge.com)** — no setup required! Just login with HuggingFace and request your conversions.

## Features
- **HuggingFace OAuth**: Guest login with HuggingFace for requesting conversions
- **Request System**: Users can request model conversions, admins approve/decline with reasons
- **Dashboard**: Real-time status of model conversions with detailed logs
- **Model Search**: Search HuggingFace models with autocomplete
- **Auto-Pipeline**: Download → Convert (FP16) → Quantize (All levels) → Upload
- **Async Processing**: Non-blocking quantization and concurrent uploads
- **Time Tracking**: Detailed timing stats for each conversion step
- **Version Tracking**: Auto-calculated version based on git commits
- **Llama.cpp Management**: Automatically clones and builds `llama.cpp`
- **Validation**: Checks disk space before processing
- **Responsive UI**: Glassmorphism design with real-time updates

## Prerequisites
1.  **Python 3.10+**
2.  **Git**: Required for cloning llama.cpp and version tracking
3.  **Build Tools**:
    - **Windows**: Install [CMake](https://cmake.org/download/) and ensure `cmake` is in your PATH. You might also need Visual Studio Build Tools.
    - **Linux**: `sudo apt install build-essential cmake`

## Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/Akicuo/automaticConversion.git
    cd automaticConversion
    ```

2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Create a `.env` file with the required secrets (see Configuration below)

## Configuration

Create a `.env` file in the project root with the following variables:

### Required Secrets

#### HuggingFace Token
```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
- **Purpose**: Upload converted models to HuggingFace
- **How to get**: 
  1. Go to [HuggingFace Settings → Access Tokens](https://huggingface.co/settings/tokens)
  2. Create a new token with **Write** permissions
  3. Copy the token (starts with `hf_`)
- **Note**: Without this, models will be quantized but not uploaded

#### OAuth Configuration (for Guest Login)
```env
OAUTH_CLIENT_ID=your_oauth_client_id
OAUTH_CLIENT_SECRET=your_oauth_client_secret
OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback
```
- **Purpose**: Enable HuggingFace OAuth login for guests to request conversions
- **How to get**:
  1. Go to [HuggingFace Settings → Connected Applications](https://huggingface.co/settings/connected-applications)
  2. Click "Create a new OAuth app"
  3. Fill in:
     - **Application name**: GGUF Forge (or your choice)
     - **Homepage URL**: `http://localhost:8000` (or your domain)
     - **Redirect URI**: `http://localhost:8000/auth/callback` (or your domain + `/auth/callback`)
     - **Scopes**: Select `openid`, `profile`, `email`
  4. Copy the **Client ID** and **Client Secret**
- **Note**: For production, update `OAUTH_REDIRECT_URI` to your actual domain

### Example `.env` File
```env
# HuggingFace Token (Required for uploads)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OAuth Configuration (Required for guest login)
OAUTH_CLIENT_ID=your_oauth_client_id_here
OAUTH_CLIENT_SECRET=your_oauth_client_secret_here
OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback
```

### Optional Configuration
You can also set these environment variables (they have defaults):
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

## Usage

### Starting the Application
1.  Start the app:
    ```bash
    python app_gguf.py
    ```

2.  **First Run**: Check the console output! It will print your **Admin Username** and **Password**. These credentials are also saved to `creds.txt`.

3.  Open [http://localhost:8000](http://localhost:8000)

### User Roles

#### Guest Users
- Can browse the dashboard and view active conversions
- Can request model conversions (requires HuggingFace OAuth login)
- Can view their request history with status and decline reasons

#### Authenticated Users (via HuggingFace OAuth)
- All guest permissions
- Can submit conversion requests
- View request status (pending/approved/rejected)
- See decline reasons if requests are rejected

#### Admin Users
- All user permissions
- Can directly start model conversions
- View and manage all pending requests
- Approve or decline requests with optional reasons
- Access to full system controls

### Converting a Model

#### As Admin:
1. Login at `/login` with admin credentials
2. Search for a model (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
3. Click **Start** to begin conversion immediately

#### As Guest/User:
1. Click **Login** and choose "Continue with HuggingFace"
2. Authorize the application
3. Search for a model
4. Click **Request** to submit a conversion request
5. Wait for admin approval
6. Check "My Requests" section for status updates

## Conversion Pipeline

Each conversion goes through these steps:

1. **Setup** (10%): Clone and build llama.cpp
2. **Download** (30%): Download model from HuggingFace
3. **Convert** (50%): Convert to FP16 GGUF format
4. **Quantize** (80%): Create all quantization levels (Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0)
5. **Upload** (90%): Concurrent upload of all quants to HuggingFace
6. **README** (100%): Generate and upload README with stats

### Conversion Stats
Each conversion tracks:
- **Job ID**: Unique identifier for the conversion
- **Version**: GGUF Forge version (auto-calculated from git commits)
- **Total Time**: Complete pipeline duration
- **Avg Time per Quant**: Average quantization time
- **Step Breakdown**: Individual timing for each step

## Notes
- The first conversion will take longer as it clones and builds `llama.cpp`
- Ensure you have enough disk space. A 7B model requires ~15GB for download + ~15GB for FP16 + ~5-10GB per quant
- All conversions run asynchronously and won't block the server
- Quantizations are uploaded concurrently (4 at a time) for faster completion
- Files are automatically cleaned up after upload

## Troubleshooting

### OAuth Not Working
- Verify `OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET` are correct
- Check that the redirect URI in HuggingFace matches `OAUTH_REDIRECT_URI` in `.env`
- Ensure the OAuth app has the correct scopes (`openid`, `profile`, `email`)

### Models Not Uploading
- Verify `HF_TOKEN` is set and has **Write** permissions
- Check the token hasn't expired
- Ensure you have permission to create repositories on HuggingFace

### Build Errors
- **Windows**: Install Visual Studio Build Tools and CMake
- **Linux**: Install `build-essential` and `cmake`
- Check that `git` is installed and in PATH

## License
MIT

## Credits
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF quantization
- [HuggingFace](https://huggingface.co) - Model hosting and OAuth
