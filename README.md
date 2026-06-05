# GGUF Forge

 - This Project is soon deprecated and will be replaced by much better architecture and code


A modern, glassmorphic FastAPI application to automate the conversion of HuggingFace models to GGUF format using `llama.cpp`.

> 🚀 **Don't want to self-host?** Use our free hosted service at **[gguforge.com](https://gguforge.com)** — no setup required! Just login with HuggingFace and request your conversions.

## Features

### Core Features
- **HuggingFace OAuth**: Guest login with HuggingFace for requesting conversions
- **Request System**: Users can request model conversions, admins approve/decline with reasons
- **Quant Selection**: Users can request specific quantization types; admins can modify the selection
- **Dashboard**: Real-time status of model conversions with detailed logs
- **Model Search**: Search HuggingFace models with autocomplete
- **Auto-Pipeline**: Download → Convert (FP16) → Quantize (Selected levels) → Upload
- **Configurable Storage**: Set custom download path for models (local, external drive, or HF_HOME)
- **Async Processing**: Non-blocking quantization and concurrent uploads
- **Time Tracking**: Detailed timing stats for each conversion step
- **Version Tracking**: Auto-calculated version based on git commits
- **Llama.cpp Management**: Automatically clones and builds `llama.cpp`
- **Validation**: Checks disk space before processing
- **Responsive UI**: Glassmorphism design with real-time updates

### Task Management
- **Job Termination**: Admins can terminate any running conversion job mid-process
- **Startup Recovery**: Automatically detects and recovers stuck jobs after server crashes
- **Ticket System**: Conversational threads between users and admins for discussing requests
- **Discussion Workflow**: Instead of instant rejection, admins can start discussions with users

### Database Support
- **SQLite**: Default, zero-config local database
- **MSSQL**: Enterprise-grade Microsoft SQL Server support for production deployments

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

## Docker Installation

For containerized deployment, use the included Docker configuration.

### Quick Start

```bash
# 1. Create .env file with your tokens
cat > .env << EOF
HF_TOKEN=hf_your_token_here
OAUTH_CLIENT_ID=your_client_id
OAUTH_CLIENT_SECRET=your_client_secret
OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback
ADMIN_USERS=your_hf_username
EOF

# 2. Build and run
docker-compose up -d

# 3. View logs
docker-compose logs -f

# 4. Check admin credentials (first run)
cat data/creds.txt
```

### Docker Image Contents

The Dockerfile includes:
- **Python 3.12** (slim-bookworm base)
- **Git** - for llama.cpp cloning and app updates
- **Vim** - for editing config files in container
- **Build tools** - cmake, g++, make (for compiling llama.cpp)
- **Microsoft ODBC Driver 18** - for SQL Server support

### Volume Mounts

| Mount | Purpose |
|-------|---------|
| `./data` | Database + credentials (persistent) |
| `./cache` | Downloaded models (can be very large!) |
| `./llama.cpp` | Compiled llama.cpp (avoids rebuilding) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for uploads (required) |
| `OAUTH_CLIENT_ID` | HF OAuth app ID (optional) |
| `OAUTH_CLIENT_SECRET` | HF OAuth secret (optional) |
| `OAUTH_REDIRECT_URI` | OAuth callback URL |
| `ADMIN_USERS` | Comma-separated HF usernames for admin access |
| `PARALLEL_QUANT_JOBS` | Number of simultaneous quantization jobs (default: 2) |
| `HOST` | Server bind address (default: 0.0.0.0) |
| `PORT` | Server port (default: 8000) |
| `MSSQL_*` | SQL Server connection settings (optional) |

### GPU Support (NVIDIA)

For GPU-accelerated quantization, uncomment the GPU profile in `docker-compose.yml`:

```bash
# Run with GPU support
docker-compose --profile gpu up -d
```

Requires NVIDIA Container Toolkit installed on the host.

### Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 32 GB |
| Storage | 50 GB | 200+ GB |
| CPU Cores | 4 | 16+ |

> **Note**: Storage requirements depend on model size. A 7B model needs ~30GB during conversion.

### Manual Docker Build

```bash
# Build the image
docker build -t gguf-forge .

# Run with environment file
docker run -d \
  --name gguf-forge \
  -p 8000:8000 \
  -v ./data:/data \
  -v ./cache:/app/.cache \
  -v ./llama.cpp:/app/llama.cpp \
  --env-file .env \
  gguf-forge
```

## Configuration

Create a `.env` file in the project root with the following variables:

### Required Secrets

#### HuggingFace Token
```env
# Parallel Quantization Jobs (default: 2)
# How many quantizations to run simultaneously. Each job gets total_cores / jobs threads.
PARALLEL_QUANT_JOBS=2

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

  ![HuggingFace OAuth Scopes](static/hf-scopes.png)
  *Ensure 'openid', 'profile', and 'email' are selected*

- **Note**: For production, update `OAUTH_REDIRECT_URI` to your actual domain

### Example `.env` File
```env
# HuggingFace Token (Required for uploads)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OAuth Configuration (Required for guest login)
OAUTH_CLIENT_ID=your_oauth_client_id_here
OAUTH_CLIENT_SECRET=your_oauth_client_secret_here
OAUTH_REDIRECT_URI=http://localhost:8000/auth/callback

# Admin Users (HuggingFace usernames, comma-separated)
ADMIN_USERS=Akicuo,ungsung
```

### Admin Configuration
Instead of using a password-based admin account, admins are configured via the `ADMIN_USERS` environment variable:

```env
# Comma-separated list of HuggingFace usernames
ADMIN_USERS=Akicuo,ungsung,anotheruser
```

When these users login via HuggingFace OAuth, they automatically get admin privileges. This means:
- No need to remember separate admin credentials
- Easy to add/remove admins by updating `.env` and restarting
- Role is updated on each login (so changes take effect immediately)

### Optional Configuration
You can also set these environment variables (they have defaults):
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `LLAMA_CPP_REPO`: Git repository used for llama.cpp (default: `https://github.com/ggml-org/llama.cpp`)
- `LLAMA_CPP_DIR`: Path to llama.cpp directory (default: `./llama.cpp`)
  - Can be **relative** (e.g., `../llama.cpp`, `custom/llama.cpp`) or **absolute** (e.g., `/opt/llama.cpp`, `C:\tools\llama.cpp`)
  - Relative paths are resolved from the project root directory
  - Useful if you want to share a single llama.cpp installation across multiple projects
  - Example: `LLAMA_CPP_DIR=/opt/llama.cpp` or `LLAMA_CPP_DIR=../shared/llama.cpp`
- HuggingFace upload acceleration is enabled by default with `HF_HUB_ENABLE_HF_TRANSFER=1` and `HF_XET_HIGH_PERFORMANCE=1`
- `PARALLEL_QUANT_JOBS`: Number of simultaneous quantization jobs (default: 2)
  - `1` = Sequential (slowest, safest for RAM)
  - `2` = Default (balanced)
  - `4` = Aggressive (requires more RAM, faster on high-core systems)
  - **Note**: CPU cores are divided between parallel jobs. A 16-core system with 2 parallel jobs will give 8 cores to each.
- `DB_POOL_MIN`: Minimum MSSQL connection pool size (default: 2)
- `DB_POOL_MAX`: Maximum MSSQL connection pool size (default: 10)
- `MSSQL_CONN_TIMEOUT`: MSSQL connection timeout in seconds (default: 60)
- `MSSQL_LOGIN_TIMEOUT`: MSSQL login timeout in seconds (default: 60)

### Model Storage Configuration

Configure where models are downloaded and processed:

```env
# Model Download Path (optional)
# Supports absolute paths (D:\Models, /mnt/external) or relative paths (./models)
MODEL_DOWNLOAD_PATH=/path/to/large/storage

# Alternatively, HuggingFace's HF_HOME is used as fallback
HF_HOME=/path/to/hf/cache
```

**Priority Order**: `MODEL_DOWNLOAD_PATH` > `HF_HOME` > default `.cache` folder

This is useful when:
- You have limited space on your system drive
- You want to store models on an external drive or network storage
- You want to share a cache with other HuggingFace applications

### Database Configuration

GGUF Forge supports two database backends: **SQLite** (default) and **MSSQL**.

#### SQLite (Default)
No configuration needed. A local `gguf_app.db` file is created automatically.

#### MSSQL Server
To use MSSQL instead of SQLite, add these variables to your `.env`:

```env
# Database type: "sqlite" (default) or "mssql"
DB_TYPE=mssql

# MSSQL Configuration
MSSQL_HOST=your-server.database.net
MSSQL_PORT=1433
MSSQL_DATABASE=gguforge
MSSQL_USER=your_username
MSSQL_PASSWORD=your_password
MSSQL_ENCRYPT=yes
MSSQL_TRUST_CERT=yes
```

**Prerequisites for MSSQL:**
1. Install the ODBC Driver for SQL Server:
   - **Windows**: Download from [Microsoft](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
   - **Linux (Ubuntu/Debian)**:
     ```bash
     curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
     curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list
     sudo apt-get update
     sudo ACCEPT_EULA=Y apt-get install -y msodbcsql17
     ```
   - **WSL (Windows Subsystem for Linux)**:
     ```bash
     # First install unixODBC (required for pyodbc)
     sudo apt-get update
     sudo apt-get install -y unixodbc unixodbc-dev
     
     # Add Microsoft repository
     curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
     curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list
     
     # Install ODBC Driver 17
     sudo apt-get update
     sudo ACCEPT_EULA=Y apt-get install -y msodbcsql17
     
     # Verify installation
     odbcinst -q -d -n "ODBC Driver 17 for SQL Server"
     
     # Reinstall pyodbc (may be needed after installing ODBC)
     pip install --force-reinstall pyodbc
     ```
   - **macOS**:
     ```bash
     # Install Homebrew if not installed
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     
     # Install unixODBC and ODBC Driver
     brew install unixodbc
     brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
     brew update
     HOMEBREW_ACCEPT_EULA=Y brew install msodbcsql17
     ```

2. Install pyodbc (already in requirements.txt):
   ```bash
   pip install pyodbc
   ```

> **💡 Tip:** If you don't need MSSQL, just use SQLite! Set `DB_TYPE=sqlite` in your `.env` file (or remove the line entirely) - no additional dependencies required.

**Testing Connection:**
After configuration, verify with:
```bash
curl http://localhost:8000/api/health
```

Response shows database status:
```json
{
  "status": "healthy",
  "database": {
    "type": "mssql",
    "connected": true,
    "message": "Successfully connected to mssql database"
  }
}
```

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
5. **Optional**: Select specific quantization types you want (or leave empty for all)
6. Wait for admin approval
7. Check "My Requests" section for status updates

### Quant Selection

Users can request specific quantization types when submitting a conversion request:

**Available Quant Types:**
| Type | Description |
|------|-------------|
| Q2_K | 2-bit (smallest, lowest quality) |
| Q3_K_S/M/L | 3-bit small/medium/large |
| Q4_0 | 4-bit legacy |
| Q4_K_S | 4-bit small (recommended for low memory) |
| Q4_K_M | 4-bit medium (good balance) |
| Q5_0 | 5-bit legacy |
| Q5_K_S/M | 5-bit small/medium (good quality) |
| Q6_K | 6-bit (high quality) |
| Q8_0 | 8-bit (highest quality, largest size) |

**Admin Quant Modification:**
When approving a request, admins can:
- Use the user's requested quants as-is
- Add additional quant types
- Remove unnecessary quant types
- Override with a completely different selection

If no quants are specified (by user or admin), all available quants are generated.

## Conversion Pipeline

Each conversion goes through these steps:

1. **Setup** (10%): Clone and build llama.cpp
2. **Download** (30%): Download model from HuggingFace
3. **Convert** (50%): Convert to FP16 GGUF format
4. **Quantize** (80%): Create all quantization levels in parallel (Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0)
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
- Quantizations run in parallel (configurable via `PARALLEL_QUANT_JOBS`) with CPU cores distributed between jobs
- Each job performs an immediate upload-and-delete after creation to minimize disk space usage
- Files are automatically cleaned up after successful upload

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

### MSSQL Connection Issues
- Ensure ODBC Driver 17 for SQL Server is installed
- Verify server allows remote connections
- Check firewall rules for the MSSQL port
- Test connection with: `curl http://localhost:8000/api/health`

### App Crashes or Conversion Failures

**High Parallel Quant Jobs**
If you set `PARALLEL_QUANT_JOBS` too high, the application may crash or conversions may fail due to resource exhaustion.

**Symptoms:**
- Application shuts down unexpectedly during quantization
- Conversions fail with "out of memory" or "killed" errors
- System becomes unresponsive during conversion

**Solution:**
Set `PARALLEL_QUANT_JOBS=1` or `PARALLEL_QUANT_JOBS=2` in your `.env` file:
```env
# Recommended: 1-2 parallel jobs for stable operation
PARALLEL_QUANT_JOBS=2
```

**Guidelines:**
- **1 job**: Safest option, uses all CPU cores for a single quantization
- **2 jobs**: Balanced option, splits CPU cores between 2 parallel quantizations
- **4+ jobs**: Only use on high-memory systems (32GB+ RAM) with many CPU cores (16+)

Each parallel job consumes significant memory for the model and quantization process. Too many parallel jobs can exceed available RAM and cause the application to be terminated by the operating system.

**Debugging Tips:**
- Check browser console (F12) for JavaScript errors - logs now show detailed conversion progress
- Check server logs for Python errors during conversion
- Monitor system resources: RAM usage, CPU load, disk space during conversion
- Try with `PARALLEL_QUANT_JOBS=1` to isolate issues

## Admin Features

### App Updates
Admins can update the application directly from the dashboard:
1. An **Update App** button appears in the top-right corner
2. The button automatically checks for available updates
3. When an update is available, the button shows the commit count
4. Click to pull the latest changes and restart the server

**Manual Update Scripts:**
```bash
# Windows PowerShell
.\update.ps1

# Linux/macOS
chmod +x update.sh  # First time only
./update.sh
```

> **Note**: Updates will interrupt any running conversion jobs.

### Job Termination
Admins can terminate any running conversion job:
1. Find the job in the "Conversion Queue" section
2. Click the orange **Terminate** button
3. The job will stop and be marked as "TERMINATED"

Jobs terminated mid-process can be restarted by re-approving the original request.

### Ticket/Discussion System
Instead of instantly rejecting requests, admins can start discussions:

1. **Starting a Discussion**: Click "Discuss" on a pending request
2. **Admin Dashboard**: View all active discussions in the "Active Discussions" section
3. **Conversation**: Both admin and user can exchange messages
4. **Resolution**: Admin can either:
   - **Approve**: Start the conversion
   - **Close**: Reject the request

Users see their discussions in "My Requests" with a "View Thread" button.

### Startup Recovery
If the server crashes during a conversion:
- On restart, stuck jobs are automatically marked as "INTERRUPTED"
- Associated requests are reset to "PENDING" for re-approval
- No manual intervention required

## API Reference

### Public Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with database status |
| `/api/hf/search?q=query` | GET | Search HuggingFace models |
| `/api/status/all` | GET | Get all conversion jobs |

### Authenticated Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/requests/submit` | POST | Submit a conversion request (with optional quants) |
| `/api/requests/my` | GET | Get user's requests |
| `/api/quants` | GET | Get available quantization types |
| `/api/tickets/my` | GET | Get user's tickets |
| `/api/tickets/{id}/messages` | GET | Get ticket messages |
| `/api/tickets/{id}/reply` | POST | Reply to a ticket |
| `/api/tickets/{id}/reopen` | POST | Reopen a closed ticket |

### Admin Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models/process` | POST | Start a conversion directly (with optional quants) |
| `/api/models/{id}/terminate` | POST | Terminate a running job |
| `/api/models/{id}/continue` | POST | Resume an interrupted job |
| `/api/requests/all` | GET | View all requests |
| `/api/requests/{id}/approve` | POST | Approve a request (with optional quant override) |
| `/api/requests/{id}/reject` | POST | Reject a request |
| `/api/tickets/create` | POST | Create a discussion ticket |
| `/api/tickets/all` | GET | View all tickets |
| `/api/tickets/{id}/approve` | POST | Approve from ticket (with optional quants) |
| `/api/tickets/{id}/close` | POST | Close/reject a ticket |
| `/api/admin/db-info` | GET | Database information |
| `/api/admin/check-update` | GET | Check for available app updates |
| `/api/admin/update-app` | POST | Pull updates and restart server |

## Project Structure

```
automaticConversion/
├── app_gguf.py          # Main FastAPI application
├── database.py          # Database abstraction (SQLite + MSSQL)
├── security.py          # Rate limiting, bot detection, spam protection
├── models.py            # Pydantic request/response models
├── managers.py          # LlamaCpp and HuggingFace managers
├── workflow.py          # Model conversion pipeline
├── websocket_manager.py # Real-time WebSocket updates
├── routes/
│   ├── auth.py          # Authentication (admin login + OAuth)
│   ├── models.py        # Model processing endpoints
│   ├── requests.py      # Request system endpoints
│   └── tickets.py       # Ticket/discussion endpoints
├── static/
│   ├── js/
│   │   └── tickets.js   # Ticket modal JavaScript
│   └── favicon.png
├── templates/
│   ├── base.html        # Base template
│   ├── index.html       # Main dashboard
│   └── login.html       # Admin login page
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose configuration
├── update.ps1           # Update script (Windows PowerShell)
├── update.sh            # Update script (Linux/macOS)
├── .env.example         # Environment template
└── requirements.txt     # Python dependencies
```

## License
MIT

## Credits
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - GGUF quantization
- [HuggingFace](https://huggingface.co) - Model hosting and OAuth
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [pyodbc](https://github.com/mkleehammer/pyodbc) - MSSQL connectivity
