# MSSQL Database Setup Guide

This guide will help you install the required ODBC libraries to connect to Microsoft SQL Server from your Linux system.

## Prerequisites

Your application is configured to use MSSQL as the database backend. This requires:

1. **unixODBC** - The ODBC driver manager for Unix/Linux systems
2. **Microsoft ODBC Driver for SQL Server** - The actual driver that connects to SQL Server

## Installation Instructions

### Ubuntu/Debian Linux

Run the following commands to install the required libraries:

```bash
# Update package lists
sudo apt-get update

# Install unixODBC
sudo apt-get install -y unixodbc-dev unixodbc

# Download and install the Microsoft GPG key (correct method for Ubuntu 20.04+)
curl https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg

# Add the Microsoft repository for your Ubuntu version
curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list

# Update package lists again
sudo apt-get update

# Install the Microsoft ODBC Driver 18 for SQL Server
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Optional: Install command-line tools (sqlcmd)
sudo ACCEPT_EULA=Y apt-get install -y mssql-tools18
echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc
source ~/.bashrc
```

### Red Hat/CentOS/Fedora

```bash
# Add Microsoft repository
sudo curl https://packages.microsoft.com/config/rhel/$(rpm -E %{rhel})/prod.repo -o /etc/yum.repos.d/mssql-release.repo

# Install unixODBC
sudo yum install -y unixODBC-devel

# Install the Microsoft ODBC Driver
sudo ACCEPT_EULA=Y yum install -y msodbcsql18
```

### macOS

```bash
# Install Homebrew if not already installed
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install unixODBC
brew install unixodbc

# Add Microsoft tap
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release

# Install the ODBC driver
brew install msodbcsql18 mssql-tools18
```

## Verify Installation

After installation, verify that the ODBC driver is properly installed:

```bash
# List available ODBC drivers
odbcinst -q -d
```

You should see output similar to:

```
[ODBC Driver 18 for SQL Server]
[ODBC Driver 17 for SQL Server]
```

## Test Connection

You can test your database connection using Python:

```python
import pyodbc

# List available drivers
print("Available ODBC drivers:")
print(pyodbc.drivers())

# Test connection (replace with your credentials)
conn_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=your-server.database.net,1433;"
    "DATABASE=your_database;"
    "UID=your_username;"
    "PWD=your_password;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

try:
    conn = pyodbc.connect(conn_str)
    print("✓ Connection successful!")
    conn.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

## Environment Configuration

Make sure your `.env` file is configured correctly:

```env
# Database type: "sqlite" (default) or "mssql"
DB_TYPE=mssql

# MSSQL Configuration
MSSQL_HOST=your-server.database.net
MSSQL_PORT=1433
MSSQL_DATABASE=your_database
MSSQL_USER=your_username
MSSQL_PASSWORD=your_password
MSSQL_ENCRYPT=yes
MSSQL_TRUST_CERT=yes

# Optional: Connection timeout settings (in seconds)
MSSQL_CONN_TIMEOUT=60
MSSQL_LOGIN_TIMEOUT=60

# Optional: Manually specify ODBC driver
# MSSQL_DRIVER=ODBC Driver 18 for SQL Server
```

## Troubleshooting

### Error: "NO_PUBKEY" or GPG signature verification failed

If you see errors like:
```
The following signatures couldn't be verified because the public key is not available: NO_PUBKEY EB3E94ADBE1229CF
```

This means the GPG key wasn't installed correctly. Fix it with:

```bash
# Remove any incorrect attempts
sudo rm -f /etc/apt/sources.list.d/mssql-release.list
sudo rm -f /etc/apt/trusted.gpg.d/microsoft.asc

# Install the key correctly (binary format, correct location)
curl https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg

# Re-add the repository
curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list

# Update and install
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18
```

**Note:** The `apt-key add` command is deprecated in Ubuntu 20.04+ and will cause issues. Always use `gpg --dearmor` to convert the key to binary format.

### Error: "libodbc.so.2: cannot open shared object file"

This means unixODBC is not installed. Install it with:

```bash
sudo apt-get install -y unixodbc unixodbc-dev
```

### Error: "Can't open lib 'ODBC Driver 18 for SQL Server'"

The Microsoft ODBC driver is not installed. Follow the installation steps above for your operating system.

### Connection Timeout or Network Errors

1. **Check firewall**: Make sure port 1433 (or your custom port) is open
2. **Check server address**: Verify the MSSQL_HOST is correct
3. **Check credentials**: Verify username and password
4. **SSL/TLS issues**: Try setting `MSSQL_TRUST_CERT=yes` if using self-signed certificates
5. **Timeout issues**: If you're experiencing disconnections during long-running jobs, increase the timeout values:
   ```env
   MSSQL_CONN_TIMEOUT=120
   MSSQL_LOGIN_TIMEOUT=120
   ```

### Session Timeout / Getting Logged Out

If you're experiencing automatic logouts during long-running conversion jobs:

1. **This is typically a MSSQL server-side setting**, not an application issue
2. **Server-side connection pooling** or firewall may be closing idle connections
3. **Solutions**:
   - Increase `MSSQL_CONN_TIMEOUT` and `MSSQL_LOGIN_TIMEOUT` in your `.env` file (default: 60 seconds)
   - Check your MSSQL server's "Remote Query Timeout" setting (default: 600 seconds)
   - Check your Azure SQL firewall rules if using Azure
   - Verify no network devices (routers, load balancers) are timing out idle connections

The application already has automatic retry logic for connection timeouts, but persistent session issues are usually server-side configuration.

### TypeError: Object of type datetime is not JSON serializable

This error occurs when datetime fields from MSSQL are not properly converted to strings for JSON responses. This has been fixed in the codebase by:

1. Adding a `to_dict()` method to `DatabaseRow` class that converts datetime objects to ISO format strings
2. Using a custom JSON serializer in WebSocket broadcasts

If you encounter this error after updating the code, make sure to restart your application to load the fixes.

### Driver Version Issues

If you have multiple ODBC driver versions installed, you can explicitly specify which one to use:

```env
MSSQL_DRIVER=ODBC Driver 17 for SQL Server
```

## Alternative: Use SQLite Instead

If you don't need the enterprise features of MSSQL, you can easily switch to SQLite:

1. Edit your `.env` file (or create one if it doesn't exist):
   ```env
   DB_TYPE=sqlite
   ```

2. Remove the MSSQL-related environment variables

3. Restart your application

SQLite requires no additional system libraries and stores everything in a local file (`gguf_app.db`).

## Additional Resources

- [Microsoft ODBC Driver Documentation](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
- [unixODBC Documentation](http://www.unixodbc.org/)
- [PyODBC Wiki](https://github.com/mkleehammer/pyodbc/wiki)

