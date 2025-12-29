"""
Database abstraction layer for GGUF Forge.
Supports both SQLite and MSSQL backends with async operations.
Uses aioodbc for async ODBC connections.
"""
import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, Any, List, Dict
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod

logger = logging.getLogger("GGUF_Forge")

# Database configuration
DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()  # sqlite or mssql
DB_PATH = None  # For SQLite

# MSSQL Configuration
MSSQL_HOST = os.getenv("MSSQL_HOST", "")
MSSQL_PORT = os.getenv("MSSQL_PORT", "1433")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "")
MSSQL_USER = os.getenv("MSSQL_USER", "")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "")
MSSQL_ENCRYPT = os.getenv("MSSQL_ENCRYPT", "yes")
MSSQL_TRUST_CERT = os.getenv("MSSQL_TRUST_CERT", "yes")

# Connection retry settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Admin Users - comma-separated list of HuggingFace usernames who should be admins
ADMIN_USERS = [u.strip().lower() for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip()]


def is_admin_user(username: str) -> bool:
    """Check if a username is in the admin list."""
    return username.lower() in ADMIN_USERS


def set_db_path(path: Path):
    """Set the SQLite database path."""
    global DB_PATH
    DB_PATH = path


class DatabaseRow:
    """A dict-like row that supports both dict access and attribute access."""
    def __init__(self, data: dict):
        self._data = data
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __contains__(self, key):
        return key in self._data
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()


class AsyncDatabaseConnection(ABC):
    """Abstract base class for async database connections."""
    
    @abstractmethod
    async def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a query and return cursor/result."""
        pass
    
    @abstractmethod
    async def commit(self):
        """Commit the transaction."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the connection."""
        pass
    
    @abstractmethod
    async def fetchone(self) -> Optional[DatabaseRow]:
        """Fetch one row from last query."""
        pass
    
    @abstractmethod
    async def fetchall(self) -> List[DatabaseRow]:
        """Fetch all rows from last query."""
        pass
    
    @property
    @abstractmethod
    def lastrowid(self) -> int:
        """Get the last inserted row ID."""
        pass


class AsyncSQLiteConnection(AsyncDatabaseConnection):
    """Async SQLite database connection wrapper using aiosqlite."""
    
    def __init__(self, conn, cursor=None):
        self.conn = conn
        self.cursor = cursor
        self._last_id = 0
    
    async def execute(self, query: str, params: tuple = ()) -> 'AsyncSQLiteConnection':
        self.cursor = await self.conn.execute(query, params)
        self._last_id = self.cursor.lastrowid
        return self
    
    async def commit(self):
        await self.conn.commit()
    
    async def close(self):
        await self.conn.close()
    
    async def fetchone(self) -> Optional[DatabaseRow]:
        if self.cursor:
            row = await self.cursor.fetchone()
            if row:
                # Get column names from cursor description
                columns = [desc[0] for desc in self.cursor.description]
                return DatabaseRow(dict(zip(columns, row)))
        return None
    
    async def fetchall(self) -> List[DatabaseRow]:
        if self.cursor:
            rows = await self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return [DatabaseRow(dict(zip(columns, row))) for row in rows]
        return []
    
    @property
    def lastrowid(self) -> int:
        return self._last_id


class AsyncMSSQLConnection(AsyncDatabaseConnection):
    """Async MSSQL database connection wrapper using aioodbc."""
    
    # Cache the detected driver to avoid repeated lookups and logging
    _detected_driver = None
    
    # Connection timeout errors that indicate session timeout
    TIMEOUT_ERROR_CODES = [
        '08S01',  # Communication link failure
        '08001',  # Unable to connect
        'HYT00',  # Timeout expired
        '40001',  # Deadlock
    ]
    
    @classmethod
    def _get_driver(cls):
        """Auto-detect available ODBC driver or use environment override."""
        # Return cached driver if already detected
        if cls._detected_driver:
            return cls._detected_driver
        
        # Allow override via environment variable
        driver_override = os.getenv("MSSQL_DRIVER", "")
        if driver_override:
            cls._detected_driver = driver_override
            logger.info(f"Using ODBC driver (override): {driver_override}")
            return driver_override
        
        # Try to auto-detect available driver (prefer newer versions)
        import pyodbc
        drivers = pyodbc.drivers()
        
        # Check for drivers in order of preference
        preferred_drivers = [
            "ODBC Driver 18 for SQL Server",
            "ODBC Driver 17 for SQL Server",
            "ODBC Driver 13 for SQL Server",
            "SQL Server",
        ]
        
        for driver in preferred_drivers:
            if driver in drivers:
                cls._detected_driver = driver
                logger.info(f"Using ODBC driver: {driver}")
                return driver
        
        # Fallback to 18 if nothing found (will error with helpful message)
        cls._detected_driver = "ODBC Driver 18 for SQL Server"
        return cls._detected_driver
    
    @classmethod
    def _get_connection_string(cls):
        """Build the ODBC connection string."""
        driver = cls._get_driver()
        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={MSSQL_HOST},{MSSQL_PORT};"
            f"DATABASE={MSSQL_DATABASE};"
            f"UID={MSSQL_USER};"
            f"PWD={MSSQL_PASSWORD};"
            f"Encrypt={MSSQL_ENCRYPT};"
            f"TrustServerCertificate={MSSQL_TRUST_CERT};"
        )
    
    def __init__(self, conn):
        self.conn = conn
        self.cursor = None
        self._columns = []
        self._last_id = 0
        self._retry_count = 0
    
    def _adapt_params(self, params: tuple) -> tuple:
        """Adapt parameter values for MSSQL compatibility."""
        if not params:
            return params
        
        adapted = []
        for p in params:
            if isinstance(p, str):
                # Convert ISO datetime format (2025-12-29T07:02:59) to MSSQL format (2025-12-29 07:02:59)
                # Check if it looks like an ISO datetime
                if len(p) >= 19 and p[4] == '-' and p[7] == '-' and p[10] == 'T':
                    p = p.replace('T', ' ')
            adapted.append(p)
        return tuple(adapted)
    
    @classmethod
    def _is_connection_timeout(cls, error) -> bool:
        """Check if the error is a connection/session timeout."""
        error_str = str(error).upper()
        
        # Check for known error codes
        for code in cls.TIMEOUT_ERROR_CODES:
            if code in error_str:
                return True
        
        # Check for timeout-related messages
        timeout_keywords = [
            'TIMEOUT', 'TIMED OUT', 'CONNECTION LOST',
            'CONNECTION RESET', 'COMMUNICATION LINK FAILURE',
            'LOGIN FAILED', 'CONNECTION FAILURE', 'NETWORK ERROR',
            'BROKEN PIPE', 'CONNECTION CLOSED'
        ]
        for keyword in timeout_keywords:
            if keyword in error_str:
                return True
        
        return False
    
    async def execute(self, query: str, params: tuple = ()) -> 'AsyncMSSQLConnection':
        """Execute a query with automatic retry on connection timeout."""
        # Convert SQLite-style placeholders and syntax
        query = self._adapt_query(query)
        
        # Adapt parameters for MSSQL compatibility
        params = self._adapt_params(params)
        
        retries = 0
        last_error = None
        
        while retries <= MAX_RETRIES:
            try:
                self.cursor = await self.conn.cursor()
                
                if params:
                    await self.cursor.execute(query, params)
                else:
                    await self.cursor.execute(query)
                
                # Get column names if available
                if self.cursor.description:
                    self._columns = [column[0] for column in self.cursor.description]
                
                # Handle INSERT to get last row ID
                if query.strip().upper().startswith('INSERT'):
                    try:
                        await self.cursor.execute("SELECT SCOPE_IDENTITY()")
                        result = await self.cursor.fetchone()
                        if result and result[0]:
                            self._last_id = int(result[0])
                    except:
                        pass
                
                # Reset retry count on success
                self._retry_count = 0
                return self
                
            except Exception as e:
                last_error = e
                
                if self._is_connection_timeout(e) and retries < MAX_RETRIES:
                    retries += 1
                    self._retry_count = retries
                    logger.warning(f"Connection timeout detected, attempting reconnect (attempt {retries}/{MAX_RETRIES}): {e}")
                    
                    # Wait before retrying
                    await asyncio.sleep(RETRY_DELAY * retries)
                    
                    # Try to reconnect
                    try:
                        await self._reconnect()
                    except Exception as reconnect_error:
                        logger.error(f"Reconnection failed: {reconnect_error}")
                        # Continue to next retry attempt
                else:
                    # Not a timeout error or max retries exceeded
                    raise
        
        # Max retries exceeded
        raise last_error
    
    async def _reconnect(self):
        """Attempt to reconnect to the database."""
        import aioodbc
        
        # Close existing connection if possible
        try:
            await self.conn.close()
        except:
            pass
        
        # Create new connection
        conn_str = self._get_connection_string()
        self.conn = await aioodbc.connect(dsn=conn_str)
        logger.info("Successfully reconnected to MSSQL database")
    
    def _adapt_query(self, query: str) -> str:
        """Adapt SQLite query syntax to MSSQL."""
        import re
        
        # Replace AUTOINCREMENT with IDENTITY
        query = query.replace("AUTOINCREMENT", "IDENTITY(1,1)")
        
        # Replace INTEGER PRIMARY KEY AUTOINCREMENT with INT PRIMARY KEY IDENTITY
        query = query.replace("INTEGER PRIMARY KEY IDENTITY(1,1)", "INT PRIMARY KEY IDENTITY(1,1)")
        
        # Replace DATETIME DEFAULT CURRENT_TIMESTAMP
        query = query.replace("DATETIME DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT GETDATE()")
        
        # Replace TEXT with NVARCHAR(MAX) for better Unicode support
        query = query.replace(" TEXT ", " NVARCHAR(MAX) ")
        query = query.replace(" TEXT,", " NVARCHAR(MAX),")
        query = query.replace(" TEXT)", " NVARCHAR(MAX))")
        
        # Handle INSERT OR REPLACE -> MERGE (simplified: just use INSERT for now)
        if "INSERT OR REPLACE" in query.upper():
            # For simplicity, we'll handle this with a DELETE + INSERT pattern
            # In production, you'd want proper MERGE statements
            query = query.replace("INSERT OR REPLACE", "INSERT")
        
        # Handle CREATE TABLE IF NOT EXISTS
        if "CREATE TABLE IF NOT EXISTS" in query:
            table_name = query.split("CREATE TABLE IF NOT EXISTS")[1].split("(")[0].strip()
            query = f"""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
            {query.replace('CREATE TABLE IF NOT EXISTS', 'CREATE TABLE')}
            """
        
        # Handle LIMIT -> TOP (MSSQL uses TOP instead of LIMIT)
        # Pattern: SELECT ... FROM ... LIMIT N  ->  SELECT TOP N ... FROM ...
        limit_match = re.search(r'\bLIMIT\s+(\d+)\s*$', query, re.IGNORECASE)
        if limit_match:
            limit_num = limit_match.group(1)
            # Remove the LIMIT clause
            query = re.sub(r'\bLIMIT\s+\d+\s*$', '', query, flags=re.IGNORECASE)
            # Add TOP after SELECT
            query = re.sub(r'^(\s*SELECT\s+)', rf'\1TOP {limit_num} ', query, flags=re.IGNORECASE)
        
        return query
    
    async def commit(self):
        await self.conn.commit()
    
    async def close(self):
        await self.conn.close()
    
    async def fetchone(self) -> Optional[DatabaseRow]:
        if self.cursor:
            row = await self.cursor.fetchone()
            if row:
                return DatabaseRow(dict(zip(self._columns, row)))
        return None
    
    async def fetchall(self) -> List[DatabaseRow]:
        if self.cursor:
            rows = await self.cursor.fetchall()
            return [DatabaseRow(dict(zip(self._columns, row))) for row in rows]
        return []
    
    @property
    def lastrowid(self) -> int:
        return self._last_id


async def get_db_connection() -> AsyncDatabaseConnection:
    """Get an async database connection based on configuration."""
    if DB_TYPE == "mssql":
        import aioodbc
        conn_str = AsyncMSSQLConnection._get_connection_string()
        conn = await aioodbc.connect(dsn=conn_str)
        return AsyncMSSQLConnection(conn)
    else:
        import aiosqlite
        conn = await aiosqlite.connect(DB_PATH)
        return AsyncSQLiteConnection(conn)


async def init_db():
    """Initialize the database with all required tables."""
    conn = await get_db_connection()
    
    try:
        if DB_TYPE == "mssql":
            await _init_mssql_tables(conn)
        else:
            await _init_sqlite_tables(conn)
        
        await conn.commit()
        logger.info(f"Database initialized successfully ({DB_TYPE})")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        await conn.close()


async def _init_sqlite_tables(conn: AsyncDatabaseConnection):
    """Initialize SQLite tables."""
    
    # Users table
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL,
            api_key TEXT
        )
    ''')
    
    # Models table
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            hf_repo_id TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER DEFAULT 0,
            log TEXT DEFAULT '',
            error_details TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME
        )
    ''')
    
    # Requests table
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hf_repo_id TEXT NOT NULL,
            requested_by TEXT,
            status TEXT DEFAULT 'pending',
            decline_reason TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # OAuth users table
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS oauth_users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT,
            avatar_url TEXT,
            session_token TEXT,
            role TEXT DEFAULT 'user',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tickets table
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            status TEXT DEFAULT 'open',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            closed_at DATETIME,
            FOREIGN KEY (request_id) REFERENCES requests(id)
        )
    ''')
    
    # Ticket messages table
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS ticket_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id INTEGER NOT NULL,
            sender TEXT NOT NULL,
            sender_role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticket_id) REFERENCES tickets(id)
        )
    ''')
    
    # Migration: Add decline_reason column if it doesn't exist (handled by IF NOT EXISTS in schema)
    # Migration: Add role column to oauth_users if it doesn't exist (handled by IF NOT EXISTS in schema)


async def _init_mssql_tables(conn: AsyncDatabaseConnection):
    """Initialize MSSQL tables."""
    
    # Users table
    await conn.execute('''
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='users' AND xtype='U')
        CREATE TABLE users (
            id INT PRIMARY KEY IDENTITY(1,1),
            username NVARCHAR(255) UNIQUE NOT NULL,
            hashed_password NVARCHAR(MAX) NOT NULL,
            role NVARCHAR(50) NOT NULL,
            api_key NVARCHAR(255)
        )
    ''')
    await conn.commit()
    
    # Models table
    await conn.execute('''
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='models' AND xtype='U')
        CREATE TABLE models (
            id NVARCHAR(255) PRIMARY KEY,
            hf_repo_id NVARCHAR(500) NOT NULL,
            status NVARCHAR(50) NOT NULL,
            progress INT DEFAULT 0,
            log NVARCHAR(MAX) DEFAULT '',
            error_details NVARCHAR(MAX) DEFAULT '',
            created_at DATETIME DEFAULT GETDATE(),
            completed_at DATETIME
        )
    ''')
    await conn.commit()
    
    # Requests table
    await conn.execute('''
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='requests' AND xtype='U')
        CREATE TABLE requests (
            id INT PRIMARY KEY IDENTITY(1,1),
            hf_repo_id NVARCHAR(500) NOT NULL,
            requested_by NVARCHAR(255),
            status NVARCHAR(50) DEFAULT 'pending',
            decline_reason NVARCHAR(MAX) DEFAULT '',
            created_at DATETIME DEFAULT GETDATE()
        )
    ''')
    await conn.commit()
    
    # OAuth users table
    await conn.execute('''
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='oauth_users' AND xtype='U')
        CREATE TABLE oauth_users (
            id NVARCHAR(255) PRIMARY KEY,
            username NVARCHAR(255) NOT NULL,
            email NVARCHAR(255),
            avatar_url NVARCHAR(500),
            session_token NVARCHAR(255),
            role NVARCHAR(50) DEFAULT 'user',
            created_at DATETIME DEFAULT GETDATE()
        )
    ''')
    await conn.commit()
    
    # Migration: Add role column to oauth_users if it doesn't exist
    try:
        await conn.execute('''
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'oauth_users' AND COLUMN_NAME = 'role')
            ALTER TABLE oauth_users ADD role NVARCHAR(50) DEFAULT 'user'
        ''')
        await conn.commit()
    except:
        pass
    
    # Tickets table
    await conn.execute('''
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='tickets' AND xtype='U')
        CREATE TABLE tickets (
            id INT PRIMARY KEY IDENTITY(1,1),
            request_id INT NOT NULL,
            status NVARCHAR(50) DEFAULT 'open',
            created_at DATETIME DEFAULT GETDATE(),
            closed_at DATETIME,
            FOREIGN KEY (request_id) REFERENCES requests(id)
        )
    ''')
    await conn.commit()
    
    # Ticket messages table
    await conn.execute('''
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ticket_messages' AND xtype='U')
        CREATE TABLE ticket_messages (
            id INT PRIMARY KEY IDENTITY(1,1),
            ticket_id INT NOT NULL,
            sender NVARCHAR(255) NOT NULL,
            sender_role NVARCHAR(50) NOT NULL,
            message NVARCHAR(MAX) NOT NULL,
            created_at DATETIME DEFAULT GETDATE(),
            FOREIGN KEY (ticket_id) REFERENCES tickets(id)
        )
    ''')
    await conn.commit()


async def test_connection() -> tuple[bool, str]:
    """Test database connection. Returns (success, message)."""
    try:
        conn = await get_db_connection()
        await conn.execute("SELECT 1")
        await conn.close()
        return True, f"Successfully connected to {DB_TYPE} database"
    except Exception as e:
        return False, f"Failed to connect to {DB_TYPE} database: {str(e)}"
