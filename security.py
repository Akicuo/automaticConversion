"""
Security modules for GGUF Forge: Rate limiting, bot detection, spam protection.
"""
import asyncio
from database import get_db_connection


class RateLimiter:
    """IP-based rate limiter to prevent DDoS attacks."""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_second: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.requests: dict = {}  # IP -> list of timestamps
        self.blocked_ips: dict = {}  # IP -> block_until timestamp
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, ip: str) -> tuple[bool, str]:
        """Check if request from IP is allowed. Returns (allowed, reason)."""
        import time
        now = time.time()
        
        async with self.lock:
            # Check if IP is blocked
            if ip in self.blocked_ips:
                if now < self.blocked_ips[ip]:
                    remaining = int(self.blocked_ips[ip] - now)
                    return False, f"IP blocked for {remaining}s due to rate limit abuse"
                else:
                    del self.blocked_ips[ip]
            
            # Initialize or clean old requests
            if ip not in self.requests:
                self.requests[ip] = []
            
            # Remove requests older than 1 minute
            self.requests[ip] = [t for t in self.requests[ip] if now - t < 60]
            
            # Check per-second rate
            recent_second = sum(1 for t in self.requests[ip] if now - t < 1)
            if recent_second >= self.requests_per_second:
                return False, "Too many requests per second"
            
            # Check per-minute rate
            if len(self.requests[ip]) >= self.requests_per_minute:
                # Block IP for 5 minutes
                self.blocked_ips[ip] = now + 300
                return False, "Rate limit exceeded. IP blocked for 5 minutes"
            
            # Allow request
            self.requests[ip].append(now)
            return True, ""
    
    async def cleanup(self):
        """Remove old entries to prevent memory bloat."""
        import time
        now = time.time()
        async with self.lock:
            # Clean request history
            for ip in list(self.requests.keys()):
                self.requests[ip] = [t for t in self.requests[ip] if now - t < 60]
                if not self.requests[ip]:
                    del self.requests[ip]
            # Clean expired blocks
            for ip in list(self.blocked_ips.keys()):
                if now >= self.blocked_ips[ip]:
                    del self.blocked_ips[ip]


class BotDetector:
    """Detect and block suspicious bot traffic."""
    
    # Known bot/suspicious user agent patterns
    SUSPICIOUS_PATTERNS = [
        "curl", "wget", "python-requests", "scrapy", "bot", "spider",
        "crawler", "scan", "http", "java/", "perl", "ruby", "go-http",
        "aiohttp", "httpx", "axios", "node-fetch", "undici"
    ]
    
    # Legitimate browser patterns
    BROWSER_PATTERNS = ["mozilla", "chrome", "safari", "firefox", "edge", "opera"]
    
    @classmethod
    def is_suspicious(cls, user_agent: str, path: str) -> tuple[bool, str]:
        """Check if request appears to be from a bot. Returns (is_bot, reason)."""
        if not user_agent:
            return True, "Missing User-Agent header"
        
        ua_lower = user_agent.lower()
        
        # Skip check for API endpoints that might legitimately use non-browser clients
        if path.startswith("/api/"):
            return False, ""
        
        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if pattern in ua_lower:
                return True, f"Suspicious User-Agent pattern: {pattern}"
        
        # For non-API routes, require browser-like user agent
        has_browser = any(pattern in ua_lower for pattern in cls.BROWSER_PATTERNS)
        if not has_browser and not path.startswith("/api/"):
            return True, "Non-browser User-Agent for browser route"
        
        return False, ""


class SpamProtection:
    """Prevent spam submissions (model requests, etc.)."""
    
    def __init__(self, max_requests_per_hour: int = 10, max_pending_per_user: int = 5):
        self.max_requests_per_hour = max_requests_per_hour
        self.max_pending_per_user = max_pending_per_user
        self.submissions: dict = {}  # user -> list of timestamps
        self.lock = asyncio.Lock()
    
    async def can_submit(self, username: str) -> tuple[bool, str]:
        """Check if user can submit a new request."""
        import time
        now = time.time()
        
        async with self.lock:
            if username not in self.submissions:
                self.submissions[username] = []
            
            # Remove submissions older than 1 hour
            self.submissions[username] = [t for t in self.submissions[username] if now - t < 3600]
            
            if len(self.submissions[username]) >= self.max_requests_per_hour:
                return False, f"Request limit reached ({self.max_requests_per_hour}/hour). Please try again later."
            
            return True, ""
    
    async def record_submission(self, username: str):
        """Record a new submission."""
        import time
        async with self.lock:
            if username not in self.submissions:
                self.submissions[username] = []
            self.submissions[username].append(time.time())
    
    async def check_pending_limit(self, username: str) -> tuple[bool, str]:
        """Check if user has too many pending requests (async DB check)."""
        conn = await get_db_connection()
        await conn.execute(
            "SELECT COUNT(*) as cnt FROM requests WHERE requested_by = ? AND status = 'pending'",
            (username,)
        )
        result = await conn.fetchone()
        await conn.close()
        
        count = result['cnt'] if result else 0
        
        if count >= self.max_pending_per_user:
            return False, f"Too many pending requests ({count}/{self.max_pending_per_user}). Wait for approval."
        return True, ""
