"""
Authentication routes for GGUF Forge.
"""
import os
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext

from database import get_db_connection

router = APIRouter()

# These will be set by main app
templates = None
pwd_context = None
OAUTH_CLIENT_ID = None
OAUTH_CLIENT_SECRET = None
OAUTH_REDIRECT_URI = None
OAUTH_AUTHORIZE_URL = "https://huggingface.co/oauth/authorize"
OAUTH_TOKEN_URL = "https://huggingface.co/oauth/token"
OAUTH_USERINFO_URL = "https://huggingface.co/oauth/userinfo"


def configure(tmpl: Jinja2Templates, pwd_ctx: CryptContext, client_id: str, client_secret: str, redirect_uri: str):
    """Configure authentication routes."""
    global templates, pwd_context, OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, OAUTH_REDIRECT_URI
    templates = tmpl
    pwd_context = pwd_ctx
    OAUTH_CLIENT_ID = client_id
    OAUTH_CLIENT_SECRET = client_secret
    OAUTH_REDIRECT_URI = redirect_uri


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = await conn.fetchone()
    await conn.close()
    
    if user and pwd_context.verify(password, user['hashed_password']):
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key="session_token",
            value=user["api_key"],
            httponly=True,
            samesite="lax",
            secure=(request.url.scheme == "https"),
            max_age=86400 * 7,  # 7 days
        )
        return response
    
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("session_token")
    return response


# --- HuggingFace OAuth Routes ---
@router.get("/auth/login")
async def oauth_login():
    """Redirect to HuggingFace OAuth authorization page."""
    if not OAUTH_CLIENT_ID:
        raise HTTPException(status_code=500, detail="OAuth not configured. Set OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET in .env")
    
    params = {
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "scope": "openid profile email",
        "response_type": "code",
        "state": secrets.token_urlsafe(16)
    }
    auth_url = f"{OAUTH_AUTHORIZE_URL}?{urlencode(params)}"
    return RedirectResponse(url=auth_url)


@router.get("/auth/callback")
async def oauth_callback(request: Request, code: str, state: str = None):
    """Handle HuggingFace OAuth callback."""
    import logging
    logger = logging.getLogger("GGUF_Forge")
    
    if not OAUTH_CLIENT_ID or not OAUTH_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="OAuth not configured")
    
    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            OAUTH_TOKEN_URL,
            data={
                "client_id": OAUTH_CLIENT_ID,
                "client_secret": OAUTH_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": OAUTH_REDIRECT_URI
            }
        )
        
        if token_response.status_code != 200:
            logger.error(f"OAuth token error: {token_response.text}")
            raise HTTPException(status_code=400, detail="Failed to authenticate with HuggingFace")
        
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        
        # Get user info
        userinfo_response = await client.get(
            OAUTH_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if userinfo_response.status_code != 200:
            logger.error(f"OAuth userinfo error: {userinfo_response.text}")
            raise HTTPException(status_code=400, detail="Failed to get user info from HuggingFace")
        
        userinfo = userinfo_response.json()
    
    # Extract user data
    hf_user_id = userinfo.get("sub")
    username = userinfo.get("preferred_username") or userinfo.get("name") or f"user_{hf_user_id[:8]}"
    email = userinfo.get("email")
    avatar_url = userinfo.get("picture")
    
    # Create session token
    session_token = secrets.token_urlsafe(32)
    
    # Check if user should be admin (from ADMIN_USERS env variable)
    from database import is_admin_user
    role = "admin" if is_admin_user(username) else "user"
    
    # Save or update user in database
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM oauth_users WHERE id = ?", (hf_user_id,))
    existing = await conn.fetchone()
    
    if existing:
        # Update existing user - also update role in case admin list changed
        await conn.execute(
            "UPDATE oauth_users SET username = ?, email = ?, avatar_url = ?, session_token = ?, role = ? WHERE id = ?",
            (username, email, avatar_url, session_token, role, hf_user_id)
        )
        if role == "admin":
            logger.info(f"Admin user logged in: {username}")
    else:
        await conn.execute(
            "INSERT INTO oauth_users (id, username, email, avatar_url, session_token, role) VALUES (?, ?, ?, ?, ?, ?)",
            (hf_user_id, username, email, avatar_url, session_token, role)
        )
        if role == "admin":
            logger.info(f"New admin user created: {username}")
    
    await conn.commit()
    await conn.close()
    
    # Set session cookie and redirect
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        samesite="lax",
        secure=(request.url.scheme == "https"),
        max_age=86400 * 7,  # 7 days
    )
    return response


@router.get("/auth/logout")
async def oauth_logout(request: Request):
    """Clear OAuth session."""
    token = request.cookies.get("session_token")
    if token:
        conn = await get_db_connection()
        await conn.execute("UPDATE oauth_users SET session_token = NULL WHERE session_token = ?", (token,))
        await conn.commit()
        await conn.close()
    
    response = RedirectResponse(url="/")
    response.delete_cookie("session_token")
    return response
