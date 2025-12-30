"""
Request system routes for GGUF Forge.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks

from database import get_db_connection
from models import ModelRequestSubmit, RejectRequest
from workflow import ModelWorkflow
from websocket_manager import broadcast_requests_update, broadcast_my_requests_update

router = APIRouter(prefix="/api/requests")

# Dependencies - will be set by main app
_require_admin_func = None
_get_current_user_func = None
_spam_protection = None


def configure(admin_dep, user_dep, spam_prot):
    """Configure routes with dependencies."""
    global _require_admin_func, _get_current_user_func, _spam_protection
    _require_admin_func = admin_dep
    _get_current_user_func = user_dep
    _spam_protection = spam_prot


async def get_admin(request: Request):
    """Dependency that uses the configured admin check."""
    if _require_admin_func:
        return await _require_admin_func(request)
    raise HTTPException(status_code=500, detail="Admin dependency not configured")


@router.post("/submit")
async def submit_request(req: ModelRequestSubmit, request: Request):
    """Submit a request for a model to be converted (requires login)."""
    user = await _get_current_user_func(request)
    
    # Require login for submissions (anti-spam)
    if not user:
        raise HTTPException(status_code=401, detail="Login required to submit requests")
    
    requester = user['username']
    
    # Check spam protection - pending limit
    allowed, reason = await _spam_protection.check_pending_limit(requester)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    
    # Check spam protection - hourly rate
    allowed, reason = await _spam_protection.can_submit(requester)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    
    conn = await get_db_connection()
    # Check if already requested
    await conn.execute("SELECT * FROM requests WHERE hf_repo_id = ? AND status = 'pending'", (req.hf_repo_id,))
    existing = await conn.fetchone()
    if existing:
        await conn.close()
        return {"status": "already_requested", "message": "This model has already been requested."}
    
    await conn.execute(
        "INSERT INTO requests (hf_repo_id, requested_by, status) VALUES (?, ?, ?)",
        (req.hf_repo_id, requester, "pending")
    )
    await conn.commit()
    await conn.close()
    
    # Record submission for rate limiting
    await _spam_protection.record_submission(requester)
    
    # Broadcast update via WebSocket
    await broadcast_requests_update()
    await broadcast_my_requests_update()
    
    return {"status": "submitted", "message": "Your request has been submitted for admin review."}


@router.get("/all")
async def get_all_requests(user = Depends(get_admin)):
    """Admin only: View all pending requests."""
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM requests ORDER BY created_at DESC")
    requests = await conn.fetchall()
    await conn.close()
    return [dict(r) for r in requests]


@router.post("/{request_id}/approve")
async def approve_request(request_id: int, background_tasks: BackgroundTasks, user = Depends(get_admin)):
    """Admin only: Approve a request and start conversion."""
    conn = await get_db_connection()
    await conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,))
    req = await conn.fetchone()
    
    if not req:
        await conn.close()
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Update request status
    await conn.execute("UPDATE requests SET status = 'approved' WHERE id = ?", (request_id,))
    await conn.commit()
    
    # Start the conversion
    hf_repo_id = req['hf_repo_id']
    new_id = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO models (id, hf_repo_id, status, progress, log, error_details) VALUES (?, ?, ?, ?, ?, ?)",
        (new_id, hf_repo_id, "pending", 0, "Queued from approved request...", "")
    )
    await conn.commit()
    await conn.close()
    
    workflow = ModelWorkflow(new_id, hf_repo_id)
    background_tasks.add_task(workflow.run_pipeline)
    
    # Broadcast update via WebSocket
    await broadcast_requests_update()
    await broadcast_my_requests_update()
    
    return {"status": "approved", "model_id": new_id}


@router.post("/{request_id}/reject")
async def reject_request(request_id: int, body: RejectRequest = None, user = Depends(get_admin)):
    """Admin only: Reject a request with optional reason."""
    reason = body.reason if body else ""
    conn = await get_db_connection()
    await conn.execute("UPDATE requests SET status = 'rejected', decline_reason = ? WHERE id = ?", (reason, request_id))
    await conn.commit()
    await conn.close()
    
    # Broadcast update via WebSocket
    await broadcast_requests_update()
    await broadcast_my_requests_update()
    
    return {"status": "rejected"}


@router.get("/my")
async def get_my_requests(request: Request):
    """Get current user's request history."""
    user = await _get_current_user_func(request)
    if not user:
        return []
    
    username = user['username']
    conn = await get_db_connection()
    await conn.execute(
        "SELECT * FROM requests WHERE requested_by = ? ORDER BY created_at DESC",
        (username,)
    )
    requests = await conn.fetchall()
    await conn.close()
    return [dict(r) for r in requests]
