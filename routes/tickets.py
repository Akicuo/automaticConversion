"""
Ticket/conversation system routes for GGUF Forge.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks

from database import get_db_connection
from models import TicketMessage, CreateTicketRequest

router = APIRouter(prefix="/api/tickets")

# Dependencies - will be set by main app
_require_admin_func = None
_get_current_user_func = None


def configure(admin_dep, user_dep):
    """Configure routes with dependencies."""
    global _require_admin_func, _get_current_user_func
    _require_admin_func = admin_dep
    _get_current_user_func = user_dep


def get_admin(request: Request):
    """Dependency that uses the configured admin check."""
    if _require_admin_func:
        return _require_admin_func(request)
    raise HTTPException(status_code=500, detail="Admin dependency not configured")


@router.post("/create")
async def create_ticket(data: CreateTicketRequest, request: Request, user = Depends(get_admin)):
    """Admin only: Create a ticket/thread for a request instead of rejecting it."""
    conn = get_db_connection()
    
    # Check if request exists
    req = conn.execute("SELECT * FROM requests WHERE id = ?", (data.request_id,)).fetchone()
    if not req:
        conn.close()
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Check if ticket already exists for this request
    existing_ticket = conn.execute(
        "SELECT * FROM tickets WHERE request_id = ?", (data.request_id,)
    ).fetchone()
    if existing_ticket:
        conn.close()
        return {"status": "exists", "ticket_id": existing_ticket['id'], "message": "Ticket already exists for this request"}
    
    # Update request status to 'discussion'
    conn.execute("UPDATE requests SET status = 'discussion' WHERE id = ?", (data.request_id,))
    
    # Create ticket
    cursor = conn.execute(
        "INSERT INTO tickets (request_id, status) VALUES (?, ?)",
        (data.request_id, "open")
    )
    ticket_id = cursor.lastrowid
    
    # Add initial message if provided
    if data.initial_message:
        conn.execute(
            "INSERT INTO ticket_messages (ticket_id, sender, sender_role, message) VALUES (?, ?, ?, ?)",
            (ticket_id, user['username'], 'admin', data.initial_message)
        )
    
    conn.commit()
    conn.close()
    
    return {"status": "created", "ticket_id": ticket_id}


@router.get("/request/{request_id}")
async def get_ticket_by_request(request_id: int, request: Request):
    """Get ticket for a specific request."""
    user = _get_current_user_func(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    
    conn = get_db_connection()
    
    # Get request first to check permission
    req = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
    if not req:
        conn.close()
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Check permission: admin can see all, users can only see their own
    if user['role'] != 'admin' and req['requested_by'] != user['username']:
        conn.close()
        raise HTTPException(status_code=403, detail="Access denied")
    
    ticket = conn.execute(
        "SELECT * FROM tickets WHERE request_id = ?", (request_id,)
    ).fetchone()
    
    if not ticket:
        conn.close()
        return {"ticket": None}
    
    conn.close()
    return {"ticket": dict(ticket)}


@router.get("/{ticket_id}/messages")
async def get_ticket_messages(ticket_id: int, request: Request):
    """Get all messages for a ticket."""
    user = _get_current_user_func(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    
    conn = get_db_connection()
    
    # Get ticket to check permission
    ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if not ticket:
        conn.close()
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Get associated request
    req = conn.execute("SELECT * FROM requests WHERE id = ?", (ticket['request_id'],)).fetchone()
    
    # Check permission
    if user['role'] != 'admin' and req['requested_by'] != user['username']:
        conn.close()
        raise HTTPException(status_code=403, detail="Access denied")
    
    messages = conn.execute(
        "SELECT * FROM ticket_messages WHERE ticket_id = ? ORDER BY created_at ASC",
        (ticket_id,)
    ).fetchall()
    conn.close()
    
    return {"messages": [dict(m) for m in messages], "ticket": dict(ticket), "request": dict(req)}


@router.post("/{ticket_id}/reply")
async def reply_to_ticket(ticket_id: int, data: TicketMessage, request: Request):
    """Add a message to a ticket thread."""
    user = _get_current_user_func(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    
    conn = get_db_connection()
    
    # Get ticket
    ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if not ticket:
        conn.close()
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    if ticket['status'] == 'closed':
        conn.close()
        raise HTTPException(status_code=400, detail="Ticket is closed. Reopen it first to reply.")
    
    # Get associated request
    req = conn.execute("SELECT * FROM requests WHERE id = ?", (ticket['request_id'],)).fetchone()
    
    # Check permission
    if user['role'] != 'admin' and req['requested_by'] != user['username']:
        conn.close()
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Add message
    conn.execute(
        "INSERT INTO ticket_messages (ticket_id, sender, sender_role, message) VALUES (?, ?, ?, ?)",
        (ticket_id, user['username'], user['role'], data.message)
    )
    conn.commit()
    conn.close()
    
    return {"status": "sent"}


@router.post("/{ticket_id}/close")
async def close_ticket(ticket_id: int, user = Depends(get_admin)):
    """Admin only: Close a ticket."""
    conn = get_db_connection()
    
    ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if not ticket:
        conn.close()
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    conn.execute(
        "UPDATE tickets SET status = 'closed', closed_at = ? WHERE id = ?",
        (datetime.now().isoformat(), ticket_id)
    )
    
    # Also update the request status
    conn.execute(
        "UPDATE requests SET status = 'rejected', decline_reason = 'Closed after discussion' WHERE id = ?",
        (ticket['request_id'],)
    )
    
    conn.commit()
    conn.close()
    
    return {"status": "closed"}


@router.post("/{ticket_id}/approve")
async def approve_ticket(ticket_id: int, background_tasks: BackgroundTasks, user = Depends(get_admin)):
    """Admin only: Approve a ticket - starts conversion and closes the ticket."""
    import uuid
    from workflow import ModelWorkflow
    
    conn = get_db_connection()
    
    ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if not ticket:
        conn.close()
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Get associated request
    req = conn.execute("SELECT * FROM requests WHERE id = ?", (ticket['request_id'],)).fetchone()
    if not req:
        conn.close()
        raise HTTPException(status_code=404, detail="Associated request not found")
    
    # Close the ticket
    conn.execute(
        "UPDATE tickets SET status = 'closed', closed_at = ? WHERE id = ?",
        (datetime.now().isoformat(), ticket_id)
    )
    
    # Approve the request
    conn.execute("UPDATE requests SET status = 'approved' WHERE id = ?", (ticket['request_id'],))
    
    # Start the conversion
    hf_repo_id = req['hf_repo_id']
    new_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO models (id, hf_repo_id, status, progress, log, error_details) VALUES (?, ?, ?, ?, ?, ?)",
        (new_id, hf_repo_id, "pending", 0, "Queued from approved discussion...", "")
    )
    conn.commit()
    conn.close()
    
    workflow = ModelWorkflow(new_id, hf_repo_id)
    background_tasks.add_task(workflow.run_pipeline)
    
    return {"status": "approved", "model_id": new_id}


@router.post("/{ticket_id}/reopen")
async def reopen_ticket(ticket_id: int, request: Request):
    """Reopen a closed ticket (user or admin)."""
    user = _get_current_user_func(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    
    conn = get_db_connection()
    
    ticket = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if not ticket:
        conn.close()
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Get associated request to check permission
    req = conn.execute("SELECT * FROM requests WHERE id = ?", (ticket['request_id'],)).fetchone()
    
    # Check permission: admin or owner can reopen
    if user['role'] != 'admin' and req['requested_by'] != user['username']:
        conn.close()
        raise HTTPException(status_code=403, detail="Access denied")
    
    conn.execute(
        "UPDATE tickets SET status = 'open', closed_at = NULL WHERE id = ?",
        (ticket_id,)
    )
    
    # Update request status back to discussion
    conn.execute(
        "UPDATE requests SET status = 'discussion' WHERE id = ?",
        (ticket['request_id'],)
    )
    
    conn.commit()
    conn.close()
    
    return {"status": "reopened"}


@router.get("/all")
async def get_all_tickets(user = Depends(get_admin)):
    """Admin only: Get all tickets."""
    conn = get_db_connection()
    tickets = conn.execute("""
        SELECT t.*, r.hf_repo_id, r.requested_by 
        FROM tickets t 
        JOIN requests r ON t.request_id = r.id 
        ORDER BY t.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(t) for t in tickets]


@router.get("/my")
async def get_my_tickets(request: Request):
    """Get current user's tickets."""
    user = _get_current_user_func(request)
    if not user:
        return []
    
    conn = get_db_connection()
    tickets = conn.execute("""
        SELECT t.*, r.hf_repo_id, r.requested_by 
        FROM tickets t 
        JOIN requests r ON t.request_id = r.id 
        WHERE r.requested_by = ?
        ORDER BY t.created_at DESC
    """, (user['username'],)).fetchall()
    conn.close()
    return [dict(t) for t in tickets]
