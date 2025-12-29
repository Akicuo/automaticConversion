"""
Model conversion routes for GGUF Forge.
"""
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request

from database import get_db_connection
from models import ProcessRequest
from workflow import ModelWorkflow, running_workflows
from managers import HuggingFaceManager

router = APIRouter(prefix="/api")

# Dependency - will be set by main app
_require_admin_func = None


def configure(admin_dependency):
    """Configure routes with dependencies."""
    global _require_admin_func
    _require_admin_func = admin_dependency


def get_admin(request: Request):
    """Dependency that uses the configured admin check."""
    if _require_admin_func:
        return _require_admin_func(request)
    raise HTTPException(status_code=500, detail="Admin dependency not configured")


@router.get("/hf/search")
async def search_hf(q: str):
    mgr = HuggingFaceManager(token=os.getenv("HF_TOKEN"))
    try:
        return mgr.search_models(q)
    except Exception as e:
        return []


@router.post("/models/process")
async def process_model(req: ProcessRequest, background_tasks: BackgroundTasks, user = Depends(get_admin)):
    conn = get_db_connection()
    existing = conn.execute("SELECT * FROM models WHERE hf_repo_id = ?", (req.model_id,)).fetchone()
    
    if existing and existing['status'] in ['pending', 'downloading', 'converting', 'quantizing', 'initializing']:
         conn.close()
         raise HTTPException(status_code=400, detail="Model already processing")
    
    new_id = str(uuid.uuid4())
    conn.execute(
        "INSERT OR REPLACE INTO models (id, hf_repo_id, status, progress, log, error_details) VALUES (?, ?, ?, ?, ?, ?)",
        (new_id, req.model_id, "pending", 0, "Queued...", "")
    )
    conn.commit()
    conn.close()
    
    workflow = ModelWorkflow(new_id, req.model_id)
    background_tasks.add_task(workflow.run_pipeline)
    
    return {"status": "started", "id": new_id}


@router.get("/status/all")
async def get_all_status():
    conn = get_db_connection()
    models = conn.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT 50").fetchall()
    conn.close()
    return [dict(m) for m in models]


@router.get("/status/model/{model_id}")
async def get_model_status(model_id: str):
    conn = get_db_connection()
    model = conn.execute("SELECT * FROM models WHERE hf_repo_id = ?", (model_id,)).fetchone()
    conn.close()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return dict(model)


@router.delete("/models/{model_id}")
async def delete_model_record(model_id: str, user = Depends(get_admin)):
    """Admin only: Delete a model record from the database."""
    conn = get_db_connection()
    conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


@router.post("/models/{model_id}/terminate")
async def terminate_model(model_id: str, user = Depends(get_admin)):
    """Admin only: Terminate a running job."""
    # Check if this workflow is currently running
    if model_id not in running_workflows:
        # Check if the job exists in DB
        conn = get_db_connection()
        model = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        conn.close()
        
        if not model:
            raise HTTPException(status_code=404, detail="Job not found")
        if model['status'] in ['complete', 'error', 'terminated', 'interrupted']:
            raise HTTPException(status_code=400, detail=f"Job already finished with status: {model['status']}")
        
        # Job exists but not in running_workflows - mark as terminated directly
        conn = get_db_connection()
        conn.execute(
            "UPDATE models SET status = ?, error_details = ? WHERE id = ?",
            ("terminated", "Terminated by administrator (job was not actively running)", model_id)
        )
        conn.commit()
        conn.close()
        return {"status": "terminated", "message": "Job marked as terminated"}
    
    # Terminate the running workflow
    workflow = running_workflows[model_id]
    workflow.terminate()
    
    return {"status": "terminating", "message": "Termination signal sent. Job will stop shortly."}
