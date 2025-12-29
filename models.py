"""
Pydantic models for API requests/responses.
"""
from typing import Optional
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class ProcessRequest(BaseModel):
    model_id: str


class ModelRequestSubmit(BaseModel):
    hf_repo_id: str


class RejectRequest(BaseModel):
    reason: Optional[str] = ""


class TicketMessage(BaseModel):
    message: str


class CreateTicketRequest(BaseModel):
    request_id: int
    initial_message: Optional[str] = ""
