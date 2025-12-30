"""
WebSocket manager for real-time updates in GGUF Forge.
"""
import json
import asyncio
import logging
from typing import Dict, Set, Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("GGUF_Forge")


class ConnectionManager:
    """Manages WebSocket connections and broadcasts updates to connected clients."""
    
    def __init__(self):
        # Active connections by channel
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "models": set(),      # Model conversion status updates
            "requests": set(),    # Request updates (admin)
            "tickets": set(),     # Ticket updates
            "my_requests": set(), # User's own requests
        }
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, channels: list = None):
        """Accept connection and add to specified channels."""
        await websocket.accept()
        async with self._lock:
            if channels is None:
                channels = ["models"]  # Default channel
            for channel in channels:
                if channel in self.active_connections:
                    self.active_connections[channel].add(websocket)
        logger.debug(f"WebSocket connected to channels: {channels}")
    
    async def disconnect(self, websocket: WebSocket):
        """Remove connection from all channels."""
        async with self._lock:
            for channel in self.active_connections.values():
                channel.discard(websocket)
    
    async def broadcast(self, channel: str, data: Any):
        """Broadcast message to all connections in a channel."""
        if channel not in self.active_connections:
            return
        
        message = json.dumps({"channel": channel, "data": data})
        dead_connections = set()
        
        async with self._lock:
            connections = self.active_connections[channel].copy()
        
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead_connections.add(connection)
        
        # Clean up dead connections
        if dead_connections:
            async with self._lock:
                for channel_set in self.active_connections.values():
                    channel_set -= dead_connections
    
    async def broadcast_all(self, data: Any):
        """Broadcast to all channels (for general updates)."""
        for channel in self.active_connections.keys():
            await self.broadcast(channel, data)
    
    def get_connection_count(self, channel: str = None) -> int:
        """Get number of active connections."""
        if channel:
            return len(self.active_connections.get(channel, set()))
        return sum(len(conns) for conns in self.active_connections.values())


# Global connection manager instance
manager = ConnectionManager()


async def broadcast_model_update(model_data: dict):
    """Broadcast model status update to all connected clients."""
    await manager.broadcast("models", {
        "type": "model_update",
        "model": model_data
    })


async def broadcast_models_list(models: list):
    """Broadcast full models list update."""
    await manager.broadcast("models", {
        "type": "models_list",
        "models": models
    })


async def broadcast_requests_update():
    """Signal clients to refresh requests."""
    await manager.broadcast("requests", {
        "type": "requests_update"
    })


async def broadcast_tickets_update():
    """Signal clients to refresh tickets."""
    await manager.broadcast("tickets", {
        "type": "tickets_update"
    })


async def broadcast_my_requests_update():
    """Signal clients to refresh their own requests."""
    await manager.broadcast("my_requests", {
        "type": "my_requests_update"
    })
