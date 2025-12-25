"""HTTP API for Memory Service.

Provides REST endpoints for session management:
- List sessions
- Create/switch sessions
- Get/delete sessions
- Manage messages within sessions
"""

from datetime import datetime
from typing import Any, Callable, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ai_assistant.memory import MemoryManager
from ai_assistant.shared.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class SessionCreate(BaseModel):
    """Request body for creating a new session."""

    set_active: bool = Field(
        default=True,
        description="Whether to set the new session as active",
    )


class SessionSwitch(BaseModel):
    """Request body for switching the active session."""

    session_id: str = Field(..., description="Session ID to switch to")


class SessionInfo(BaseModel):
    """Session information response."""

    session_id: str
    message_count: int = 0
    created_at: Optional[str] = None
    last_activity: Optional[str] = None


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    sessions: list[SessionInfo]
    total: int
    limit: int
    offset: int


class MessageInfo(BaseModel):
    """Message information."""

    id: str
    session_id: str
    role: str
    content: str
    timestamp: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageListResponse(BaseModel):
    """Response for listing messages."""

    messages: list[MessageInfo]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str = "memory-service"
    current_session: Optional[str] = None


class DeleteResponse(BaseModel):
    """Response for delete operations."""

    deleted_count: int
    message: str


# =============================================================================
# API Factory
# =============================================================================


def create_api(
    memory: MemoryManager,
    get_current_session: Callable[[], str],
    set_current_session: Callable[[str], None],
) -> FastAPI:
    """Create the FastAPI application for memory service.

    Args:
        memory: MemoryManager instance for database operations
        get_current_session: Callback to get the current active session ID
        set_current_session: Callback to set the current active session ID
            (this callback should also publish session changed events)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Memory Service API",
        description="REST API for session and memory management",
        version="1.0.0",
    )

    # =========================================================================
    # Health Check
    # =========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        """Check service health and get current session."""
        return HealthResponse(
            status="healthy",
            current_session=get_current_session(),
        )

    # =========================================================================
    # Session Endpoints
    # =========================================================================

    @app.get("/sessions", response_model=SessionListResponse, tags=["Sessions"])
    async def list_sessions(
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> SessionListResponse:
        """List all sessions with pagination."""
        try:
            sessions, total = memory.list_sessions(limit=limit, offset=offset)

            session_list = []
            for s in sessions:
                session_list.append(
                    SessionInfo(
                        session_id=s.get("session_id", ""),
                        message_count=s.get("message_count", 0),
                        created_at=s.get("first_message_at"),
                        last_activity=s.get("last_message_at"),
                    )
                )

            return SessionListResponse(
                sessions=session_list,
                total=total,
                limit=limit,
                offset=offset,
            )
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/sessions", response_model=SessionInfo, tags=["Sessions"])
    async def create_session(body: SessionCreate = SessionCreate()) -> SessionInfo:
        """Create a new session and optionally set it as active."""
        try:
            session_id = str(uuid4())
            created_at = datetime.utcnow().isoformat()

            if body.set_active:
                # set_current_session automatically publishes session changed event
                set_current_session(session_id)
                logger.info(f"Created and activated new session: {session_id}")
            else:
                logger.info(f"Created new session (not active): {session_id}")

            return SessionInfo(
                session_id=session_id,
                message_count=0,
                created_at=created_at,
                last_activity=created_at,
            )
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sessions/current", response_model=SessionInfo, tags=["Sessions"])
    async def get_current_session_info() -> SessionInfo:
        """Get information about the current active session."""
        try:
            session_id = get_current_session()
            info = memory.get_session_info(session_id)

            if info:
                return SessionInfo(
                    session_id=session_id,
                    message_count=info.get("message_count", 0),
                    created_at=info.get("first_message_at"),
                    last_activity=info.get("last_message_at"),
                )
            else:
                # Session exists but has no messages yet
                return SessionInfo(
                    session_id=session_id,
                    message_count=0,
                    created_at=None,
                    last_activity=None,
                )
        except Exception as e:
            logger.error(f"Error getting current session: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/sessions/current", response_model=SessionInfo, tags=["Sessions"])
    async def switch_active_session(body: SessionSwitch) -> SessionInfo:
        """Switch to a different active session."""
        try:
            new_session_id = body.session_id
            old_session_id = get_current_session()

            if new_session_id == old_session_id:
                # Already active, just return info
                info = memory.get_session_info(new_session_id)
                return SessionInfo(
                    session_id=new_session_id,
                    message_count=info.get("message_count", 0) if info else 0,
                    created_at=info.get("first_message_at") if info else None,
                    last_activity=info.get("last_message_at") if info else None,
                )

            # Switch session (automatically publishes session changed event)
            set_current_session(new_session_id)

            logger.info(f"Switched session: {old_session_id} -> {new_session_id}")

            # Get info about new session
            info = memory.get_session_info(new_session_id)
            return SessionInfo(
                session_id=new_session_id,
                message_count=info.get("message_count", 0) if info else 0,
                created_at=info.get("first_message_at") if info else None,
                last_activity=info.get("last_message_at") if info else None,
            )
        except Exception as e:
            logger.error(f"Error switching session: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
    async def get_session(session_id: str) -> SessionInfo:
        """Get information about a specific session."""
        try:
            info = memory.get_session_info(session_id)

            if not info:
                raise HTTPException(status_code=404, detail="Session not found")

            return SessionInfo(
                session_id=session_id,
                message_count=info.get("message_count", 0),
                created_at=info.get("first_message_at"),
                last_activity=info.get("last_message_at"),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/sessions/{session_id}", response_model=DeleteResponse, tags=["Sessions"])
    async def delete_session(session_id: str) -> DeleteResponse:
        """Delete a session and all its messages."""
        try:
            # Check if trying to delete current session
            current = get_current_session()
            if session_id == current:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot delete the current active session. Switch to another session first.",
                )

            deleted = memory.delete_session(session_id)

            if deleted == 0:
                raise HTTPException(status_code=404, detail="Session not found or already empty")

            logger.info(f"Deleted session {session_id}: {deleted} messages removed")

            return DeleteResponse(
                deleted_count=deleted,
                message=f"Deleted session {session_id} with {deleted} messages",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Message Endpoints
    # =========================================================================

    @app.get(
        "/sessions/{session_id}/messages",
        response_model=MessageListResponse,
        tags=["Messages"],
    )
    async def get_session_messages(
        session_id: str,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> MessageListResponse:
        """Get messages for a session with pagination."""
        try:
            messages, total = memory.get_session_messages(
                session_id=session_id,
                limit=limit,
                offset=offset,
            )

            message_list = []
            for m in messages:
                message_list.append(
                    MessageInfo(
                        id=m.id,
                        session_id=m.session_id,
                        role=m.role,
                        content=m.content,
                        timestamp=m.timestamp.isoformat() if m.timestamp else None,
                        metadata=m.metadata or {},
                    )
                )

            return MessageListResponse(
                messages=message_list,
                total=total,
                limit=limit,
                offset=offset,
            )
        except Exception as e:
            logger.error(f"Error getting messages for session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/sessions/{session_id}/messages",
        response_model=DeleteResponse,
        tags=["Messages"],
    )
    async def clear_session_messages(session_id: str) -> DeleteResponse:
        """Clear all messages in a session (keeps the session valid)."""
        try:
            deleted = memory.clear_session(session_id)

            logger.info(f"Cleared session {session_id}: {deleted} messages removed")

            return DeleteResponse(
                deleted_count=deleted,
                message=f"Cleared {deleted} messages from session {session_id}",
            )
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app
