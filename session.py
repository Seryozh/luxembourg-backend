import asyncio
import uuid
import time
from dataclasses import dataclass, field

from config import settings


@dataclass
class Session:
    session_id: str
    openrouter_key: str
    project_map: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    cached_metadata: dict[str, str] = field(default_factory=dict)
    cached_scripts: dict[str, str] = field(default_factory=dict)
    pending_requests: dict[str, dict] = field(default_factory=dict)
    fulfilled_responses: dict[str, asyncio.Event] = field(default_factory=dict)
    fulfilled_data: dict[str, dict] = field(default_factory=dict)
    action_queue: list[dict] = field(default_factory=list)
    pending_creations: dict[str, dict] = field(default_factory=dict)  # Track recently created objects
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


_sessions: dict[str, Session] = {}


def get_or_create_session(session_id: str, openrouter_key: str) -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = Session(
            session_id=session_id,
            openrouter_key=openrouter_key,
        )
    session = _sessions[session_id]
    session.last_accessed = time.time()
    return session


def get_session(session_id: str) -> Session | None:
    session = _sessions.get(session_id)
    if session:
        session.last_accessed = time.time()
    return session


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def cleanup_expired_sessions() -> int:
    """Remove sessions that haven't been accessed in settings.session_ttl seconds. Returns count of deleted sessions."""
    now = time.time()
    expired = [
        session_id
        for session_id, session in _sessions.items()
        if now - session.last_accessed > settings.session_ttl
    ]
    for session_id in expired:
        _sessions.pop(session_id, None)
    return len(expired)


def get_session_count() -> int:
    """Get count of active sessions (useful for monitoring)."""
    return len(_sessions)


async def request_from_plugin(session: Session, request_type: str, target: str) -> dict:
    rid = str(uuid.uuid4())
    event = asyncio.Event()

    session.pending_requests[rid] = {"type": request_type, "target": target}
    session.fulfilled_responses[rid] = event

    try:
        await asyncio.wait_for(event.wait(), timeout=float(settings.poll_timeout))
    except asyncio.TimeoutError:
        session.pending_requests.pop(rid, None)
        session.fulfilled_responses.pop(rid, None)
        raise TimeoutError(f"Plugin did not respond for {target}")

    data = session.fulfilled_data.pop(rid)
    session.fulfilled_responses.pop(rid, None)
    return data
