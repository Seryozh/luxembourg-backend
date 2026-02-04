import asyncio
import uuid
import time
import hashlib
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
    executed_action_hashes: set = field(default_factory=set)  # Track executed actions for idempotency
    pending_creations: dict[str, dict] = field(default_factory=dict)  # Track recently created objects
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def clear_caches(self):
        """Clear all caches when project state changes"""
        self.cached_metadata.clear()
        self.cached_scripts.clear()
        self.pending_creations.clear()

    def compute_action_hash(self, action: dict) -> str:
        """Compute a hash for an action to detect duplicates"""
        # Create a stable string representation including source for scripts
        action_str = f"{action.get('type')}:{action.get('target')}:{action.get('name', '')}:{action.get('class_name', '')}"
        # Include source hash for script actions to differentiate by content
        if action.get('type') in ['create_script', 'modify_script'] and action.get('source'):
            source_hash = hashlib.md5(action['source'].encode()).hexdigest()[:8]
            action_str += f":{source_hash}"
        return hashlib.md5(action_str.encode()).hexdigest()[:12]

    def queue_actions_deduplicated(self, actions: list[dict]) -> list[dict]:
        """Queue actions with deduplication. Returns list of newly queued actions."""
        newly_queued = []
        for action in actions:
            action_hash = self.compute_action_hash(action)
            if action_hash not in self.executed_action_hashes:
                self.executed_action_hashes.add(action_hash)
                self.action_queue.append(action)
                newly_queued.append(action)
        return newly_queued


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
