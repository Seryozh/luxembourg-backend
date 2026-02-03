import asyncio
import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    openrouter_key: str
    project_map: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    cached_metadata: dict[str, str] = field(default_factory=dict)  # script name → description
    cached_scripts: dict[str, str] = field(default_factory=dict)  # script name → source code

    # Pending requests the agent needs fulfilled by the plugin
    pending_requests: dict[str, dict] = field(default_factory=dict)  # request_id → request info
    # Fulfilled responses from the plugin
    fulfilled_responses: dict[str, asyncio.Event] = field(default_factory=dict)
    fulfilled_data: dict[str, dict] = field(default_factory=dict)


# In-memory store
_sessions: dict[str, Session] = {}


def get_or_create_session(session_id: str, openrouter_key: str) -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = Session(
            session_id=session_id,
            openrouter_key=openrouter_key,
        )
    return _sessions[session_id]


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)


def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


async def request_from_plugin(session: Session, request_type: str, target: str) -> dict:
    """
    Called by agent tools. Creates a pending request, waits for the plugin
    to fulfill it via polling, then returns the data.
    """
    rid = str(uuid.uuid4())
    event = asyncio.Event()

    session.pending_requests[rid] = {"type": request_type, "target": target}
    session.fulfilled_responses[rid] = event

    try:
        await asyncio.wait_for(event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        session.pending_requests.pop(rid, None)
        session.fulfilled_responses.pop(rid, None)
        raise TimeoutError(f"Plugin did not respond for {target}")

    data = session.fulfilled_data.pop(rid)
    session.fulfilled_responses.pop(rid, None)
    return data
