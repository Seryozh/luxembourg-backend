import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from models import (
    Action,
    ChatRequest,
    ChatResponse,
    PendingRequest,
    PluginPollResponse,
)
from session import get_or_create_session, get_session, cleanup_expired_sessions, get_session_count
from agent import run_agent

log = logging.getLogger("luxembourg")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


async def session_cleanup_task():
    """Background task to clean up expired sessions periodically."""
    from config import settings
    while True:
        await asyncio.sleep(settings.cleanup_interval)
        deleted = cleanup_expired_sessions()
        if deleted > 0:
            log.info(f"Cleaned up {deleted} expired sessions. Active sessions: {get_session_count()}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: start background tasks on startup, cleanup on shutdown."""
    cleanup_task = asyncio.create_task(session_cleanup_task())
    log.info("Started session cleanup background task")
    yield
    cleanup_task.cancel()
    log.info("Shutdown complete")


app = FastAPI(
    title="Luxembourg Backend",
    description="AI-powered Roblox Studio assistant backend",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from Roblox Studio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with user-friendly messages."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Invalid request data"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unexpected errors."""
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "message": str(exc)},
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Receives user message and project map, runs the AI agent,
    and returns actions to execute in Roblox Studio.

    The agent uses a two-model architecture:
    - Orchestrator: Analyzes request, explores project, creates worker tasks
    - Worker: Generates specific actions based on orchestrator's task

    Returns JSON with assistant message and actions array.
    """
    session = get_or_create_session(request.session_id, request.openrouter_key)
    session.project_map = request.project_map

    session.conversation_history.append({
        "role": "user",
        "content": request.user_message,
    })

    result = await run_agent(session, request.user_message)

    actions = []
    for action_data in result.get("actions", []):
        actions.append(Action(**action_data))

    assistant_message = result.get("description", "Done.")

    session.conversation_history.append({
        "role": "assistant",
        "content": assistant_message,
    })

    return ChatResponse(
        session_id=session.session_id,
        message=assistant_message,
        actions=actions,
        metadata_updates=result.get("metadata_updates", {}),
    )


@app.get("/poll/{session_id}")
async def poll(session_id: str):
    """
    Polling endpoint for the plugin to check for pending requests from the agent.

    The agent uses this to request script metadata or full script contents from
    the plugin. Returns array of pending requests with type (get_metadata, get_full_script)
    and target (script name/path).
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    pending = [
        PendingRequest(
            request_id=rid,
            request_type=info["type"],
            target=info["target"],
        )
        for rid, info in session.pending_requests.items()
    ]
    return {"pending_requests": pending}


@app.post("/poll/{session_id}/respond")
async def poll_respond(session_id: str, response: PluginPollResponse):
    """
    Response endpoint for the plugin to send back requested data.

    When the agent requests script metadata or full source, the plugin polls,
    sees the request, fetches the data, and sends it back via this endpoint.
    This wakes up the asyncio.Event in the agent so it can continue processing.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    rid = response.request_id
    if rid not in session.pending_requests:
        raise HTTPException(status_code=404, detail="Request not found")

    session.fulfilled_data[rid] = response.data
    session.pending_requests.pop(rid)

    if rid in session.fulfilled_responses:
        session.fulfilled_responses[rid].set()

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    from config import settings
    uvicorn.run(app, host=settings.host, port=settings.port)
