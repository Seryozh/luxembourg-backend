from fastapi import FastAPI, HTTPException

from models import (
    Action,
    ChatRequest,
    ChatResponse,
    PendingRequest,
    PluginPollResponse,
)
from session import get_or_create_session, get_session
from agent import run_agent

app = FastAPI(title="Luxembourg Backend")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main endpoint. Plugin sends the user message + project map here.
    Backend creates/reuses a session, runs the agent, returns the result.
    """
    session = get_or_create_session(request.session_id, request.openrouter_key)
    session.project_map = request.project_map  # always update (may have changed)

    # Add user message to conversation history
    session.conversation_history.append({
        "role": "user",
        "content": request.user_message,
    })

    # Run the two-model agent
    result = await run_agent(session, request.user_message)

    # Convert agent output into response format
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
    Plugin polls this endpoint to check if the agent needs anything.
    Returns pending requests (get_metadata, get_full_script) or empty list.
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
    Plugin sends back data the agent requested (metadata or script content).
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    rid = response.request_id
    if rid not in session.pending_requests:
        raise HTTPException(status_code=404, detail="Request not found")

    # Store the response and signal the waiting agent
    session.fulfilled_data[rid] = response.data
    session.pending_requests.pop(rid)

    if rid in session.fulfilled_responses:
        session.fulfilled_responses[rid].set()

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    from config import settings
    uvicorn.run(app, host=settings.host, port=settings.port)
