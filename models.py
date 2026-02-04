from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    session_id: str = Field(..., min_length=1, max_length=128, description="Unique session identifier")
    user_message: str = Field(..., min_length=1, max_length=10000, description="User's message")
    project_map: str = Field(..., max_length=100000, description="Roblox project structure map")
    openrouter_key: str = Field(..., min_length=1, description="OpenRouter API key")

    @field_validator("openrouter_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Ensure API key looks valid."""
        if not v or len(v) < 10:
            raise ValueError("OpenRouter API key appears invalid")
        return v


class PluginRequestResponse(BaseModel):
    """Response model from plugin to /poll/{session_id}/respond endpoint."""
    session_id: str = Field(..., min_length=1)
    request_id: str = Field(..., min_length=1)
    data: dict = Field(default_factory=dict)


class PendingRequest(BaseModel):
    """Pending request from agent to plugin."""
    request_id: str
    request_type: str = Field(..., pattern="^(get_metadata|get_full_script|list_children|search_project)$")
    target: str = Field(..., min_length=1)


class Action(BaseModel):
    """Action to execute in Roblox Studio. Represents a single modification."""
    type: str = Field(
        ...,
        pattern="^(set_property|create_instance|delete_instance|move_instance|clone_instance|create_script|modify_script|delete_script)$"
    )
    target: str = Field(..., min_length=1, description="Target instance path (e.g. 'game.Workspace.Part')")
    properties: dict = Field(default_factory=dict)
    source: str = ""
    class_name: str = ""
    name: str = ""
    new_parent: str = ""
    original_hash: str = ""


class ChatResponse(BaseModel):
    """Response from /chat endpoint with agent's message and actions."""
    session_id: str
    message: str = Field(..., min_length=1)
    actions: list[Action] = Field(default_factory=list)
    metadata_updates: dict[str, str] = Field(default_factory=dict)
    plan: dict = Field(default_factory=dict)  # Multi-step plan progress


class PluginPollResponse(BaseModel):
    """Response model for the /poll/{session_id} endpoint."""
    pending_requests: list[PendingRequest] = Field(default_factory=list)
    queued_actions: list[Action] = Field(default_factory=list)
