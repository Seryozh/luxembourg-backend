from pydantic import BaseModel


# --- Plugin → Backend ---

class ChatRequest(BaseModel):
    """What the plugin sends when the user hits 'send'."""
    session_id: str
    user_message: str
    project_map: str  # the game tree as text (names, types, hierarchy)
    openrouter_key: str


class PluginPollResponse(BaseModel):
    """What the plugin sends back when fulfilling a pending request."""
    session_id: str
    request_id: str
    data: dict  # the metadata or script content the agent asked for


# --- Backend → Plugin ---

class PendingRequest(BaseModel):
    """A request the agent needs the plugin to fulfill."""
    request_id: str
    request_type: str  # "get_metadata" or "get_full_script"
    target: str  # script name or path


class Action(BaseModel):
    """A single action the agent wants to perform in Studio."""
    type: str        # "set_property", "create_instance", "delete_instance",
                     # "modify_script", "create_script", "delete_script",
                     # "move_instance", "clone_instance"
    target: str      # instance path, e.g. "game.Lighting" or "game.Workspace.Enemy"
    properties: dict = {}  # property name → value pairs
    # For scripts:
    source: str = ""       # script source code (for create_script / modify_script)
    # For create_instance:
    class_name: str = ""   # e.g. "Part", "Script", "RemoteEvent"
    name: str = ""         # name of the new instance
    # For move_instance:
    new_parent: str = ""   # new parent path


class ChatResponse(BaseModel):
    """What the backend returns to the plugin after the agent finishes."""
    session_id: str
    message: str  # the text response for the user
    actions: list[Action] = []
    metadata_updates: dict[str, str] = {}  # script name → updated description
