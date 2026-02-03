from session import Session, request_from_plugin


async def get_metadata(session: Session, script_name: str) -> str:
    """
    Agent calls this when it sees a script in the project map and wants
    to know what it does (~100 word description) before reading the full code.

    Flow: agent calls this → backend creates pending request → plugin picks
    it up on next poll → plugin reads the script, summarizes it → sends back.
    """
    result = await request_from_plugin(session, "get_metadata", script_name)
    # Cache it so we don't ask again
    session.cached_metadata[script_name] = result.get("metadata", "")
    return session.cached_metadata[script_name]


async def get_full_script(session: Session, script_name: str) -> str:
    """
    Agent calls this when it needs the actual source code of a script.
    This is Level 3 exploration — only for scripts the agent truly needs.

    Flow: same as get_metadata, but plugin sends the full source code.
    """
    result = await request_from_plugin(session, "get_full_script", script_name)
    # Cache it
    session.cached_scripts[script_name] = result.get("source", "")
    return session.cached_scripts[script_name]
