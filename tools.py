from session import Session, request_from_plugin


async def get_metadata(session: Session, script_name: str) -> str:
    result = await request_from_plugin(session, "get_metadata", script_name)
    session.cached_metadata[script_name] = result.get("metadata", "")
    return session.cached_metadata[script_name]


async def get_full_script(session: Session, script_name: str) -> str:
    result = await request_from_plugin(session, "get_full_script", script_name)
    session.cached_scripts[script_name] = result.get("source", "")
    return session.cached_scripts[script_name]
