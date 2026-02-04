from session import Session, request_from_plugin


async def search_project(session: Session, query: str) -> str:
    """SEMANTIC SEARCH: Search for scripts by name or keyword."""
    # Check cache first to avoid duplicate plugin requests
    cache_key = f"search:{query}"
    if cache_key in session.cached_metadata:
        return session.cached_metadata[cache_key]

    result = await request_from_plugin(session, "search_project", query)
    search_results = result.get("results", "")
    session.cached_metadata[cache_key] = search_results
    return search_results


async def list_children(session: Session, path: str) -> str:
    """LAZY LOADING: List children of a specific game path."""
    # Check cache first to avoid duplicate plugin requests
    cache_key = f"children:{path}"
    if cache_key in session.cached_metadata:
        return session.cached_metadata[cache_key]

    result = await request_from_plugin(session, "list_children", path)
    children_list = result.get("children", "")
    session.cached_metadata[cache_key] = children_list
    return children_list


async def get_metadata(session: Session, script_name: str) -> str:
    # Check cache first to avoid duplicate plugin requests
    if script_name in session.cached_metadata:
        return session.cached_metadata[script_name]

    result = await request_from_plugin(session, "get_metadata", script_name)
    session.cached_metadata[script_name] = result.get("metadata", "")
    return session.cached_metadata[script_name]


async def get_full_script(session: Session, script_name: str) -> str:
    # Check cache first to avoid duplicate plugin requests
    if script_name in session.cached_scripts:
        return session.cached_scripts[script_name]

    result = await request_from_plugin(session, "get_full_script", script_name)
    source = result.get("source", "")
    hash_value = result.get("hash", "")
    path = result.get("path", script_name)

    # Format response with hash for verification
    formatted_response = f"Path: {path}\nHash: {hash_value}\n\nSource:\n{source}"

    session.cached_scripts[script_name] = formatted_response
    return formatted_response
