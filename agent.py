import logging
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

from session import Session
from tools import search_project, list_children, get_metadata, get_full_script
from config import settings

log = logging.getLogger("luxembourg")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


class AgentState(TypedDict):
    session: Session
    user_message: str
    messages: Annotated[list, add]
    final_response: dict


def make_agent(openrouter_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model="google/gemini-3-flash-preview",
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )


def format_pending_creations(pending: dict) -> str:
    """Format pending creations for system prompt"""
    if not pending:
        return "None"

    lines = []
    for name, info in pending.items():
        lines.append(f"  - {info['target']} ({info['type']}{', name: ' + name if name else ''})")
    return "\n".join(lines)


UNIFIED_SYSTEM = """Expert Roblox Studio AI with full game control.

CRITICAL: You receive a PARTIAL VIEW of the project (top-level containers only). The project map does NOT show scripts or detailed children. You MUST use tools to explore before making changes.

STEP-BY-STEP EXECUTION (CRITICAL - ALWAYS DO THIS):
For ANY task with multiple components, you MUST work incrementally:

1. ANALYZE & PLAN: Break task into logical steps (UI → Scripts → Wiring)
2. EXECUTE ONE STEP: Use tools, create actions for ONLY the current step
3. VERIFY: Describe what you did and what's next
4. CONTINUE: When user responds or plan continues, do next step

Multi-step tasks include:
- Creating new systems (e.g., "add sprint and stamina" = UI + handler + events)
- Modifying multiple scripts (each script is a step)
- Complex features (anything with "and" or multiple components)

Single-step tasks:
- Simple property changes ("make the part red")
- Single script modifications

REQUIRED FORMAT for multi-step tasks:
{
  "actions": [only actions for current step],
  "description": "Step 1/3: Created stamina UI (ScreenGui with Frame and TextLabel). Next: Creating stamina handler script.",
  "plan": {
    "steps": ["Create stamina UI", "Create stamina handler script", "Wire up events"],
    "current_step": 1,
    "total_steps": 3
  }
}

When you see "Active Multi-Step Plan" in the context, continue with the next step immediately.

REQUIRED WORKFLOW:
1. When user mentions existing functionality (e.g., "stamina system", "player script"), use search_project() to find it
2. Before modifying scripts, ALWAYS use get_full_script() to read current code
3. When creating related features, search for existing systems first to avoid duplicates

Tools (USE THEM PROACTIVELY):
- search_project(query): Search for scripts/objects by name or keyword. Use this FIRST when user mentions existing features.
  Example: search_project("stamina") finds "StaminaBar", "StaminaHandler", etc.
- list_children(path): List contents of a game path (e.g., "game.Workspace" or "game.ServerScriptService")
- get_metadata(script_name): Preview script (type, location, lines, first few lines) - use for quick inspection
- get_full_script(script_name): Read full source code (returns Path, Hash, and Source)
  REQUIRED before any modify_script action to get the hash and current code

EXAMPLES:
User: "add dash ability that uses stamina"
❌ BAD: Create new stamina bar without checking
✅ GOOD: search_project("stamina") → find existing StaminaBar → use it

User: "modify the player movement script"
❌ BAD: Guess the script location
✅ GOOD: search_project("movement") or search_project("player") → get_full_script() → modify

Only skip tools for simple property changes that don't involve scripts.

HASH VERIFICATION (CRITICAL):
- get_full_script() returns a "Hash: ..." line
- You MUST copy this hash value to the original_hash field in modify_script actions
- This prevents data loss from concurrent edits
- If you modify a script without the hash, it will fail

Output:
- Conversational: Plain text
- Modifications: JSON with actions array

JSON Format:
{"actions": [<action_objects>], "description": "summary", "metadata_updates": {}}

Action Types:
- set_property: {"type": "set_property", "target": "game.Lighting", "properties": {"ClockTime": 0}}
- create_instance: {"type": "create_instance", "target": "game.Workspace", "class_name": "Part", "name": "Floor", "properties": {"Size": [100,1,100], "Anchored": true}}
- delete_instance: {"type": "delete_instance", "target": "game.Workspace.OldPart"}
- move_instance: {"type": "move_instance", "target": "game.Workspace.Model", "new_parent": "game.ServerStorage"}
- clone_instance: {"type": "clone_instance", "target": "game.ServerStorage.Template", "new_parent": "game.Workspace", "name": "Clone1"}
- create_script: {"type": "create_script", "target": "game.ServerScriptService", "name": "Manager", "class_name": "Script", "source": "code"}
- modify_script: {"type": "modify_script", "target": "game.ServerScriptService.Manager", "source": "full_code", "original_hash": "hash_from_get_full_script"}
- delete_script: {"type": "delete_script", "target": "game.ServerScriptService.Manager"}

Rules:
- Prefer direct property changes over scripts
- Vector3/Color3/UDim2: use arrays [x,y,z] / [r,g,b 0-255] / [xS,xO,yS,yO]
- BrickColor: "Bright red", Enums: "Enum.Material.Grass"
- Full paths from "game.", complete source for script modifications
- Set all relevant properties on create, break complex tasks into multiple actions
- CRITICAL: When modifying scripts, you MUST first call get_full_script() to read the current source, then include the hash in the modify_script action to prevent data loss

REMEMBER: The project map is incomplete. When in doubt, USE TOOLS to explore. Don't guess - search first!"""


async def unified_agent_node(state: AgentState) -> dict:
    """Single agent that handles all requests - conversational and modifications"""
    log.info("=" * 60)
    log.info("AGENT START")
    log.info(f"User message: {state['user_message']}")
    log.info(f"Project map:\n{state['session'].project_map[:500]}")
    log.info(f"Conversation history: {len(state['session'].conversation_history)} messages")

    session = state["session"]
    llm = make_agent(session.openrouter_key)

    # Bind tools with session context
    async def _search_project(query: str) -> str:
        return await search_project(session, query)

    async def _list_children(path: str) -> str:
        return await list_children(session, path)

    async def _get_metadata(script_name: str) -> str:
        return await get_metadata(session, script_name)

    async def _get_full_script(script_name: str) -> str:
        return await get_full_script(session, script_name)

    llm_with_tools = llm.bind_tools([_search_project, _list_children, _get_metadata, _get_full_script])

    # Build message history
    pending_context = ""
    if session.pending_creations:
        pending_context = f"\n\n# Recently Created Objects (not yet in project map)\n{format_pending_creations(session.pending_creations)}"

    # Check for active plan and auto-continue
    plan_context = ""
    auto_continue_plan = False
    if session.active_plan and session.active_plan.get("steps"):
        current = session.active_plan.get("current_step", 0)
        total = session.active_plan.get("total_steps", 0)
        steps = session.active_plan.get("steps", [])
        completed_steps = steps[:current] if current > 0 else []
        next_step = steps[current] if current < len(steps) else None

        # Auto-continue if user message is simple continuation trigger
        user_msg_lower = state['user_message'].lower().strip()
        if user_msg_lower in ["continue", "next", "next step", "go on", "proceed", "ok", "yes"]:
            auto_continue_plan = True

        if next_step and auto_continue_plan:
            plan_context = f"\n\n# CONTINUING MULTI-STEP PLAN\n"
            plan_context += f"Progress: {current}/{total} steps completed\n\n"
            if completed_steps:
                plan_context += "Completed:\n" + "\n".join(f"  ✓ {s}" for s in completed_steps) + "\n\n"
            plan_context += f"NOW EXECUTING: {next_step}\n\n"
            if current + 1 < len(steps):
                plan_context += "Upcoming: " + ", ".join(steps[current + 1:]) + "\n\n"

            plan_context += f"USER REQUESTED CONTINUATION. Execute step {current + 1} now:\n"
            plan_context += f"1. Use tools if needed to explore/read code\n"
            plan_context += f"2. Create actions for '{next_step}' ONLY\n"
            plan_context += f"3. Return with updated plan showing current_step={current + 1}"
        elif next_step:
            # Show plan status but don't auto-continue
            plan_context = f"\n\n# Active Plan: Step {current}/{total} completed\n"
            plan_context += f"Next step: {next_step}\n"
            plan_context += "(User can say 'continue' to proceed)"

    messages = [
        SystemMessage(content=UNIFIED_SYSTEM),
        # Project map as separate system message for better caching
        SystemMessage(content=f"# Current Project Structure\n{session.project_map}{pending_context}{plan_context}")
    ]

    # Include recent conversation history with sliding window (last 20 messages)
    MAX_HISTORY_MESSAGES = 20
    recent_history = session.conversation_history[:-1] if session.conversation_history else []
    if len(recent_history) > MAX_HISTORY_MESSAGES:
        recent_history = recent_history[-MAX_HISTORY_MESSAGES:]

    for msg in recent_history:
        messages.append(
            HumanMessage(content=msg["content"])
            if msg["role"] == "user"
            else SystemMessage(content=msg["content"])
        )

    # Add current request WITHOUT project map (it's already in system message)
    messages.append(HumanMessage(content=state['user_message']))

    # Agentic loop
    loop_count = 0
    max_loops = 8  # Reduced from 15 - most tasks complete in 3-5 loops
    json_retry_count = 0
    max_json_retries = settings.max_json_retries

    while loop_count < max_loops:
        loop_count += 1
        log.info(f"AGENT loop {loop_count} — calling LLM...")
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        log.info(f"AGENT response: {response.content[:300]}...")

        # Handle tool calls - PARALLEL EXECUTION for performance
        if response.tool_calls:
            import asyncio

            log.info(f"AGENT executing {len(response.tool_calls)} tool call(s) in parallel")

            # Prepare all tool call tasks
            async def execute_tool(tool_call):
                func_name = tool_call["name"]
                args = tool_call["args"]
                log.info(f"AGENT tool call: {func_name}({args})")

                try:
                    if func_name == "_search_project":
                        result = await _search_project(**args)
                    elif func_name == "_list_children":
                        result = await _list_children(**args)
                    elif func_name == "_get_metadata":
                        result = await _get_metadata(**args)
                    elif func_name == "_get_full_script":
                        # Handle both parameter names for backward compatibility
                        script_name = args.get("script_name") or args.get("name") or ""
                        if not script_name:
                            result = "Error: script_name parameter is required"
                        else:
                            result = await _get_full_script(script_name)
                    else:
                        result = f"Unknown tool: {func_name}"

                    log.info(f"AGENT tool result: {result[:200]}...")
                    return (tool_call["id"], result)
                except Exception as e:
                    log.error(f"AGENT tool error: {str(e)}")
                    return (tool_call["id"], f"Error: {str(e)}")

            # Execute all tool calls in parallel
            results = await asyncio.gather(*[execute_tool(tc) for tc in response.tool_calls])

            # Append results in order
            for tool_call_id, result in results:
                messages.append(ToolMessage(content=result, tool_call_id=tool_call_id))

            # Prune old tool calls to prevent context bloat, but preserve conversation history
            if len(messages) > 40:
                # Keep: system messages (first 2) + conversation history + recent tool calls
                system_msgs = messages[:2]
                conversation_msgs = messages[2:2+len(recent_history)]
                current_turn = messages[2+len(recent_history):]

                # Keep last 20 messages of current turn (includes tool calls)
                if len(current_turn) > 20:
                    current_turn = current_turn[-20:]

                messages = system_msgs + conversation_msgs + current_turn

            continue

        # Process response content
        content = response.content

        # Try to parse as JSON (for action requests) with aggressive extraction
        result = None
        try:
            # Strategy 1: Extract from markdown code blocks (case-insensitive)
            if "```json" in content.lower():
                import re
                match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
                if match:
                    result = json.loads(match.group(1).strip())
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                result = json.loads(content.strip())
            else:
                # Strategy 2: Try parsing as-is
                result = json.loads(content.strip())

            # Valid JSON response - return as actions
            log.info(f"AGENT result: {json.dumps(result, indent=2)[:500]}")

            # STREAMING OPTIMIZATION: Push actions to session queue immediately
            if isinstance(result, dict) and "actions" in result:
                log.info(f"AGENT pushing {len(result['actions'])} actions to session queue")
                session.action_queue.extend(result["actions"])

                # Track created objects in working memory
                for action in result["actions"]:
                    if action.get("type") in ["create_instance", "create_script"]:
                        target = action.get("target", "")
                        name = action.get("name", "")
                        class_name = action.get("class_name", "")
                        full_path = f"{target}.{name}" if name else target

                        session.pending_creations[name or full_path] = {
                            "type": class_name or action.get("type"),
                            "target": full_path,
                            "action_type": action.get("type")
                        }
                        log.info(f"AGENT tracked creation: {name or full_path} ({class_name})")

                # Handle multi-step plan tracking
                if "plan" in result:
                    plan = result["plan"]
                    current_step = plan.get("current_step", 1)
                    total_steps = plan.get("total_steps", 1)

                    # If this is a new plan or different plan, initialize it
                    if not session.active_plan or session.active_plan.get("steps") != plan.get("steps"):
                        session.active_plan = {
                            "steps": plan.get("steps", []),
                            "current_step": current_step,
                            "total_steps": total_steps,
                            "description": result.get("description", "")
                        }
                        log.info(f"AGENT new plan initialized: step {current_step}/{total_steps}")
                    else:
                        # Continuing existing plan - increment the step
                        session.active_plan["current_step"] = current_step + 1
                        log.info(f"AGENT plan advanced: step {current_step + 1}/{total_steps}")

                    # Clear plan if completed
                    if current_step >= total_steps:
                        log.info("AGENT plan complete, clearing")
                        session.active_plan = {}
                elif not result.get("actions"):
                    # No actions and no plan means conversational response
                    # Don't clear plan unless user explicitly started a new task
                    pass

            log.info("AGENT DONE (actions)")
            return {
                "messages": messages,
                "final_response": result,
            }

        except (json.JSONDecodeError, AttributeError, TypeError):
            # Not JSON - check if it's intentionally conversational
            conversational_indicators = ["hello", "hi", "thanks", "thank you", "welcome", "sure", "okay"]
            action_indicators = ["action", "modify", "create", "set", "change", "update", "delete"]

            is_likely_conversational = (
                any(word in content.lower() for word in conversational_indicators)
                or (loop_count == 1 and not any(word in content.lower() for word in action_indicators))
            )

            if is_likely_conversational and not any(word in content.lower() for word in ["json", "```"]):
                # Likely a conversational response
                log.info(f"AGENT → conversational reply: {content[:200]}")
                log.info("AGENT DONE (chat)")
                return {
                    "messages": messages,
                    "final_response": {"description": content},
                }

            # JSON was expected but parsing failed - try one more aggressive extraction
            if result is None and json_retry_count == 0:
                import re
                # Try to find JSON object in the response
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        if isinstance(result, dict):
                            log.info("AGENT → recovered JSON with regex")
                            log.info("AGENT DONE (actions)")
                            return {
                                "messages": messages,
                                "final_response": result,
                            }
                    except:
                        pass

            # Still couldn't parse - retry with LLM
            json_retry_count += 1
            if json_retry_count >= max_json_retries:
                log.error(f"AGENT — JSON parse failed after {max_json_retries} retries")
                return {
                    "messages": messages,
                    "final_response": {
                        "actions": [],
                        "description": "Sorry, I couldn't generate valid actions. Please try rephrasing your request.",
                    },
                }

            log.warning(f"AGENT — JSON parse failed (attempt {json_retry_count}/{max_json_retries})")
            messages.append(HumanMessage(
                content="Your response was not valid JSON. Please respond with ONLY the JSON object in the format specified, or with plain text if this is conversational."
            ))
            continue

    # Max loops reached
    log.error("AGENT — max loops reached")
    return {
        "messages": messages,
        "final_response": {
            "description": "I couldn't complete that request. Please try breaking it into smaller steps.",
        },
    }


def build_graph():
    """Simplified single-node graph"""
    graph = StateGraph(AgentState)
    graph.add_node("agent", unified_agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()


async def run_agent(session: Session, user_message: str) -> dict:
    """Run the unified agent"""
    graph = build_graph()
    result = await graph.ainvoke({
        "session": session,
        "user_message": user_message,
        "messages": [],
        "final_response": {},
    })
    return result["final_response"]
