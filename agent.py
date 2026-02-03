import logging
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

from session import Session
from tools import get_metadata, get_full_script
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
    orchestrator_messages: Annotated[list, add]
    worker_task: str
    worker_messages: Annotated[list, add]
    final_response: dict


def make_orchestrator(openrouter_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model="google/gemini-3-flash-preview",
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )


def make_worker(openrouter_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model="google/gemini-3-flash-preview",
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )


ORCHESTRATOR_SYSTEM = """You are an AI assistant that helps modify Roblox games.

You receive the project map (the game tree showing all objects, scripts, and hierarchy) and the user's request.

Your job:
1. Analyze the user's request and determine what kind of task it is:
   - Property changes (lighting, terrain, atmosphere) → NO scripts needed
   - Creating instances (parts, models, UI) → NO scripts needed
   - Game logic (movement, combat, AI, events) → scripts ARE needed
2. ONLY use get_metadata/get_full_script if the task genuinely involves reading or modifying existing scripts. Do NOT read scripts for tasks like "change the sky color" or "add a part to workspace".
3. Write a clear, detailed task for the Worker model.

IMPORTANT: Be efficient. If the user asks to change a property or create an instance, just describe the task — don't explore scripts that aren't relevant.

When you are done, respond with your task wrapped in <worker_task> tags:
<worker_task>
Your detailed task here...
</worker_task>

If the user's message is conversational (like "hi", "thanks", questions about the project) and does NOT require any changes, respond with a <chat_reply> tag instead:
<chat_reply>
Your conversational response here...
</chat_reply>"""

WORKER_SYSTEM = """You are an expert Roblox Studio AI with full control over the game.

You receive a task describing what to build or modify. Use get_full_script to read any scripts you need.

You have FULL control over Roblox Studio. You can:
- Set properties on any instance (lighting, terrain, parts, UI, etc.)
- Create any instance (Parts, Models, Scripts, UI elements, RemoteEvents, etc.)
- Delete instances
- Move/reparent instances
- Clone instances
- Create, modify, or delete scripts

You MUST respond with valid JSON containing an "actions" array and a "description":
{
  "actions": [
    {"type": "set_property", "target": "game.Lighting", "properties": {"ClockTime": 0, "Brightness": 1}},
    {"type": "create_instance", "target": "game.Workspace", "class_name": "Part", "name": "Floor", "properties": {"Size": [100, 1, 100], "Position": [0, 0, 0], "Anchored": true}},
    {"type": "delete_instance", "target": "game.Workspace.OldPart"},
    {"type": "move_instance", "target": "game.Workspace.MyModel", "new_parent": "game.ServerStorage"},
    {"type": "clone_instance", "target": "game.ServerStorage.Template", "new_parent": "game.Workspace", "name": "Clone1"},
    {"type": "create_script", "target": "game.ServerScriptService", "name": "GameManager", "class_name": "Script", "source": "print('hello')"},
    {"type": "modify_script", "target": "game.ServerScriptService.GameManager", "source": "full new source code"},
    {"type": "delete_script", "target": "game.ServerScriptService.GameManager"}
  ],
  "description": "What was changed and why",
  "metadata_updates": {}
}

Rules:
- Use DIRECT property changes when possible (like setting ClockTime) instead of creating scripts.
- Only create scripts when actual runtime game logic is needed.
- For Vector3 values, use arrays: [x, y, z]. For Color3, use arrays: [r, g, b] (0-255).
- For UDim2, use arrays: [xScale, xOffset, yScale, yOffset].
- For BrickColor, use the name string: "Bright red".
- For Enum values, use strings: "Enum.Material.Grass".
- Always use full instance paths starting with "game."
- Include COMPLETE source code for modified scripts.
- Keep descriptions concise.
- When creating instances, set ALL relevant properties (Size, Position, Color, Material, Anchored, etc.). Be thorough — don't leave default values if the user expects something specific.
- Each action should be a single logical step that the user can approve or deny individually. Break complex tasks into multiple actions rather than one giant action."""


async def orchestrator_node(state: AgentState) -> dict:
    log.info("=" * 60)
    log.info("ORCHESTRATOR START")
    log.info(f"User message: {state['user_message']}")
    log.info(f"Project map:\n{state['session'].project_map[:500]}")
    log.info(f"Conversation history: {len(state['session'].conversation_history)} messages")

    session = state["session"]
    llm = make_orchestrator(session.openrouter_key)

    async def _get_metadata(script_name: str) -> str:
        return await get_metadata(session, script_name)

    async def _get_full_script(script_name: str) -> str:
        return await get_full_script(session, script_name)

    llm_with_tools = llm.bind_tools([_get_metadata, _get_full_script])

    messages = [SystemMessage(content=ORCHESTRATOR_SYSTEM)]

    for msg in session.conversation_history[:-1]:
        messages.append(HumanMessage(content=msg["content"]) if msg["role"] == "user"
                       else SystemMessage(content=msg["content"]))

    messages.append(HumanMessage(
        content=f"Project Map:\n{session.project_map}\n\nUser Request: {state['user_message']}"
    ))

    loop_count = 0
    while True:
        loop_count += 1
        log.info(f"ORCHESTRATOR loop {loop_count} — calling LLM...")
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        log.info(f"ORCHESTRATOR response: {response.content[:300]}...")

        if response.tool_calls:
            for tool_call in response.tool_calls:
                func_name = tool_call["name"]
                args = tool_call["args"]
                log.info(f"ORCHESTRATOR tool call: {func_name}({args})")
                if func_name == "_get_metadata":
                    result = await _get_metadata(**args)
                elif func_name == "_get_full_script":
                    result = await _get_full_script(**args)
                else:
                    result = f"Unknown tool: {func_name}"
                log.info(f"ORCHESTRATOR tool result: {result[:200]}...")
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            continue

        content = response.content
        if "<chat_reply>" in content and "</chat_reply>" in content:
            reply = content.split("<chat_reply>")[1].split("</chat_reply>")[0].strip()
            log.info(f"ORCHESTRATOR → chat reply: {reply}")
            return {
                "orchestrator_messages": messages,
                "worker_task": "",
                "final_response": {"description": reply},
            }

        if "<worker_task>" in content and "</worker_task>" in content:
            task = content.split("<worker_task>")[1].split("</worker_task>")[0].strip()
            log.info(f"ORCHESTRATOR → worker task:\n{task}")
            return {
                "orchestrator_messages": messages,
                "worker_task": task,
            }

        log.info("ORCHESTRATOR — no task or tools, nudging...")
        messages.append(HumanMessage(
            content="Please either use a tool to explore further, or write your <worker_task> if you have enough information."
        ))


async def worker_node(state: AgentState) -> dict:
    log.info("=" * 60)
    log.info("WORKER START")
    log.info(f"Task:\n{state['worker_task'][:500]}")

    session = state["session"]
    llm = make_worker(session.openrouter_key)

    async def _get_full_script(script_name: str) -> str:
        return await get_full_script(session, script_name)

    llm_with_tools = llm.bind_tools([_get_full_script])

    messages = [
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=state["worker_task"]),
    ]

    loop_count = 0
    max_retries = settings.max_json_retries
    json_retry_count = 0

    while True:
        loop_count += 1
        log.info(f"WORKER loop {loop_count} — calling LLM...")
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                log.info(f"WORKER tool call: get_full_script({tool_call['args']})")
                # Standardize parameter name (accept both script_name and name for backward compatibility)
                args = tool_call["args"]
                script_name = args.get("script_name") or args.get("name") or ""
                if not script_name:
                    log.error(f"WORKER tool call missing script_name parameter: {args}")
                    result = "Error: script_name parameter is required"
                else:
                    result = await _get_full_script(script_name)
                    log.info(f"WORKER tool result: {result[:200]}...")
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            continue

        content = response.content
        log.info(f"WORKER raw response: {content[:500]}...")

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
        except json.JSONDecodeError:
            json_retry_count += 1
            if json_retry_count >= max_retries:
                log.error(f"WORKER — JSON parse failed after {max_retries} retries, returning error")
                return {
                    "worker_messages": messages,
                    "final_response": {
                        "actions": [],
                        "description": "Sorry, I couldn't generate valid actions. Please try rephrasing your request.",
                    },
                }
            log.warning(f"WORKER — JSON parse failed (attempt {json_retry_count}/{max_retries}), asking for retry")
            messages.append(HumanMessage(
                content="Your response was not valid JSON. Please respond with ONLY the JSON object in the format specified."
            ))
            continue

        log.info(f"WORKER result: {json.dumps(result, indent=2)[:500]}")
        log.info("WORKER DONE")
        return {
            "worker_messages": messages,
            "final_response": result,
        }


def should_run_worker(state: AgentState) -> str:
    if state.get("final_response"):
        return END
    return "worker"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("worker", worker_node)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", should_run_worker)
    graph.add_edge("worker", END)
    return graph.compile()


async def run_agent(session: Session, user_message: str) -> dict:
    graph = build_graph()
    result = await graph.ainvoke({
        "session": session,
        "user_message": user_message,
        "orchestrator_messages": [],
        "worker_task": "",
        "worker_messages": [],
        "final_response": {},
    })
    return result["final_response"]
