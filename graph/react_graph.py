# graph/react_graph.py  — AirGuard v3
#
# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH REACT GRAPH WITH CONTEXT ENGINEERING + SHORT-TERM MEMORY
# ══════════════════════════════════════════════════════════════════════════════
#
# Changes from v2 graph (plain ReAct):
#
#   v2:  START → react_agent_node ↔ tool_node → END
#        • Static system prompt compiled at import time
#        • ToolNode appended raw JSON ToolMessages
#        • No memory across runs
#
#   v3:  START → inject_context_node → react_agent_node ↔ compressed_tool_node → END
#        • inject_context_node:   rebuilds system prompt from live SESSION MEMORY
#        • react_agent_node:      trims message history before each LLM call
#        • compressed_tool_node:  compresses tool results ~80% before storage
#        • After END:             _post_run_memory_update() persists state
#
# ══════════════════════════════════════════════════════════════════════════════
# GRAPH TOPOLOGY
# ══════════════════════════════════════════════════════════════════════════════
#
#   START
#     │
#     ▼
#   inject_context_node        ← build dynamic system prompt from SESSION MEMORY
#     │
#     ▼
#   react_agent_node           ← trim history, call LLM with bound tools
#     │
#     └─► should_continue()   ← conditional edge
#               │ "tools"
#               ▼
#         compressed_tool_node ← execute tools, compress results
#               │
#               └──► react_agent_node  (loop)
#               │ "end"
#               ▼
#             END

import operator
from typing import TypedDict, List, Optional, Any, Annotated

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from agents.react_agent   import ALL_TOOLS, MAX_ITERATIONS, _post_run_memory_update
from memory.short_term_memory import SESSION_MEMORY
from context.context_engineer import CONTEXT_ENGINEER


# ─────────────────────────────────────────────────────────────────────────────
# State schema
# ─────────────────────────────────────────────────────────────────────────────

class ReactAirQualityState(TypedDict, total=False):
    # Core ReAct state
    messages:         Annotated[List[Any], operator.add]
    scenario:         Optional[str]
    iteration_count:  int

    # Output fields populated from tool log after loop ends
    target_region:               Optional[str]
    analysis_timestamp:          Optional[str]
    ground_readings:             List[Any]
    satellite_observations:      List[Any]
    meteorological_summary:      Optional[Any]
    pollution_sources:           List[Any]
    health_impact:               Optional[Any]
    risk_scores:                 List[Any]
    mitigation_recommendations:  List[Any]
    public_alerts:               List[Any]
    regulatory_actions:          List[Any]
    situation_report:            Optional[str]
    current_agent:               Optional[str]
    emergency_triggered:         bool
    errors:                      List[Any]


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — inject_context_node
# Runs once at the START of every graph invocation.
# Rebuilds the system prompt with the current SESSION MEMORY state.
# ─────────────────────────────────────────────────────────────────────────────

def inject_context_node(state: ReactAirQualityState) -> dict:
    """
    Build the per-run dynamic system prompt and inject it as SystemMessage[0].

    Without this node, the system prompt would be a static string compiled
    at import time — it would never see AQI trends or alert history from
    previous runs.
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
    except ImportError:
        return {}

    scenario = state.get("scenario", "standard")
    ctx      = CONTEXT_ENGINEER.build_context_for_run(scenario, SESSION_MEMORY)

    print(f"  [Context] System prompt rebuilt — ~{ctx['token_estimate']} tokens")
    print(f"  [Memory]  {SESSION_MEMORY.context_summary().split(chr(10))[0]}")

    # Replace any existing system message; preserve other messages
    existing   = state.get("messages", [])
    non_system = [
        m for m in existing
        if "SystemMessage" not in type(m).__name__
    ]

    return {
        "messages":        [
            SystemMessage(content=ctx["system_prompt"]),
            HumanMessage(content=ctx["initial_human_message"]),
        ] + non_system,
        "iteration_count": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — react_agent_node
# Core LLM call.  Trims message history before invoking.
# ─────────────────────────────────────────────────────────────────────────────

def react_agent_node(state: ReactAirQualityState) -> dict:
    """
    Call the LLM with bound tools.
    CONTEXT ENGINEERING: message history is trimmed before each call.
    """
    try:
        from langchain_openai import ChatOpenAI
        from config.settings  import air_config
    except ImportError:
        return {"errors": ["langchain_openai not installed"]}

    llm            = ChatOpenAI(
        model       = air_config.model_name,
        temperature = air_config.temperature,
        api_key     = air_config.openai_api_key,
        max_tokens  = 2000,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    messages       = state.get("messages", [])
    iteration      = state.get("iteration_count", 0) + 1

    # CONTEXT ENGINEERING: trim stale messages before LLM call
    trimmed = CONTEXT_ENGINEER.trim_message_history(messages)

    print(f"  [LangGraph] Iteration {iteration}: "
          f"{len(trimmed)} messages (trimmed from {len(messages)})")

    response = llm_with_tools.invoke(trimmed)
    n_calls  = len(getattr(response, "tool_calls", []) or [])
    print(f"    → {n_calls} tool call(s)" if n_calls else "    → FINAL RESPONSE")

    return {
        "messages":        [response],
        "iteration_count": iteration,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — compressed_tool_node
# Executes tool calls and stores COMPRESSED results in the message history.
# ─────────────────────────────────────────────────────────────────────────────

def compressed_tool_node(state: ReactAirQualityState) -> dict:
    """
    Execute each tool call from the last AIMessage and return compressed
    ToolMessages.  CONTEXT ENGINEERING: raw results are compressed ~80%
    before they enter the message history.
    """
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        # Fallback: use raw LangGraph ToolNode without compression
        from langgraph.prebuilt import ToolNode
        return ToolNode(ALL_TOOLS).invoke(state)

    tool_map = {
        getattr(t, "name", getattr(t, "__name__", str(t))): t
        for t in ALL_TOOLS
    }

    messages   = state.get("messages", [])
    last_msg   = messages[-1] if messages else None
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    new_messages = []
    for tc in tool_calls:
        name      = tc["name"]
        args      = tc.get("args", {})
        t         = tool_map.get(name)
        result    = (
            {"error": f"Unknown tool: {name}"}
            if not t
            else _safe_invoke_graph(t, args)
        )

        # CONTEXT ENGINEERING: compress before adding to history
        compressed = CONTEXT_ENGINEER.compress_tool_result(name, result)
        print(f"  [Tool] {name} → {compressed[:70]}...")

        new_messages.append(ToolMessage(
            content      = compressed,
            tool_call_id = tc["id"],
        ))

    return {"messages": new_messages}


def _safe_invoke_graph(tool, args: dict):
    try:
        return tool.invoke(args) if args else tool.invoke({})
    except Exception as ex:
        return {"error": str(ex)}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge
# ─────────────────────────────────────────────────────────────────────────────

def should_continue(state: ReactAirQualityState) -> str:
    messages   = state.get("messages", [])
    iteration  = state.get("iteration_count", 0)
    last_msg   = messages[-1] if messages else None
    tool_calls = getattr(last_msg, "tool_calls", None) or []

    if iteration >= MAX_ITERATIONS:
        print(f"  [LangGraph] Max iterations ({MAX_ITERATIONS}) reached → END")
        return "end"
    return "tools" if tool_calls else "end"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_react_graph():
    """
    Compile the v3 LangGraph ReAct graph.
    Returns None if langgraph is not installed.
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    g = StateGraph(ReactAirQualityState)

    g.add_node("inject_context",    inject_context_node)
    g.add_node("react_agent_node",  react_agent_node)
    g.add_node("tool_node",         compressed_tool_node)

    g.add_edge(START, "inject_context")
    g.add_edge("inject_context", "react_agent_node")

    g.add_conditional_edges(
        "react_agent_node",
        should_continue,
        {"tools": "tool_node", "end": END},
    )
    g.add_edge("tool_node", "react_agent_node")

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Fallback — used when langgraph is not installed
# ─────────────────────────────────────────────────────────────────────────────

def run_react_pipeline(scenario: str = "standard") -> dict:
    """Direct fallback: call react_agent.run_react_agent() without LangGraph."""
    from agents.react_agent import run_react_agent
    return run_react_agent(scenario)
