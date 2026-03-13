# graph/react_graph.py
#
# LangGraph StateGraph for the ReAct Air Pollution Monitoring Agent.
#
# ════════════════════════════════════════════════════════════════════
# TOPOLOGY COMPARISON
# ════════════════════════════════════════════════════════════════════
#
# LINEAR (original pollution_graph.py):
#
#   START
#     → supervisor_init        (runs once)
#     → ground_sensor_agent    (runs once)
#     → satellite_agent        (runs once)
#     → meteorological_agent   (runs once)
#     → source_identification  (runs once)
#     → health_impact_agent    (runs once)
#     → mitigation_agent       (runs once)
#     → alert_agent            (runs once)
#     → supervisor_synthesis   (runs once)
#   END
#
#   9 nodes. 8 hard-wired add_edge() calls. Zero conditional logic.
#   emergency_triggered is a data flag — it never changes the path.
#
# ════════════════════════════════════════════════════════════════════
#
# REACT (this file):
#
#   START
#     → react_agent_node       (runs N times in a loop)
#       |
#       +-- should_continue()  ← conditional edge
#             ↓ "continue"
#         react_agent_node     (loop back — LLM has more tool calls)
#             ↓ "end"
#           END                (LLM produced no tool calls — reasoning complete)
#
#   1 node. 1 conditional edge. The LLM controls when the loop stops.
#
# ════════════════════════════════════════════════════════════════════
#
# WHY THIS IS ReAct:
#
#   The node react_agent_node wraps an LLM call with bind_tools(ALL_TOOLS).
#   The LLM response is inspected by should_continue():
#     - If the AIMessage has tool_calls → the node routes to "continue"
#       → tool executor runs each tool, appends ToolMessages to state
#       → react_agent_node is called again with the enriched message history
#     - If the AIMessage has NO tool_calls → routes to "end"
#       → the message content is the final situation report
#
#   This is exactly the pattern described in the LangGraph ReAct tutorial:
#   https://langchain-ai.github.io/langgraph/tutorials/introduction/
#
# ════════════════════════════════════════════════════════════════════

from typing import TypedDict, List, Optional, Any, Annotated
import operator

try:
    from langgraph.graph import StateGraph, START, END, MessagesState
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from agents.react_agent import ALL_TOOLS, REACT_SYSTEM_PROMPT, MAX_ITERATIONS


# ── State schema ─────────────────────────────────────────────────────────────
#
# ReAct state is MESSAGE-BASED, not field-based.
#
# KEY DIFFERENCE from linear AirQualityState TypedDict:
#   Linear: each field (ground_readings, satellite_observations, ...) was
#           written by a dedicated agent, held as structured data.
#   ReAct:  ALL information lives in the `messages` list — the LLM reads
#           its own prior ToolMessages as its "memory" of observations.
#
# The `Annotated[List, operator.add]` reducer means new messages are
# appended to the list, not overwritten (LangGraph merge behaviour).
#
# Auxiliary fields mirror the original state schema so that main.py's
# print_report_section() requires zero changes.

class ReactAirQualityState(TypedDict, total=False):
    # ── Core ReAct state ─────────────────────────────────────────────────────
    messages: Annotated[List[Any], operator.add]  # Full conversation history

    # ── Auxiliary fields (populated by state extractor after loop ends) ──────
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
    iteration_count:             int
    emergency_triggered:         bool
    errors:                      List[Any]


# ── Node: ReAct agent ─────────────────────────────────────────────────────────
#
# This is the ONLY "agent" node. It replaces all nine agent nodes from the
# linear graph. On every entry, it calls the LLM with the full message history
# and the bound tools. It appends the LLM's response to the messages list.

def react_agent_node(state: ReactAirQualityState) -> dict:
    """
    Single ReAct agent node. Calls LLM with all tools bound.
    Returns {"messages": [ai_response]} which LangGraph appends to state.messages.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
    except ImportError:
        return {"errors": ["langchain_openai not installed"]}

    from config.settings import air_config

    llm = ChatOpenAI(
        model=air_config.model_name,
        temperature=air_config.temperature,
        api_key=air_config.openai_api_key,
        max_tokens=2000,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = state.get("messages", [])

    # Inject system prompt on first call if not already present
    from langchain_core.messages import SystemMessage as SM
    if not messages or not isinstance(messages[0], SM):
        messages = [SM(content=REACT_SYSTEM_PROMPT)] + list(messages)

    response = llm_with_tools.invoke(messages)

    iteration = state.get("iteration_count", 0) + 1
    n_tool_calls = len(getattr(response, "tool_calls", []) or [])
    print(f"  [LangGraph ReAct] Iteration {iteration}: "
          f"{'→ ' + str(n_tool_calls) + ' tool call(s)' if n_tool_calls else '→ FINAL RESPONSE (no tool calls)'}")

    return {
        "messages":       [response],
        "iteration_count": iteration,
    }


# ── Conditional edge: should_continue ────────────────────────────────────────
#
# This is what makes the graph a loop instead of a straight line.
# should_continue() is called AFTER react_agent_node every time.
# It inspects the last AIMessage and routes accordingly.

def should_continue(state: ReactAirQualityState) -> str:
    """
    Routing function for the conditional edge after react_agent_node.

    Returns:
        "continue" → route to tool_node (execute tool calls, loop back)
        "end"      → route to END (write final report from last message)
    """
    messages    = state.get("messages", [])
    iteration   = state.get("iteration_count", 0)
    last_msg    = messages[-1] if messages else None

    tool_calls  = getattr(last_msg, "tool_calls", None) or []

    if iteration >= MAX_ITERATIONS:
        print(f"  [LangGraph ReAct] Max iterations ({MAX_ITERATIONS}) reached → forcing END")
        return "end"

    if tool_calls:
        return "continue"   # → ToolNode → back to react_agent_node
    return "end"            # → END


# ── Build and compile the graph ───────────────────────────────────────────────

def build_react_graph():
    """
    Construct and compile the LangGraph ReAct graph.

    Graph topology:
        START → react_agent_node
                     ↕ (loop via conditional edge)
                  tool_node
                     ↓
                    END

    Falls back gracefully if LangGraph is not installed.
    """
    if not LANGGRAPH_AVAILABLE:
        print("  [WARNING] LangGraph not installed — ReAct graph unavailable.")
        return None

    graph = StateGraph(ReactAirQualityState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("react_agent_node", react_agent_node)

    # ToolNode is a LangGraph built-in that:
    #   1. reads tool_calls from the last AIMessage
    #   2. executes each tool via its .invoke() method
    #   3. appends the results as ToolMessages to state.messages
    graph.add_node("tool_node", ToolNode(ALL_TOOLS))

    # ── Wire the graph ────────────────────────────────────────────────────────
    graph.add_edge(START, "react_agent_node")

    # Conditional edge: after every react_agent_node call, check should_continue
    graph.add_conditional_edges(
        "react_agent_node",
        should_continue,
        {
            "continue": "tool_node",   # LLM wants to call tools → execute them
            "end":      END,           # LLM is done → finish
        },
    )

    # After tool execution, always go back to the agent for the next thought
    graph.add_edge("tool_node", "react_agent_node")

    return graph.compile()


# ── Fallback: run without LangGraph ──────────────────────────────────────────

def run_react_pipeline(scenario: str = "standard") -> dict:
    """
    Fallback that runs the ReAct agent without the LangGraph graph wrapper.
    Used when LangGraph is not installed.
    Delegates entirely to agents/react_agent.py which has its own loop.
    """
    from agents.react_agent import run_react_agent
    return run_react_agent(scenario)
