# graph/pollution_graph.py
# LangGraph StateGraph for the Air Pollution Monitoring Agent pipeline.
#
# Graph topology:
#
#   START
#     -> supervisor_init
#     -> ground_sensor_agent
#     -> satellite_agent
#     -> meteorological_agent
#     -> source_identification_agent
#     -> health_impact_agent
#     -> mitigation_agent
#     -> alert_agent
#     -> supervisor_synthesis
#   END
#
# The pipeline is strictly sequential: each agent enriches the shared state
# before passing it to the next. All domain agents run regardless of scenario
# severity. The emergency flag in state can be read by downstream agents to
# escalate their outputs, but does not alter the graph topology.

from typing import TypedDict, List, Optional, Any

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from agents.all_agents import (
    supervisor_init_agent,
    ground_sensor_agent,
    satellite_agent,
    meteorological_agent,
    source_identification_agent,
    health_impact_agent,
    mitigation_agent,
    alert_agent,
    supervisor_synthesis_agent,
)


# ---------------------------------------------------------------------------
# TypedDict state schema for LangGraph
# ---------------------------------------------------------------------------

class AirQualityState(TypedDict, total=False):
    target_region:               Optional[str]
    analysis_timestamp:          Optional[str]
    ground_readings:             List[Any]
    satellite_observations:      List[Any]
    meteorological_summary:      Optional[Any]
    pollution_sources:           List[Any]
    ground_analysis:             Optional[str]
    satellite_analysis:          Optional[str]
    source_analysis:             Optional[str]
    met_analysis:                Optional[str]
    health_analysis:             Optional[str]
    dispersion_analysis:         Optional[str]
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


def build_graph():
    """
    Construct and compile the LangGraph StateGraph.

    Returns the compiled graph, ready for .invoke() or .stream().
    Falls back gracefully if LangGraph is not installed.
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(AirQualityState)

    # Register all agent nodes
    graph.add_node("supervisor_init",            supervisor_init_agent)
    graph.add_node("ground_sensor_agent",        ground_sensor_agent)
    graph.add_node("satellite_agent",            satellite_agent)
    graph.add_node("meteorological_agent",       meteorological_agent)
    graph.add_node("source_identification_agent", source_identification_agent)
    graph.add_node("health_impact_agent",        health_impact_agent)
    graph.add_node("mitigation_agent",           mitigation_agent)
    graph.add_node("alert_agent",                alert_agent)
    graph.add_node("supervisor_synthesis",       supervisor_synthesis_agent)

    # Wire the sequential pipeline
    graph.add_edge(START,                         "supervisor_init")
    graph.add_edge("supervisor_init",             "ground_sensor_agent")
    graph.add_edge("ground_sensor_agent",         "satellite_agent")
    graph.add_edge("satellite_agent",             "meteorological_agent")
    graph.add_edge("meteorological_agent",        "source_identification_agent")
    graph.add_edge("source_identification_agent", "health_impact_agent")
    graph.add_edge("health_impact_agent",         "mitigation_agent")
    graph.add_edge("mitigation_agent",            "alert_agent")
    graph.add_edge("alert_agent",                 "supervisor_synthesis")
    graph.add_edge("supervisor_synthesis",        END)

    return graph.compile()


def run_direct_pipeline(initial_state: dict) -> dict:
    """
    Fallback pipeline that runs all agents in sequence without LangGraph.
    Used automatically when LangGraph is not installed.
    """
    state = initial_state.copy()
    agents = [
        supervisor_init_agent,
        ground_sensor_agent,
        satellite_agent,
        meteorological_agent,
        source_identification_agent,
        health_impact_agent,
        mitigation_agent,
        alert_agent,
        supervisor_synthesis_agent,
    ]
    for agent_fn in agents:
        try:
            state = agent_fn(state)
        except Exception as e:
            print(f"  [ERROR in {agent_fn.__name__}]: {e}")
            state["errors"].append({"agent": agent_fn.__name__, "error": str(e)})
    return state
