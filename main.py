#!/usr/bin/env python3
# main.py  (ReAct version)
#
# Drop-in replacement for the original linear main.py.
# The command-line interface and print_report_section() output are identical.
#
# USAGE:
#   python main.py                    # standard scenario
#   python main.py episode            # severe pollution episode
#
# THREE RUN MODES (selected automatically, same as original):
#
#   Mode 1 — DEMO (no API key):
#     Simulates the ReAct loop deterministically using all observation tools
#     in a data-driven order, then calls all applicable action tools.
#     No LLM required. Full structured output.
#
#   Mode 2 — LANGGRAPH (LangGraph installed, no API key):
#     N/A — LangGraph ReAct graph requires a live LLM (bind_tools needs responses).
#     Falls back to Mode 1 demo run.
#
#   Mode 3 — LIVE LLM (LangGraph + OpenAI API key):
#     Full ReAct loop. LLM selects tools dynamically, reasons between steps,
#     and produces a live situation report.
#     Requires: pip install -r requirements.txt && OPENAI_API_KEY set in .env
#
# ══════════════════════════════════════════════════════════════════════════════
# WHAT CHANGED FROM THE LINEAR VERSION
# ══════════════════════════════════════════════════════════════════════════════
#
#  File              Linear                         ReAct
#  ─────────────────────────────────────────────────────────────────────────
#  main.py           imports build_graph()          imports build_react_graph()
#                    9-node sequential graph        1-node loop graph
#
#  graph/            pollution_graph.py             react_graph.py
#                    9 add_edge() calls             1 add_conditional_edges()
#                    AirQualityState TypedDict      ReactAirQualityState
#                    (field-per-agent design)       (messages list design)
#
#  agents/           all_agents.py (9 functions)   react_agent.py (1 function)
#                    each agent: 1 tool + 1 LLM    1 agent: all tools + 1 LLM loop
#                    system prompt per agent        1 master system prompt
#
#  tools/            UNCHANGED                      UNCHANGED
#  data/             UNCHANGED                      UNCHANGED
#  config/           UNCHANGED                      UNCHANGED
# ══════════════════════════════════════════════════════════════════════════════

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.simulation import SCENARIOS, describe_scenario


# ── Report printer ─────────────────────────────────────────────────────────────
# Identical to the original main.py — the ReAct agent populates the same keys.

def print_report_section(state: dict) -> None:
    """Print a human-readable summary of the completed analysis to stdout."""
    print("\n\n" + "=" * 70)
    print("  FINAL SITUATION REPORT EXTRACT")
    print("=" * 70)

    readings = state.get("ground_readings", [])
    if readings:
        print(f"\nGROUND SENSOR NETWORK ({len(readings)} stations)")
        print("-" * 50)
        for r in sorted(readings, key=lambda x: -x.get("aqi", 0)):
            marker = (
                " *** CRITICAL" if r["aqi_category"] in ("very_unhealthy", "hazardous") else
                " ** UNHEALTHY" if r["aqi_category"] == "unhealthy" else
                " * SENSITIVE"  if r["aqi_category"] == "unhealthy_sensitive" else ""
            )
            print(f"  {r['station_id']} | {r['district']:<30} | AQI={r['aqi']:>4} "
                  f"({r['aqi_category']:<25}) | "
                  f"PM2.5={r['pm25_ug_m3']:>6} | SO2={r['so2_ug_m3']:>6} | "
                  f"NO2={r['no2_ug_m3']:>6}{marker}")

    sat_obs = state.get("satellite_observations", [])
    if sat_obs:
        print(f"\nSATELLITE OBSERVATIONS ({len(sat_obs)} platforms)")
        print("-" * 50)
        for o in sat_obs:
            aod   = o.get("aerosol_optical_depth")
            fires = o.get("active_fire_count", 0)
            plume = "PLUME DETECTED" if o.get("plume_detected") else ""
            print(f"  {o['satellite']:<15} | AOD={aod if aod else 'N/A':>6} | "
                  f"Fires={fires:>3} | {plume}")
            if o.get("plume_origin_description"):
                print(f"    Plume origin: {o['plume_origin_description']}")

    met = state.get("meteorological_summary", {})
    if met:
        print("\nMETEOROLOGICAL CONDITIONS")
        print("-" * 50)
        print(f"  Wind: {met.get('wind_speed_ms')} m/s from "
              f"{met.get('wind_direction_label', '')} ({met.get('wind_direction_deg')} deg)")
        print(f"  Mixing height: {met.get('mixing_height_m')} m | "
              f"Stability class: {met.get('stability_class')} ({met.get('stability_label')})")
        print(f"  Ventilation coefficient: {met.get('ventilation_coefficient_m2_s')} m²/s "
              f"[{met.get('dispersion_quality', '').upper()}]")
        print(f"  Visibility: {met.get('visibility_km')} km | "
              f"Accumulation risk: {met.get('pollution_accumulation_risk', '').upper()}")
        print(f"  Forecast: {met.get('forecast_12h', '')}")

    sources = state.get("pollution_sources", [])
    if sources:
        non_compliant = [s for s in sources if s.get("compliance_status") != "compliant"]
        print(f"\nPOLLUTION SOURCES ({len(sources)} in inventory, {len(non_compliant)} non-compliant)")
        print("-" * 50)
        for s in sources:
            nc_flag = " *** " + s["compliance_status"].upper() if s["compliance_status"] != "compliant" else ""
            print(f"  {s['source_id']} | {s['source_name']:<40} | {s['source_category']:<15} | "
                  f"{s['operating_status']}{nc_flag}")

    health = state.get("health_impact", {})
    if health:
        print("\nHEALTH IMPACT")
        print("-" * 50)
        print(f"  Max AQI: {health.get('max_aqi')} | Avg AQI: {health.get('avg_aqi')}")
        print(f"  Max PM2.5: {health.get('max_pm25_ug_m3')} ug/m3")
        print(f"  Hospital alert level: {health.get('hospital_alert_level', '').upper()}")
        print(f"  Emergency response: {health.get('emergency_response', False)}")

    mitigations   = state.get("mitigation_recommendations", [])
    public_alerts = state.get("public_alerts", [])
    reg_actions   = state.get("regulatory_actions", [])

    print("\nACTIONS TAKEN")
    print("-" * 50)
    print(f"  Mitigation recommendations : {len(mitigations)}")
    print(f"  Public alerts issued       : {len(public_alerts)}")
    print(f"  Regulatory actions         : {len(reg_actions)}")

    for r in mitigations:
        print(f"  [REC-{r.get('priority','?').upper():9}] "
              f"Expected AQI reduction: {r.get('expected_aqi_reduction', 0):.1f} pts")
    for a in public_alerts:
        print(f"  [ALERT-{a.get('severity','?').upper():8}] "
              f"{a.get('districts_notified', 0)} district(s) notified")
    for e in reg_actions:
        if isinstance(e, dict) and e.get("action_type"):
            print(f"  [REGULATORY] {e.get('action_type','').upper()} — {e.get('source_name','')}")

    report = state.get("situation_report")
    if report and not report.startswith("[DEMO"):
        print("\nSITUATION REPORT")
        print("-" * 50)
        print(report)
    elif report:
        print(f"\n{report[:400]}")

    print("\n" + "=" * 70)
    print(f"  Errors encountered: {len(state.get('errors', []))}")
    if state.get("errors"):
        for err in state["errors"]:
            print(f"    {err}")
    print("=" * 70 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    scenario = "standard"
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in SCENARIOS:
            scenario = arg
        else:
            print(f"Unknown scenario '{arg}'. Available: {list(SCENARIOS.keys())}")
            print("Defaulting to 'standard'.")

    print(f"\nScenario: {scenario}")
    print(f"Description: {describe_scenario(scenario)}")

    # ── Attempt to use the LangGraph ReAct graph ───────────────────────────────
    # The LangGraph graph only works with a live LLM (bind_tools requires responses).
    # In demo mode (no API key) we fall straight through to run_react_agent().

    from agents.base import _demo_mode

    langgraph_available = False
    compiled_graph = None

    if not _demo_mode():
        try:
            from graph.react_graph import build_react_graph
            compiled_graph = build_react_graph()
            if compiled_graph is not None:
                langgraph_available = True
        except Exception as e:
            print(f"  [WARNING] Could not build LangGraph ReAct graph: {e}")

    if langgraph_available and compiled_graph is not None:
        print("\nRunning via LangGraph ReAct graph (graph.stream)...\n")

        from langchain_core.messages import HumanMessage

        initial_state = {
            "messages": [
                HumanMessage(content=(
                    f"Analyse the current air quality situation for scenario='{scenario}'. "
                    "Gather all required data, issue appropriate actions, and produce "
                    "the final situation report. Begin by fetching ground sensor data."
                ))
            ],
            "target_region":              scenario,
            "ground_readings":            [],
            "satellite_observations":     [],
            "meteorological_summary":     {},
            "pollution_sources":          [],
            "health_impact":              {},
            "risk_scores":                [],
            "mitigation_recommendations": [],
            "public_alerts":              [],
            "regulatory_actions":         [],
            "situation_report":           None,
            "current_agent":              "react_agent_node",
            "iteration_count":            0,
            "emergency_triggered":        False,
            "errors":                     [],
        }

        final_state = None
        for step in compiled_graph.stream(initial_state):
            node_name  = list(step.keys())[0]
            node_state = step[node_name]
            print(f"  [LangGraph] Node: {node_name}")
            final_state = node_state

        if final_state is None:
            final_state = compiled_graph.invoke(initial_state)

        # Extract structured data from message history for the report printer
        from agents.react_agent import _extract_state_from_tool_log, _make_empty_state
        tool_log = []
        for msg in final_state.get("messages", []):
            # ToolMessages contain the raw tool results in their content
            from langchain_core.messages import ToolMessage, AIMessage
            if isinstance(msg, ToolMessage):
                import json as _json
                try:
                    result = _json.loads(msg.content)
                except Exception:
                    result = {"raw": msg.content}
                # We can't recover tool name from ToolMessage alone, skip extraction
                # The final state will at least have situation_report from AIMessage
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_state["situation_report"] = msg.content

        # Fallback: reconstruct state via react_agent module
        base_state = _make_empty_state(scenario)
        base_state.update({k: v for k, v in final_state.items()
                           if k not in ("messages",) and v is not None})
        final_state = base_state

    else:
        print("\nRunning ReAct agent directly (no LangGraph graph)...\n")
        from agents.react_agent import run_react_agent
        final_state = run_react_agent(scenario)

    print_report_section(final_state)
    return final_state


if __name__ == "__main__":
    main()
