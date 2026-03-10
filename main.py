#!/usr/bin/env python3
# main.py
# Entry point for the Air Pollution Monitoring Agent System.
#
# USAGE:
#   python main.py                    # standard scenario (typical urban day)
#   python main.py episode            # severe pollution episode scenario
#   python main.py standard           # explicit standard scenario
#
# THREE RUN MODES (selected automatically):
#
#   Mode 1 — DEMO (no API key, no LangGraph):
#     Runs the full pipeline using simulation data.
#     All agents execute and produce structured output.
#     LLM analysis sections contain demo placeholders.
#     Requires: stdlib only
#
#   Mode 2 — LANGGRAPH (LangGraph installed, no API key):
#     Runs via LangGraph StateGraph with graph.stream().
#     Full agent pipeline with node-by-node streaming output.
#     LLM sections contain demo placeholders.
#     Requires: pip install langgraph langchain langchain-core
#
#   Mode 3 — LIVE LLM (LangGraph + OpenAI API key):
#     Full system with real GPT-4o analysis at each agent step.
#     Requires: pip install -r requirements.txt && OPENAI_API_KEY set in .env

import sys
import os

# Allow running from project root or from the package directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.models import make_agent_state
from data.simulation import SCENARIOS, describe_scenario


def print_report_section(state: dict) -> None:
    """Print a human-readable summary of the completed analysis to stdout."""
    print("\n\n" + "=" * 70)
    print("  FINAL SITUATION REPORT EXTRACT")
    print("=" * 70)

    # Ground sensor summary
    readings = state.get("ground_readings", [])
    if readings:
        print(f"\nGROUND SENSOR NETWORK ({len(readings)} stations)")
        print("-" * 50)
        for r in sorted(readings, key=lambda x: -x.get("aqi", 0)):
            marker = " *** CRITICAL" if r["aqi_category"] in ("very_unhealthy", "hazardous") else \
                     " ** UNHEALTHY" if r["aqi_category"] in ("unhealthy",) else \
                     " * SENSITIVE"  if r["aqi_category"] in ("unhealthy_sensitive",) else ""
            print(f"  {r['station_id']} | {r['district']:<30} | AQI={r['aqi']:>4} ({r['aqi_category']:<25}) | "
                  f"PM2.5={r['pm25_ug_m3']:>6} | SO2={r['so2_ug_m3']:>6} | NO2={r['no2_ug_m3']:>6}{marker}")

    # Satellite summary
    sat_obs = state.get("satellite_observations", [])
    if sat_obs:
        print(f"\nSATELLITE OBSERVATIONS ({len(sat_obs)} platforms)")
        print("-" * 50)
        for o in sat_obs:
            aod = o.get("aerosol_optical_depth")
            fires = o.get("active_fire_count", 0)
            plume = "PLUME DETECTED" if o.get("plume_detected") else ""
            print(f"  {o['satellite']:<15} | AOD={aod if aod else 'N/A':>6} | "
                  f"Fires={fires:>3} | {plume}")
            if o.get("plume_origin_description"):
                print(f"    Plume origin: {o['plume_origin_description']}")

    # Met summary
    met = state.get("meteorological_summary", {})
    if met:
        print(f"\nMETEOROLOGICAL CONDITIONS")
        print("-" * 50)
        print(f"  Wind: {met.get('wind_speed_ms')} m/s from {met.get('wind_direction_label', '')} "
              f"({met.get('wind_direction_deg')} deg)")
        print(f"  Mixing height: {met.get('mixing_height_m')} m | "
              f"Stability class: {met.get('stability_class')} ({met.get('stability_label')})")
        print(f"  Ventilation coefficient: {met.get('ventilation_coefficient_m2_s')} m2/s "
              f"[{met.get('dispersion_quality', '').upper()}]")
        print(f"  Visibility: {met.get('visibility_km')} km | "
              f"Accumulation risk: {met.get('pollution_accumulation_risk', '').upper()}")
        print(f"  Forecast: {met.get('forecast_12h', '')}")

    # Source attribution
    sources = state.get("pollution_sources", [])
    if sources:
        non_compliant = [s for s in sources if s.get("compliance_status") != "compliant"]
        print(f"\nPOLLUTION SOURCES ({len(sources)} in inventory, {len(non_compliant)} non-compliant)")
        print("-" * 50)
        for s in sources:
            nc_flag = " *** " + s["compliance_status"].upper() if s["compliance_status"] != "compliant" else ""
            print(f"  {s['source_id']} | {s['source_name']:<40} | {s['source_category']:<15} | "
                  f"{s['operating_status']}{nc_flag}")

    # Health impact
    health = state.get("health_impact", {})
    if health:
        print(f"\nHEALTH IMPACT")
        print("-" * 50)
        print(f"  Max AQI: {health.get('max_aqi')} | Avg AQI: {health.get('avg_aqi')}")
        print(f"  Max PM2.5: {health.get('max_pm25_ug_m3')} ug/m3")
        print(f"  Hospital alert level: {health.get('hospital_alert_level', '').upper()}")
        print(f"  Emergency response: {health.get('emergency_response', False)}")

    # Actions
    mitigations    = state.get("mitigation_recommendations", [])
    public_alerts  = state.get("public_alerts", [])
    reg_actions    = state.get("regulatory_actions", [])

    print(f"\nACTIONS TAKEN")
    print("-" * 50)
    print(f"  Mitigation recommendations: {len(mitigations)}")
    print(f"  Public alerts issued:        {len(public_alerts)}")
    print(f"  Regulatory actions:          {len(reg_actions)}")
    for r in mitigations:
        print(f"  [REC-{r.get('priority','?').upper():9}] "
              f"Expected AQI reduction: {r.get('expected_aqi_reduction', 0):.1f} pts")
    for a in public_alerts:
        print(f"  [ALERT-{a.get('severity','?').upper():8}] {a.get('districts_notified', 0)} district(s) notified")
    for e in reg_actions:
        if isinstance(e, dict) and e.get("action_type"):
            print(f"  [REGULATORY] {e.get('action_type','').upper()} — {e.get('source_name','')}")

    # Full situation report
    report = state.get("situation_report")
    if report and not report.startswith("[DEMO"):
        print(f"\nSITUATION REPORT")
        print("-" * 50)
        print(report)
    elif report:
        print(f"\n{report[:200]}")

    print("\n" + "=" * 70)
    print(f"  Errors encountered: {len(state.get('errors', []))}")
    if state.get("errors"):
        for err in state["errors"]:
            print(f"    {err}")
    print("=" * 70 + "\n")


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

    # Build initial state
    initial_state = make_agent_state()
    initial_state["target_region"] = scenario

    # Try LangGraph first
    langgraph_available = False
    try:
        from graph.pollution_graph import build_graph, run_direct_pipeline
        compiled_graph = build_graph()
        if compiled_graph is not None:
            langgraph_available = True
    except ImportError:
        compiled_graph = None

    if langgraph_available and compiled_graph is not None:
        print("\nRunning via LangGraph StateGraph (graph.stream)...\n")
        final_state = None
        for step in compiled_graph.stream(initial_state):
            node_name = list(step.keys())[0]
            node_state = step[node_name]
            print(f"  [GRAPH] Node completed: {node_name}")
            final_state = node_state
        if final_state is None:
            final_state = compiled_graph.invoke(initial_state)
    else:
        print("\nLangGraph not installed — running direct sequential pipeline...\n")
        from graph.pollution_graph import run_direct_pipeline
        final_state = run_direct_pipeline(initial_state)

    print_report_section(final_state)
    return final_state


if __name__ == "__main__":
    main()
