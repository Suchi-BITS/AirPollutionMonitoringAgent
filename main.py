#!/usr/bin/env python3
# main.py  — AirGuard v3: Context Engineering + Short-Term Memory
#
# ══════════════════════════════════════════════════════════════════════════════
# USAGE
# ══════════════════════════════════════════════════════════════════════════════
#
#   python main.py                      # single standard run
#   python main.py episode              # single episode run
#   python main.py standard --runs 3    # 3-cycle monitoring loop (shows memory)
#   python main.py episode  --runs 2    # 2 episode cycles
#   python main.py --reset-memory       # clear session memory and exit
#
# ══════════════════════════════════════════════════════════════════════════════
# THE KEY DIFFERENCE: MULTI-RUN MONITORING LOOP
# ══════════════════════════════════════════════════════════════════════════════
#
# Without memory (v1/v2), every run is isolated:
#   Run 1 → AQI 140. Advisory issued.  Compliance order issued.
#   Run 2 → AQI 155. Same advisory issued AGAIN.  Same order issued AGAIN.
#   Run 3 → AQI 130. Agent has no idea if things are improving.
#
# With short-term memory (v3), the agent carries context across cycles:
#   Run 1 → AQI 140. Advisory issued.  Memory updated.
#   Run 2 → AQI 155. Memory: "worsening +15 pts".
#            Advisory already issued → suppressed.  Issues WARNING (escalation).
#   Run 3 → AQI 130. Memory: "improving -25 pts".
#            WARNING already issued → suppressed.  Logs monitoring note.
#
# This mirrors a real 30-minute polling loop with deduplication.

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.simulation          import SCENARIOS, describe_scenario
from memory.short_term_memory import SESSION_MEMORY


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report_section(state: dict) -> None:
    """Print a human-readable extract of the run result."""
    print("\n\n" + "=" * 70)
    print("  FINAL SITUATION REPORT EXTRACT")
    print("=" * 70)

    readings = state.get("ground_readings", [])
    if readings:
        print(f"\nGROUND SENSOR NETWORK  ({len(readings)} stations)")
        print("-" * 50)
        for r in sorted(readings, key=lambda x: -x.get("aqi", 0)):
            marker = (
                " *** CRITICAL" if r["aqi_category"] in ("very_unhealthy", "hazardous") else
                " **  UNHEALTHY" if r["aqi_category"] == "unhealthy" else
                " *   SENSITIVE" if r["aqi_category"] == "unhealthy_sensitive" else ""
            )
            print(f"  {r['station_id']} | {r['district']:<30} | AQI={r['aqi']:>4} "
                  f"({r['aqi_category']:<25}) | "
                  f"PM2.5={r['pm25_ug_m3']:>6.1f} | "
                  f"SO2={r['so2_ug_m3']:>6.1f} | "
                  f"NO2={r['no2_ug_m3']:>6.1f}{marker}")

    sat_obs = state.get("satellite_observations", [])
    if sat_obs:
        print(f"\nSATELLITE OBSERVATIONS  ({len(sat_obs)} platforms)")
        print("-" * 50)
        for o in sat_obs:
            aod   = o.get("aerosol_optical_depth")
            fires = o.get("active_fire_count", 0)
            plume = "PLUME DETECTED" if o.get("plume_detected") else ""
            print(f"  {o['satellite']:<15} | AOD={aod if aod else 'N/A':>6} | "
                  f"Fires={fires:>3} | {plume}")

    met = state.get("meteorological_summary") or {}
    if met:
        print("\nMETEOROLOGICAL CONDITIONS")
        print("-" * 50)
        print(f"  Wind: {met.get('wind_speed_ms')} m/s from "
              f"{met.get('wind_direction_label', '')} ({met.get('wind_direction_deg')}°)")
        print(f"  Mixing height: {met.get('mixing_height_m')} m | "
              f"Stability: {met.get('stability_class')} ({met.get('stability_label')})")
        print(f"  Ventilation coefficient: {met.get('ventilation_coefficient_m2_s')} m²/s "
              f"[{met.get('dispersion_quality', '').upper()}]")
        print(f"  Forecast: {met.get('forecast_12h', '')}")

    sources = state.get("pollution_sources", [])
    if sources:
        non_c = [s for s in sources if s.get("compliance_status") != "compliant"]
        print(f"\nPOLLUTION SOURCES  ({len(sources)} in inventory, {len(non_c)} non-compliant)")
        print("-" * 50)
        for s in sources:
            nc = " *** " + s["compliance_status"].upper() if s["compliance_status"] != "compliant" else ""
            print(f"  {s['source_id']} | {s['source_name']:<40} | "
                  f"{s['source_category']:<15} | {s['operating_status']}{nc}")

    health = state.get("health_impact") or {}
    if health:
        print("\nHEALTH IMPACT")
        print("-" * 50)
        print(f"  Max AQI: {health.get('max_aqi')} | Avg AQI: {health.get('avg_aqi')}")
        print(f"  Max PM2.5: {health.get('max_pm25_ug_m3')} µg/m³")
        print(f"  Hospital alert: {health.get('hospital_alert_level', '').upper()}")
        print(f"  Emergency:      {health.get('emergency_response', False)}")

    mits   = state.get("mitigation_recommendations", [])
    alerts = state.get("public_alerts", [])
    regs   = state.get("regulatory_actions", [])

    print("\nACTIONS TAKEN  (this run)")
    print("-" * 50)
    print(f"  Mitigation recommendations : {len(mits)}")
    print(f"  Public alerts issued       : {len(alerts)}")
    print(f"  Regulatory actions         : {len(regs)}")
    for r in mits:
        print(f"    [REC-{r.get('priority','?').upper():9}] "
              f"expected Δ AQI: -{r.get('expected_aqi_reduction', 0):.1f}")
    for a in alerts:
        if isinstance(a, dict):
            print(f"    [ALERT-{a.get('severity','?').upper():8}] "
                  f"{a.get('districts_notified', 0)} district(s) notified")
    for e in regs:
        if isinstance(e, dict) and e.get("action_type"):
            print(f"    [REG]  {e.get('action_type','').upper()} — {e.get('source_name','')}")

    # Session-level memory summary
    ah = SESSION_MEMORY.get_alert_history_summary()
    print("\nSESSION MEMORY SUMMARY")
    print("-" * 50)
    print(f"  Runs this session          : {SESSION_MEMORY.run_count}")
    print(f"  Public alerts (session)    : {ah['public_alerts_issued']}")
    print(f"  Hospital alerts (session)  : {ah['hospital_alerts_issued']}")
    print(f"  Regulatory actions (session): {ah['regulatory_actions_issued']}")
    print(f"  Deferred actions pending   : {ah['deferred_actions_pending']}")
    print(f"  Episode declared           : {SESSION_MEMORY.episode_declared}")
    if SESSION_MEMORY.episode_declared:
        print(f"  Episode peak AQI           : {SESSION_MEMORY.episode_peak_aqi}")

    report = state.get("situation_report", "")
    if report:
        print("\nSITUATION REPORT (excerpt)")
        print("-" * 50)
        # Print first 500 chars; full report is in state["situation_report"]
        print(report[:1500] + ("..." if len(report) > 1500 else ""))

    print("\n" + "=" * 70)
    if state.get("errors"):
        print(f"  Errors: {state['errors']}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_single(scenario: str) -> dict:
    from agents.react_agent import run_react_agent
    return run_react_agent(scenario)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-run monitoring loop
# ─────────────────────────────────────────────────────────────────────────────

def run_monitoring_loop(scenario: str, n_runs: int, delay_secs: float = 0.3) -> None:
    """
    Simulate N consecutive 30-minute monitoring cycles.
    Demonstrates memory deduplication, trend detection, and episode tracking.
    """
    print(f"\n{'=' * 70}")
    print(f"  MONITORING LOOP: {n_runs} cycles | scenario={scenario.upper()}")
    print(f"  Memory window: {SESSION_MEMORY.max_aqi_history} runs")
    print(f"{'=' * 70}\n")

    for i in range(n_runs):
        print(f"\n{'─' * 70}")
        print(f"  MONITORING CYCLE {i + 1} of {n_runs}")
        print(f"{'─' * 70}")

        state = run_single(scenario)
        print_report_section(state)

        if i < n_runs - 1:
            print(f"  [Simulating 30-min interval... ({delay_secs}s)]\n")
            time.sleep(delay_secs)

    # Final session summary
    print(f"\n{'=' * 70}")
    print("  SESSION COMPLETE — MEMORY SUMMARY")
    print(f"{'=' * 70}")
    print(SESSION_MEMORY.context_summary())

    trend = SESSION_MEMORY.get_aqi_trend()
    if trend.get("available") and SESSION_MEMORY.aqi_history:
        print(f"\n  AQI history ({trend['runs_in_window']} runs):")
        for snap in SESSION_MEMORY.aqi_history:
            bar = "█" * min(snap.max_aqi // 10, 50)
            print(f"    {snap.timestamp[11:16]} | AQI {snap.max_aqi:>4} | {bar}")

    print(f"{'=' * 70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--reset-memory" in args:
        SESSION_MEMORY.reset()
        print("Session memory cleared.")
        return

    scenario = "standard"
    n_runs   = 1

    for arg in args:
        if arg.lower() in SCENARIOS:
            scenario = arg.lower()

    for i, arg in enumerate(args):
        if arg == "--runs" and i + 1 < len(args):
            try:
                n_runs = int(args[i + 1])
            except ValueError:
                pass
        elif arg.startswith("--runs="):
            try:
                n_runs = int(arg.split("=", 1)[1])
            except ValueError:
                pass

    print(f"\nScenario    : {scenario}")
    print(f"Cycles      : {n_runs}")
    print(f"Description : {describe_scenario(scenario)}")

    if n_runs > 1:
        run_monitoring_loop(scenario, n_runs)
    else:
        state = run_single(scenario)
        print_report_section(state)


if __name__ == "__main__":
    main()
