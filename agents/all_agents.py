# agents/react_agent.py
#
# ReAct (Reason + Act) agent that replaces the entire linear nine-agent pipeline.
#
# PATTERN:
#   The LLM is given ALL tools at once (sensor + action).
#   It iterates through a Thought → Tool call → Observation loop
#   until it decides it has enough information to write the final report.
#
# LINEAR vs ReAct CONTRAST:
#
#   Linear (old):
#     supervisor_init → ground_sensor → satellite → met → source
#       → health → mitigation → alert → supervisor_synthesis
#     Each node runs EXACTLY ONCE in a hard-coded order.
#     emergency_triggered modifies text but never changes the graph path.
#
#   ReAct (new):
#     A single node runs in a loop. The LLM decides:
#       - which tool to call next
#       - whether to re-call a tool (e.g. re-check sensors after mitigation)
#       - whether to skip a tool (clean day → no emergency tools needed)
#       - when it has enough information to stop and write the report
#
# This file contains:
#   - REACT_SYSTEM_PROMPT   : the master instructions given to the LLM
#   - run_react_agent()     : the main loop
#   - _demo_react_run()     : deterministic fallback when no API key is set

import os
import json
from datetime import datetime
from config.settings import air_config

# ── All tools (sensor + action) gathered into one flat list ─────────────────
from tools.sensor_tools import (
    fetch_ground_sensor_data,
    fetch_satellite_imagery,
    fetch_meteorological_data,
    fetch_emission_inventory,
    fetch_health_risk_tables,
)
from tools.action_tools import (
    issue_public_health_alert,
    issue_regulatory_action,
    log_mitigation_recommendation,
    notify_hospital_network,
    request_traffic_restriction,
    get_action_log,
)

ALL_TOOLS = [
    # ── Observation tools ────────────────────────────────────────────────────
    fetch_ground_sensor_data,       # Ground station PM2.5/NO2/SO2/O3 readings
    fetch_satellite_imagery,        # Sentinel-5P / MODIS / Landsat observations
    fetch_meteorological_data,      # Wind, mixing height, stability class
    fetch_emission_inventory,       # Registered emission sources & compliance
    fetch_health_risk_tables,       # Population, vulnerable groups, C-R coefficients
    # ── Action tools ─────────────────────────────────────────────────────────
    log_mitigation_recommendation,  # Log a prioritised action plan item
    issue_public_health_alert,      # Publish advisory/warning/alert/emergency
    issue_regulatory_action,        # Issue NOV or shutdown order
    notify_hospital_network,        # Alert ED / hospital command
    request_traffic_restriction,    # Submit LEZ / diesel ban / HGV curfew
    get_action_log,                 # Review what has already been logged
]

# ── Maximum loop iterations before forcing a stop ───────────────────────────
MAX_ITERATIONS = 20


# ============================================================================
# ReAct System Prompt
# ============================================================================
# This replaces the nine separate system prompts that each linear agent held.
# One prompt, all domain knowledge, full autonomy over tool order.

REACT_SYSTEM_PROMPT = f"""You are the Air Quality Monitoring ReAct Agent for {air_config.city}.

You have access to five observation tools and five action tools.
Your job is to:
  1. Gather all data needed to understand the current air quality situation.
  2. Issue appropriate mitigation recommendations, public alerts, and regulatory actions.
  3. Produce a final structured SITUATION REPORT.

═══════════════════════════════════════════════════════════
OBSERVATION TOOLS — call these to gather data
═══════════════════════════════════════════════════════════
fetch_ground_sensor_data(scenario)
  → All station PM2.5, PM10, NO2, SO2, CO, O3 readings + AQI per station.
  → ALWAYS call this first. It gives you the primary pollution picture.

fetch_satellite_imagery(scenario)
  → Sentinel-5P / MODIS AOD, NO2/SO2 columns, fire count, plume detection.
  → Call this after ground sensors to cross-validate and identify hotspots.

fetch_meteorological_data(scenario)
  → Wind speed/direction, mixing height, Pasquill-Gifford stability class.
  → Ventilation coefficient < 3000 m²/s = very poor dispersion = accumulation risk.
  → Call this to understand whether pollution will disperse or accumulate.

fetch_emission_inventory()
  → All registered emission sources, compliance status, primary pollutants.
  → Call this to identify which facilities may be driving exceedances.

fetch_health_risk_tables()
  → District population, sensitive groups, C-R coefficients for PM2.5/NO2.
  → Call this to estimate how many people are at risk and at what severity.

═══════════════════════════════════════════════════════════
ACTION TOOLS — call these to respond to the situation
═══════════════════════════════════════════════════════════
log_mitigation_recommendation(priority, category, target_entity, title,
    description, expected_aqi_reduction, implementation_timeline,
    regulatory_basis, estimated_cost_tier, co_benefits)
  → Log a structured action-plan item.
  → Priority: 'emergency' | 'high' | 'medium' | 'low'
  → Category: 'regulatory' | 'traffic' | 'public_health' | 'operational' | 'infrastructure'

issue_public_health_alert(severity, affected_districts, aqi_level,
    dominant_pollutant, health_message, recommended_actions,
    sensitive_groups_warning, duration_hours, channels)
  → severity: 'advisory' (AQI 101-150) | 'warning' (151-200) |
              'alert' (201-300) | 'emergency' (>300)

issue_regulatory_action(source_id, source_name, violation_type,
    pollutant, measured_concentration, permitted_limit,
    action_type, required_action, compliance_deadline,
    enforcement_authority, regulatory_basis)
  → action_type: 'notice_of_violation' | 'compliance_order' | 'emergency_shutdown_order'

notify_hospital_network(alert_level, affected_districts, primary_pollutant,
    aqi, expected_case_types, expected_volume_increase_pct, special_instructions)
  → alert_level: 'normal' | 'elevated' | 'high' | 'critical'

request_traffic_restriction(restriction_type, affected_zones, vehicles_affected,
    start_time, end_time, reason, legal_basis)
  → restriction_type: 'low_emission_zone' | 'diesel_ban' | 'hgv_ban' | 'odd_even'

get_action_log(limit)
  → Review everything logged so far in this session.

═══════════════════════════════════════════════════════════
REASONING STRATEGY
═══════════════════════════════════════════════════════════
Think step by step before each tool call. After each observation, reason about:
  - What does this data tell me?
  - Is this consistent with other readings I have?
  - Do I need more data before acting?
  - What is the most severe problem I should address first?

DECISION TREE FOR ACTIONS:
  max AQI > 300 (Hazardous)  → emergency_shutdown_order, emergency alert, CRITICAL hospital alert
  max AQI 201-300 (Very Unhealthy) → compliance_order, alert severity, high hospital alert
  max AQI 151-200 (Unhealthy) → notice_of_violation for non-compliant sources, warning severity
  max AQI 101-150 (Unhealthy for Sensitive Groups) → advisory severity, elevated hospital alert
  
  Non-compliant sources → always issue regulatory_action for each one
  Ventilation coefficient < 3000 → add HGV curfew recommendation
  Active fires > 5 → add agricultural burn alert in public health advisory
  Sensitive district population > 50000 → notify hospitals regardless of AQI

═══════════════════════════════════════════════════════════
STOPPING CRITERION
═══════════════════════════════════════════════════════════
Stop calling tools and write your final answer when ALL of the following are true:
  ✓ You have called all five observation tools at least once
  ✓ You have issued at least one public health alert (if AQI > 100)
  ✓ You have issued regulatory actions for all non-compliant sources found
  ✓ You have logged at least one mitigation recommendation

═══════════════════════════════════════════════════════════
FINAL SITUATION REPORT FORMAT
═══════════════════════════════════════════════════════════
Write your final report with these sections:

SECTION 1: EXECUTIVE SUMMARY
SECTION 2: CURRENT AIR QUALITY STATUS  (max AQI, avg AQI, worst stations, dominant pollutants)
SECTION 3: POLLUTION SOURCE ATTRIBUTION (which sources are driving it, compliance status)
SECTION 4: HEALTH IMPACT ASSESSMENT    (population at risk, sensitive groups, hospital level)
SECTION 5: METEOROLOGICAL ASSESSMENT   (dispersion quality, 12h forecast)
SECTION 6: ACTIONS TAKEN               (summary of all tools you called)
SECTION 7: OUTSTANDING PRIORITY ACTIONS

Conclude with:  OVERALL RISK LEVEL: CRITICAL | HIGH | ELEVATED | MODERATE | LOW
                OVERALL AQI FORECAST (next 6 hours): <value>

Disclaimer: {air_config.disclaimer}
"""


# ============================================================================
# Core ReAct loop (live LLM path)
# ============================================================================

def run_react_agent(scenario: str = "standard") -> dict:
    """
    Run the ReAct agent loop.

    Returns a state dict compatible with the original linear pipeline's
    output format so main.py / print_report_section() needs zero changes.

    Args:
        scenario: 'standard' or 'episode'

    Returns:
        dict with all state fields populated
    """
    from agents.base import _demo_mode

    state = _make_empty_state(scenario)

    print("\n" + "=" * 70)
    print(f"  {air_config.system_name} — ReAct Mode")
    print(f"  {air_config.city} | {air_config.region}")
    print(f"  Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Scenario: {scenario.upper()}")
    print(f"  Mode: {'DEMO (no API key)' if _demo_mode() else 'LIVE LLM — GPT-4o'}")
    print("=" * 70)

    if _demo_mode():
        return _demo_react_run(state, scenario)

    return _live_react_run(state, scenario)


# ── Live LLM path ────────────────────────────────────────────────────────────

def _live_react_run(state: dict, scenario: str) -> dict:
    """
    Full ReAct loop with a live LLM.

    The LLM is given all tools at once via bind_tools().
    We loop until the model produces a message with no tool_calls.
    Each tool result is fed back as a ToolMessage so the LLM can reason
    over its own observations before deciding what to call next.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    except ImportError as e:
        print(f"  [ERROR] LangChain not installed: {e}")
        return _demo_react_run(state, scenario)

    llm = ChatOpenAI(
        model=air_config.model_name,
        temperature=air_config.temperature,
        api_key=air_config.openai_api_key,
        max_tokens=2000,
    )

    # Bind ALL tools to the LLM — it can now call any of them at any point
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Tool lookup by name for dispatching
    tool_map = {
        getattr(t, "name", getattr(t, "__name__", str(t))): t
        for t in ALL_TOOLS
    }

    # Initial conversation: system prompt + task description
    messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Analyse the current air quality situation for scenario='{scenario}'. "
            f"Gather all required data, issue appropriate actions, and produce "
            f"the final situation report. Begin by fetching ground sensor data."
        )),
    ]

    tool_results_log = []
    iteration = 0

    print(f"\n[REACT AGENT] Starting ReAct loop (max {MAX_ITERATIONS} iterations)...")

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n  [Iteration {iteration}] Calling LLM...")

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # ── No tool calls → LLM is done, response.content is the final report ──
        if not response.tool_calls:
            print(f"\n  [REACT AGENT] Loop complete after {iteration} iterations.")
            print(f"  Tools called: {len(tool_results_log)}")
            state["situation_report"] = response.content
            state = _extract_state_from_tool_log(state, tool_results_log, scenario)
            _print_completion_summary(state, iteration)
            return state

        # ── Execute each requested tool call ──────────────────────────────────
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})

            print(f"  → Tool call: {tool_name}({_fmt_args(tool_args)})")

            t = tool_map.get(tool_name)
            if not t:
                result_str = f"Error: unknown tool '{tool_name}'"
                result_obj = {"error": result_str}
            else:
                try:
                    result_obj = t.invoke(tool_args)
                    result_str = json.dumps(result_obj, default=str)
                except Exception as ex:
                    result_str = f"Tool error: {ex}"
                    result_obj = {"error": str(ex)}

            tool_results_log.append({
                "tool":   tool_name,
                "args":   tool_args,
                "result": result_obj,
            })

            # Feed the observation back to the LLM as a ToolMessage
            messages.append(ToolMessage(
                content=result_str,
                tool_call_id=tc["id"],
            ))

    # ── Max iterations reached: force a synthesis call without tools ──────────
    print(f"\n  [REACT AGENT] Max iterations ({MAX_ITERATIONS}) reached. Forcing synthesis.")
    state = _extract_state_from_tool_log(state, tool_results_log, scenario)

    summary_msg = HumanMessage(content=(
        "You have reached the maximum number of tool calls. "
        "Write your final SITUATION REPORT now based on all data collected so far. "
        "Do NOT call any more tools."
    ))
    messages.append(summary_msg)

    # Re-invoke without tools bound so the LLM cannot make more tool calls
    plain_llm = ChatOpenAI(
        model=air_config.model_name,
        temperature=air_config.temperature,
        api_key=air_config.openai_api_key,
        max_tokens=2000,
    )
    final_response = plain_llm.invoke(messages)
    state["situation_report"] = final_response.content
    _print_completion_summary(state, iteration)
    return state


# ── Demo path (no API key) ────────────────────────────────────────────────────

def _demo_react_run(state: dict, scenario: str) -> dict:
    """
    Deterministic demo mode.

    Simulates the ReAct loop by calling every observation tool in a logical
    order and then every applicable action tool, without an LLM.

    This demonstrates the ReAct pattern's key property: the tool call sequence
    is data-driven. In live mode the LLM makes these same decisions dynamically.
    """
    print("\n  [DEMO MODE] Simulating ReAct loop (no API key)...")
    print("  In live mode the LLM selects tools dynamically based on observations.\n")

    tool_results_log = []

    def _call(tool, **kwargs):
        name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        print(f"  [ReAct] Thought: I need data from {name}.")
        try:
            # Support both LangChain tool objects (.invoke) and plain functions
            fn = getattr(tool, "func", tool)   # .func unwraps LangChain @tool wrappers
            if hasattr(tool, "invoke") and kwargs:
                result = tool.invoke(kwargs)
            elif hasattr(tool, "invoke"):
                result = tool.invoke({})
            elif kwargs:
                result = fn(**kwargs)
            else:
                result = fn()
        except Exception as ex:
            result = {"error": str(ex)}
        print(f"  [ReAct] Observation: {_summarise(name, result)}")
        tool_results_log.append({"tool": name, "args": kwargs, "result": result})
        return result

    # ── PHASE 1: Observations (data gathering) ────────────────────────────────
    print("  ── Phase 1: Observation ──")

    ground = _call(fetch_ground_sensor_data, scenario=scenario)
    sat    = _call(fetch_satellite_imagery,  scenario=scenario)
    met    = _call(fetch_meteorological_data, scenario=scenario)
    inv    = _call(fetch_emission_inventory)
    health = _call(fetch_health_risk_tables)

    # ── PHASE 2: Reasoning ────────────────────────────────────────────────────
    print("\n  ── Phase 2: Reasoning ──")

    max_aqi         = ground.get("network_max_aqi", 0)
    avg_aqi         = ground.get("network_avg_aqi", 0)
    critical_stns   = ground.get("critical_stations", [])
    all_stations    = ground.get("stations", [])
    non_compliant   = inv.get("non_compliant_sources", [])
    vc              = met.get("ventilation_coefficient_m2_s", 9999)
    fire_count      = sat.get("total_active_fire_count", 0)
    disp_quality    = met.get("dispersion_quality", "good")
    emergency       = max_aqi >= air_config.aqi_very_unhealthy

    # Determine dominant pollutant across all stations
    pol_counts: dict = {}
    for stn in all_stations:
        dp = stn.get("dominant_pollutant", "PM2.5")
        pol_counts[dp] = pol_counts.get(dp, 0) + 1
    dominant_pol = max(pol_counts, key=pol_counts.get) if pol_counts else "PM2.5"

    # Districts with critical readings
    critical_districts = list({
        stn.get("district", "Unknown")
        for stn in all_stations
        if stn.get("aqi_category") in ("unhealthy", "very_unhealthy", "hazardous")
    })

    print(f"  max AQI={max_aqi}, dominant={dominant_pol}, "
          f"non-compliant sources={len(non_compliant)}, "
          f"VC={vc} m²/s ({disp_quality}), fires={fire_count}")

    if emergency:
        print("  *** ReAct reasoning: emergency threshold exceeded → escalating all actions ***")
    else:
        print(f"  ReAct reasoning: AQI {max_aqi} → {'elevated' if max_aqi > 100 else 'moderate'} response")

    # ── PHASE 3: Actions ──────────────────────────────────────────────────────
    print("\n  ── Phase 3: Action ──")

    # 3a. Mitigation recommendations
    if emergency:
        _call(log_mitigation_recommendation,
              priority="emergency",
              category="regulatory",
              target_entity="regulator",
              title="Initiate Emergency Episode Plan — Stage III",
              description=(
                  f"Network max AQI of {max_aqi} has triggered the emergency threshold "
                  f"({air_config.aqi_very_unhealthy}). Activate Stage III of the Air Pollution "
                  "Episode Plan: mandatory industrial curtailments, traffic restrictions, "
                  "and public shelter-in-place advisory."
              ),
              expected_aqi_reduction=40.0,
              implementation_timeline="immediate",
              regulatory_basis="State AQMD Air Pollution Episode Plan — Stage III",
              estimated_cost_tier="high",
              co_benefits=["reduced acute health burden", "legal compliance"],
        )
    else:
        _call(log_mitigation_recommendation,
              priority="high" if max_aqi > 150 else "medium",
              category="regulatory",
              target_entity="regulator",
              title="Issue Compliance Orders to Non-Compliant Emission Sources",
              description=(
                  f"Current network AQI of {max_aqi} correlates with emissions from "
                  f"{len(non_compliant)} non-compliant source(s). Issue formal compliance "
                  "orders requiring emission reductions within 24 hours."
              ),
              expected_aqi_reduction=10.0,
              implementation_timeline="within_24h",
              regulatory_basis="Clean Air Act Section 113 / State AQMD Regulation 2-1",
              estimated_cost_tier="low",
              co_benefits=["improved public health", "regulatory compliance"],
        )

    if max_aqi > 100:
        _call(log_mitigation_recommendation,
              priority="high",
              category="traffic",
              target_entity="city_transport",
              title="Activate Low Emission Zone — Affected Districts",
              description=(
                  f"AQI of {max_aqi} and {disp_quality} dispersion conditions require "
                  "vehicular emission controls. Restrict Euro 4 and older diesel vehicles "
                  "from all critical-reading districts."
              ),
              expected_aqi_reduction=8.5,
              implementation_timeline="within_24h",
              regulatory_basis="Local Air Quality Management Order 2022, Section 4.2",
              estimated_cost_tier="low",
              co_benefits=["noise reduction", "traffic calming"],
        )

    if vc < 6000:
        _call(log_mitigation_recommendation,
              priority="medium",
              category="infrastructure",
              target_entity="city_transport",
              title="HGV Curfew — Poor Dispersion Conditions",
              description=(
                  f"Ventilation coefficient of {vc} m²/s indicates poor to very poor dispersion. "
                  "Suspend HGV operations on the South Highway Corridor 06:00-22:00 until "
                  "ventilation coefficient exceeds 6000 m²/s."
              ),
              expected_aqi_reduction=5.5,
              implementation_timeline="within_24h",
              regulatory_basis="Road Traffic Regulation Act — Air Quality Emergency Provisions",
              estimated_cost_tier="medium",
              co_benefits=["noise reduction", "road safety"],
        )

    # 3b. Public health alert
    if max_aqi >= air_config.aqi_unhealthy_sensitive:
        alert_severity = (
            "emergency" if max_aqi > 300 else
            "alert"     if max_aqi > 200 else
            "warning"   if max_aqi > 150 else
            "advisory"
        )
        _call(issue_public_health_alert,
              severity=alert_severity,
              affected_districts=critical_districts or ["Downtown Core"],
              aqi_level=max_aqi,
              dominant_pollutant=dominant_pol,
              health_message=(
                  f"Air quality has reached {alert_severity.upper()} level (AQI {max_aqi}). "
                  f"The dominant pollutant is {dominant_pol}. "
                  "Prolonged outdoor exposure poses a health risk, particularly for vulnerable groups."
              ),
              recommended_actions=[
                  "Avoid prolonged outdoor exercise",
                  "Keep windows closed",
                  "Wear an N95 mask if outdoor activity is unavoidable",
                  "Check AQI before travelling",
              ],
              sensitive_groups_warning=(
                  "Children, elderly, and those with asthma or cardiovascular disease "
                  "should remain indoors and keep rescue inhalers accessible."
              ),
              duration_hours=12,
              channels=["mobile_app", "sms", "website", "digital_signage", "media"],
        )

    # 3c. Regulatory actions for non-compliant sources
    for source in non_compliant:
        action = "emergency_shutdown_order" if emergency else "compliance_order"
        primary_pol = source.get("primary_pollutants", ["PM2.5"])[0]
        # emission_rate_kg_hr is a dict per-pollutant; extract the primary pollutant rate
        rate_map = source.get("emission_rate_kg_hr", {})
        if isinstance(rate_map, dict):
            measured = float(rate_map.get(primary_pol, rate_map.get("PM2.5", 50.0)))
        else:
            measured = float(rate_map or 50.0)
        permitted = measured * 0.5  # assume limit is 50% of current for demo
        _call(issue_regulatory_action,
              source_id=source.get("source_id", "SRC-UNK"),
              source_name=source.get("source_name", "Unknown Source"),
              violation_type=source.get("compliance_status", "exceedance").replace(" ", "_"),
              pollutant=primary_pol,
              measured_concentration=float(measured),
              permitted_limit=float(permitted),
              action_type=action,
              required_action=(
                  "Immediate cessation of non-compliant emission activities. "
                  "Submit hourly compliance reports to AQMD until normalised."
                  if emergency else
                  "Reduce emission rates to permitted limits within 24 hours. "
                  "Submit compliance confirmation within 48 hours."
              ),
              compliance_deadline=(
                  datetime.now().strftime("%Y-%m-%d") + "T23:59:00"
              ),
              enforcement_authority="State Air Quality Management District",
              regulatory_basis="Clean Air Act Sec. 113; State AQMD Rule 1001",
        )

    # 3d. Hospital network alert (always call for AQI > 100)
    if max_aqi > 100:
        hosp_level = "critical" if emergency else "high" if max_aqi > 150 else "elevated"
        vol_increase = 45.0 if emergency else 20.0 if max_aqi > 150 else 10.0
        _call(notify_hospital_network,
              alert_level=hosp_level,
              affected_districts=critical_districts or ["Downtown Core"],
              primary_pollutant=dominant_pol,
              aqi=max_aqi,
              expected_case_types=[
                  "asthma exacerbation",
                  "COPD exacerbation",
                  "chest pain",
                  "respiratory distress",
              ],
              expected_volume_increase_pct=vol_increase,
              special_instructions=(
                  "Pre-position nebulisers, bronchodilators, and supplemental O2. "
                  "Activate respiratory triage protocol. "
                  + ("Activate emergency surge capacity." if emergency else "")
              ),
        )

    # 3e. Traffic restriction (if dispersion poor)
    if vc < 6000 and max_aqi > 100:
        _call(request_traffic_restriction,
              restriction_type="hgv_ban",
              affected_zones=critical_districts or ["Downtown Core", "East Industrial District"],
              vehicles_affected="Heavy goods vehicles (>3.5t), pre-Euro 5 diesel",
              start_time="06:00",
              end_time="22:00",
              reason=(
                  f"Air quality episode: AQI {max_aqi}, ventilation coefficient {vc} m²/s."
              ),
              legal_basis="Road Traffic Regulation Act s.1(1) — Air Quality Emergency",
        )

    # ── Final: synthesise situation report in demo mode ───────────────────────
    state["situation_report"] = _build_demo_report(
        state, max_aqi, avg_aqi, dominant_pol, non_compliant,
        met, sat, tool_results_log, scenario,
    )
    state = _extract_state_from_tool_log(state, tool_results_log, scenario)

    print(f"\n  [DEMO ReAct] Loop complete. Tools called: {len(tool_results_log)}")
    _print_completion_summary(state, len(tool_results_log))
    return state


# ============================================================================
# State extraction helpers
# ============================================================================

def _extract_state_from_tool_log(state: dict, tool_log: list, scenario: str) -> dict:
    """
    Reconstruct the pipeline-compatible state dict from the flat tool results log.
    Mirrors what the individual linear agents used to populate in state directly.
    """
    for entry in tool_log:
        name   = entry["tool"]
        result = entry["result"]

        if not isinstance(result, dict):
            continue

        if name == "fetch_ground_sensor_data":
            state["ground_readings"] = result.get("stations", [])
            readings = state["ground_readings"]
            if readings:
                max_aqi  = max(r["aqi"] for r in readings)
                avg_aqi  = round(sum(r["aqi"] for r in readings) / len(readings), 1)
                critical = [r for r in readings if r["aqi_category"] in
                            ("unhealthy", "very_unhealthy", "hazardous")]
                state["risk_scores"].append({
                    "domain":  "ground_sensors",
                    "max_aqi": max_aqi, "avg_aqi": avg_aqi,
                    "level":   critical[0]["aqi_category"] if critical else "moderate",
                    "critical_station_count": len(critical),
                })
                if max_aqi >= air_config.aqi_very_unhealthy:
                    state["emergency_triggered"] = True

        elif name == "fetch_satellite_imagery":
            state["satellite_observations"] = result.get("observations", [])
            obs = state["satellite_observations"]
            if obs:
                max_aod = max((o.get("aerosol_optical_depth") or 0) for o in obs)
                fires   = sum(o.get("active_fire_count", 0) for o in obs)
                plumes  = sum(1 for o in obs if o.get("plume_detected"))
                state["risk_scores"].append({
                    "domain":          "satellite",
                    "max_aod":         max_aod,
                    "fire_count":      fires,
                    "plumes_detected": plumes,
                    "level": (
                        "hazardous"    if max_aod > 0.8 else
                        "very_unhealthy" if max_aod > 0.6 else
                        "unhealthy"    if max_aod > 0.4 else "moderate"
                    ),
                })

        elif name == "fetch_meteorological_data":
            state["meteorological_summary"] = result

        elif name == "fetch_emission_inventory":
            state["pollution_sources"] = result.get("sources", [])

        elif name in ("issue_public_health_alert", "notify_hospital_network"):
            state["public_alerts"].append(result)

        elif name in ("issue_regulatory_action",):
            state["regulatory_actions"].append(result)

        elif name == "log_mitigation_recommendation":
            # The tool returns a confirmation dict; reconstruct the full record
            state["mitigation_recommendations"].append({
                **entry["args"],
                "recommendation_id": result.get("recommendation_id", "REC-DEMO"),
                "expected_aqi_reduction": entry["args"].get("expected_aqi_reduction", 0.0),
            })

    # Build health_impact summary from ground readings
    readings = state.get("ground_readings", [])
    if readings:
        max_aqi = max(r["aqi"] for r in readings)
        avg_aqi = round(sum(r["aqi"] for r in readings) / len(readings), 1)
        state["health_impact"] = {
            "max_aqi":             max_aqi,
            "avg_aqi":             avg_aqi,
            "max_pm25_ug_m3":      max((r.get("pm25_ug_m3") or 0) for r in readings),
            "hospital_alert_level": (
                "critical"  if max_aqi > 200 else
                "high"      if max_aqi > 150 else
                "elevated"  if max_aqi > 100 else "normal"
            ),
            "emergency_response":  state.get("emergency_triggered", False),
        }

    return state


def _build_demo_report(state, max_aqi, avg_aqi, dominant_pol,
                       non_compliant, met, sat, tool_log, scenario) -> str:
    """Build a structured demo situation report (no LLM)."""
    n_alerts = sum(1 for e in tool_log if e["tool"] == "issue_public_health_alert")
    n_reg    = sum(1 for e in tool_log if e["tool"] == "issue_regulatory_action")
    n_rec    = sum(1 for e in tool_log if e["tool"] == "log_mitigation_recommendation")
    n_hosp   = sum(1 for e in tool_log if e["tool"] == "notify_hospital_network")

    risk_level = (
        "CRITICAL"  if max_aqi > 300 else
        "HIGH"      if max_aqi > 200 else
        "ELEVATED"  if max_aqi > 150 else
        "MODERATE"  if max_aqi > 100 else
        "LOW"
    )

    aqi_forecast = max(50, max_aqi - 20) if met.get("dispersion_quality") not in ("very_poor", "poor") \
                   else min(500, max_aqi + 30)

    return (
        f"[DEMO MODE — configure OPENAI_API_KEY in .env for live LLM analysis]\n\n"
        f"SITUATION REPORT — {air_config.city} Air Quality Monitoring System\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Scenario: {scenario.upper()}\n\n"
        f"SECTION 1: EXECUTIVE SUMMARY\n"
        f"The air quality monitoring network is reporting a maximum AQI of {max_aqi} "
        f"(average {avg_aqi}) across {len(state.get('ground_readings', []))} stations. "
        f"The dominant pollutant is {dominant_pol}. "
        f"{len(non_compliant)} emission source(s) are currently non-compliant. "
        f"The ReAct agent took {len(tool_log)} tool calls to complete the analysis.\n\n"
        f"SECTION 2: CURRENT AIR QUALITY STATUS\n"
        f"  Max AQI: {max_aqi} | Avg AQI: {avg_aqi} | Dominant pollutant: {dominant_pol}\n"
        f"  Dispersion: {met.get('dispersion_quality', 'N/A').upper()} "
        f"(VC={met.get('ventilation_coefficient_m2_s', 'N/A')} m²/s)\n"
        f"  Satellite AOD: {sat.get('max_aerosol_optical_depth', 'N/A')} | "
        f"Active fires: {sat.get('total_active_fire_count', 0)}\n\n"
        f"SECTION 3: SOURCE ATTRIBUTION\n"
        f"  {len(non_compliant)} non-compliant source(s) identified in emission inventory.\n\n"
        f"SECTION 4: HEALTH IMPACT\n"
        f"  Hospital alert level: {'CRITICAL' if max_aqi > 200 else 'HIGH' if max_aqi > 150 else 'ELEVATED'}\n\n"
        f"SECTION 5: METEOROLOGICAL ASSESSMENT\n"
        f"  {met.get('forecast_12h', 'Forecast unavailable')}\n\n"
        f"SECTION 6: ACTIONS TAKEN\n"
        f"  Mitigation recommendations logged : {n_rec}\n"
        f"  Public health alerts issued       : {n_alerts}\n"
        f"  Regulatory enforcement actions    : {n_reg}\n"
        f"  Hospital network notifications    : {n_hosp}\n\n"
        f"SECTION 7: OUTSTANDING ACTIONS\n"
        f"  Await compliance confirmation from all non-compliant sources.\n"
        f"  Monitor ventilation coefficient for improvement before relaxing restrictions.\n\n"
        f"OVERALL RISK LEVEL: {risk_level}\n"
        f"OVERALL AQI FORECAST (next 6h): {aqi_forecast}\n\n"
        f"Disclaimer: {air_config.disclaimer}"
    )


# ============================================================================
# Utility helpers
# ============================================================================

def _make_empty_state(scenario: str) -> dict:
    """Return the initial state dict matching the original schema."""
    return {
        "target_region":              scenario,
        "analysis_timestamp":         datetime.now().isoformat(),
        "ground_readings":            [],
        "satellite_observations":     [],
        "meteorological_summary":     {},
        "pollution_sources":          [],
        "ground_analysis":            None,
        "satellite_analysis":         None,
        "source_analysis":            None,
        "met_analysis":               None,
        "health_analysis":            None,
        "dispersion_analysis":        None,
        "health_impact":              {},
        "risk_scores":                [],
        "mitigation_recommendations": [],
        "public_alerts":              [],
        "regulatory_actions":         [],
        "situation_report":           None,
        "current_agent":              "react_agent",
        "iteration_count":            0,
        "emergency_triggered":        False,
        "errors":                     [],
    }


def _fmt_args(args: dict) -> str:
    """Format tool call arguments for compact console printing."""
    if not args:
        return ""
    short = {k: (v if len(str(v)) < 40 else str(v)[:37] + "...") for k, v in args.items()}
    return ", ".join(f"{k}={repr(v)}" for k, v in list(short.items())[:3])


def _summarise(tool_name: str, result) -> str:
    """One-line summary of a tool result for console printing."""
    if not isinstance(result, dict):
        return str(result)[:100]
    if tool_name == "fetch_ground_sensor_data":
        return (f"max_aqi={result.get('network_max_aqi')}, "
                f"stations={result.get('station_count')}, "
                f"critical={result.get('critical_stations')}")
    if tool_name == "fetch_satellite_imagery":
        return (f"max_aod={result.get('max_aerosol_optical_depth')}, "
                f"fires={result.get('total_active_fire_count')}, "
                f"plumes={result.get('plume_detections')}")
    if tool_name == "fetch_meteorological_data":
        return (f"vc={result.get('ventilation_coefficient_m2_s')} m²/s, "
                f"dispersion={result.get('dispersion_quality')}")
    if tool_name == "fetch_emission_inventory":
        return (f"sources={result.get('total_sources')}, "
                f"non_compliant={result.get('non_compliant_count')}")
    if tool_name == "issue_public_health_alert":
        return f"alert_id={result.get('alert_id')}, severity={result.get('severity')}"
    if tool_name == "issue_regulatory_action":
        return f"case={result.get('case_number')}, type={result.get('action_type')}"
    if tool_name == "log_mitigation_recommendation":
        return f"rec_id={result.get('recommendation_id')}, priority={result.get('priority')}"
    if tool_name == "notify_hospital_network":
        return f"notif_id={result.get('notification_id')}, level={result.get('alert_level')}"
    return str(result)[:80]


def _print_completion_summary(state: dict, iterations: int) -> None:
    """Print the same completion block as the original supervisor_synthesis_agent."""
    readings = state.get("ground_readings", [])
    max_aqi  = max((r["aqi"] for r in readings), default=0)
    risk_level = (
        "CRITICAL" if max_aqi > 300 else "VERY HIGH" if max_aqi > 200 else
        "HIGH"     if max_aqi > 150 else "ELEVATED"  if max_aqi > 100 else "MODERATE"
    )
    print("\n" + "=" * 70)
    print("  REACT SITUATION REPORT COMPLETE")
    print(f"  Overall Risk Level: {risk_level}")
    print(f"  ReAct iterations:   {iterations}")
    print(f"  Ground readings:    {len(readings)} stations")
    print(f"  Satellite obs:      {len(state.get('satellite_observations', []))} platforms")
    print(f"  Sources identified: {len(state.get('pollution_sources', []))}")
    print(f"  Mitigations logged: {len(state.get('mitigation_recommendations', []))}")
    print(f"  Public alerts:      {len(state.get('public_alerts', []))}")
    print(f"  Regulatory actions: {len(state.get('regulatory_actions', []))}")
    print(f"  Emergency flag:     {state.get('emergency_triggered', False)}")
    print(f"  Disclaimer: {air_config.disclaimer}")
    print("=" * 70)
