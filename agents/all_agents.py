# agents/react_agent.py  — AirGuard v3
#
# ══════════════════════════════════════════════════════════════════════════════
# REACT AGENT WITH CONTEXT ENGINEERING + SHORT-TERM MEMORY
# ══════════════════════════════════════════════════════════════════════════════
#
# Changes from v2 (plain ReAct, no memory):
#
#   1. DYNAMIC SYSTEM PROMPT
#      v2: one static REACT_SYSTEM_PROMPT string baked in at startup.
#      v3: CONTEXT_ENGINEER.build_context_for_run() rebuilds the system prompt
#          every run, injecting the live SESSION MEMORY block and the
#          severity-calibrated response guidance tier.
#
#   2. COMPRESSED TOOL RESULTS
#      v2: raw tool JSON appended directly into ToolMessage — 600-2500 tokens each.
#      v3: CONTEXT_ENGINEER.compress_tool_result() reduces each result ~80%
#          before it enters the message history, preserving all critical numbers.
#
#   3. TRIMMED MESSAGE HISTORY
#      v2: message list grew unboundedly across iterations.
#      v3: CONTEXT_ENGINEER.trim_message_history() collapses stale exchanges
#          into a compact summary before each LLM call.
#
#   4. MEMORY-GATED ACTIONS
#      v2: every alert and regulatory order was unconditionally issued.
#      v3: before dispatching any alert, the agent checks
#          SESSION_MEMORY.can_issue_alert() / can_issue_regulatory_action().
#          Identical alerts are suppressed; escalations are always allowed.
#
#   5. POST-RUN MEMORY UPDATE
#      v2: state was returned and discarded.
#      v3: _post_run_memory_update() writes the run's observations and actions
#          into SESSION_MEMORY so the next run can read them.
#
# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE (per-session view)
# ══════════════════════════════════════════════════════════════════════════════
#
#   SESSION_MEMORY (persists across all calls to run_react_agent() in a process)
#       │ context_summary() → compact text
#       ▼
#   CONTEXT_ENGINEER
#       │ build_context_for_run() → {system_prompt, opening_message, token_estimate}
#       │ compress_tool_result()  → compact JSON string per tool call
#       │ trim_message_history()  → pruned messages list
#       ▼
#   ReAct loop  (one iteration = LLM call + N tool calls + compressed observations)
#       │ tool results → SESSION_MEMORY.can_issue_alert() gate
#       │ final state  → _post_run_memory_update()
#       ▼
#   SESSION_MEMORY updated for next run

import json
from datetime import datetime

from config.settings  import air_config
from memory.short_term_memory import SESSION_MEMORY
from context.context_engineer import CONTEXT_ENGINEER

# ── Tools ─────────────────────────────────────────────────────────────────────
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
    fetch_ground_sensor_data,
    fetch_satellite_imagery,
    fetch_meteorological_data,
    fetch_emission_inventory,
    fetch_health_risk_tables,
    log_mitigation_recommendation,
    issue_public_health_alert,
    issue_regulatory_action,
    notify_hospital_network,
    request_traffic_restriction,
    get_action_log,
]

MAX_ITERATIONS = 20


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_react_agent(scenario: str = "standard") -> dict:
    """
    Execute one monitoring cycle.  Called once per 30-minute poll.

    Steps:
      1. Context engineer builds the dynamic system prompt (memory injected)
      2. ReAct loop: LLM + compressed tool results + trimmed history
      3. Memory updated with this run's snapshots and dispatched actions
    """
    from agents.base import _demo_mode

    state = _make_empty_state(scenario)
    ctx   = CONTEXT_ENGINEER.build_context_for_run(scenario, SESSION_MEMORY)

    print("\n" + "=" * 70)
    print(f"  {air_config.system_name}")
    print(f"  {air_config.city} | {air_config.region}")
    print(f"  Analysis started : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Run #            : {SESSION_MEMORY.run_count + 1} | "
          f"Scenario: {scenario.upper()}")
    print(f"  Mode             : "
          f"{'DEMO (no API key)' if _demo_mode() else 'LIVE — GPT-4o'}")
    print(f"  Context tokens   : ~{ctx['token_estimate']}")
    print("=" * 70)

    if _demo_mode():
        return _demo_react_run(state, scenario, ctx)
    return _live_react_run(state, scenario, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# Live LLM path
# ══════════════════════════════════════════════════════════════════════════════

def _live_react_run(state: dict, scenario: str, ctx: dict) -> dict:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    except ImportError as e:
        print(f"  [ERROR] LangChain not installed: {e}")
        return _demo_react_run(state, scenario, ctx)

    llm = ChatOpenAI(
        model      = air_config.model_name,
        temperature= air_config.temperature,
        api_key    = air_config.openai_api_key,
        max_tokens = 2000,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_map = {
        getattr(t, "name", getattr(t, "__name__", str(t))): t
        for t in ALL_TOOLS
    }

    # System prompt is rebuilt each run — includes live SESSION MEMORY
    messages = [
        SystemMessage(content=ctx["system_prompt"]),
        HumanMessage(content=ctx["initial_human_message"]),
    ]

    tool_log  = []
    iteration = 0

    print(f"\n[REACT] Starting loop (max {MAX_ITERATIONS} iterations)...")

    while iteration < MAX_ITERATIONS:
        iteration += 1

        # CONTEXT ENGINEERING: trim stale history before calling LLM
        messages = CONTEXT_ENGINEER.trim_message_history(messages)

        print(f"\n  [iter {iteration}] LLM call — {len(messages)} messages in context")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # LLM produced final report — no more tool calls
            print(f"\n[REACT] Loop complete — {iteration} iterations")
            state["situation_report"] = response.content
            state = _extract_state(state, tool_log)
            _post_run_memory_update(state, tool_log)
            _print_summary(state, iteration)
            return state

        for tc in tool_calls:
            name   = tc["name"]
            args   = tc.get("args", {})
            print(f"    → {name}({_fmt_args(args)})")

            t = tool_map.get(name)
            result_obj = (
                {"error": f"unknown tool '{name}'"}
                if not t
                else _safe_invoke(t, args)
            )
            tool_log.append({"tool": name, "args": args, "result": result_obj})

            # CONTEXT ENGINEERING: compress before inserting into history
            compressed = CONTEXT_ENGINEER.compress_tool_result(name, result_obj)
            messages.append(ToolMessage(content=compressed, tool_call_id=tc["id"]))

    # Max iterations reached — force a synthesis call
    print(f"\n[REACT] Max iterations reached — forcing final synthesis")
    state = _extract_state(state, tool_log)

    from langchain_core.messages import HumanMessage as HM
    messages.append(HM(content=(
        "Maximum tool iterations reached.  "
        "Write your FINAL SITUATION REPORT now.  Do NOT call any more tools."
    )))
    final = llm_with_tools.invoke(messages)
    state["situation_report"] = final.content
    _post_run_memory_update(state, tool_log)
    _print_summary(state, iteration)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Demo path (deterministic — no LLM required)
# ══════════════════════════════════════════════════════════════════════════════

def _demo_react_run(state: dict, scenario: str, ctx: dict) -> dict:
    """
    Runs deterministically without an LLM API key.
    Shows the full memory + context engineering integration:
      • What the LLM would see at the start of the run (memory block)
      • Compressed observations (what enters the message history)
      • Memory gating (suppressed duplicates printed to console)
      • Memory update after the run
    """
    print("\n  [DEMO] Simulating ReAct + Memory + Context Engineering")
    print("  ── Memory context injected into system prompt ──")
    for line in SESSION_MEMORY.context_summary().split("\n"):
        print(f"  {line}")
    print()

    tool_log: list = []

    def _call(tool_fn, **kwargs):
        """Invoke a tool, compress its result, and append to tool_log."""
        name = getattr(tool_fn, "name", None) or getattr(tool_fn, "__name__", str(tool_fn))
        print(f"  [ReAct] Thought → calling {name}")
        result = _safe_invoke(tool_fn, kwargs)
        compressed = CONTEXT_ENGINEER.compress_tool_result(name, result)
        print(f"  [ReAct] Observation (compressed) → {_one_liner(name, compressed)}")
        tool_log.append({"tool": name, "args": kwargs, "result": result,
                         "compressed": compressed})
        return result

    # ── Phase 1: Observations ─────────────────────────────────────────────────
    print("  ── Phase 1: Observe (compressed) ──")
    ground = _call(fetch_ground_sensor_data, scenario=scenario)
    sat    = _call(fetch_satellite_imagery,  scenario=scenario)
    met    = _call(fetch_meteorological_data, scenario=scenario)
    inv    = _call(fetch_emission_inventory)
    _call(fetch_health_risk_tables)

    # ── Phase 2: Reasoning ────────────────────────────────────────────────────
    print("\n  ── Phase 2: Reason (memory-aware) ──")

    max_aqi        = ground.get("network_max_aqi", 0)
    avg_aqi        = ground.get("network_avg_aqi", 0)
    all_stns       = ground.get("stations", [])
    non_compliant  = inv.get("non_compliant_sources", [])
    vc             = met.get("ventilation_coefficient_m2_s", 9999)
    disp_quality   = met.get("dispersion_quality", "good")
    emergency      = max_aqi >= air_config.aqi_very_unhealthy

    pol_counts: dict = {}
    for stn in all_stns:
        dp = stn.get("dominant_pollutant", "PM2.5")
        pol_counts[dp] = pol_counts.get(dp, 0) + 1
    dominant_pol = max(pol_counts, key=pol_counts.get) if pol_counts else "PM2.5"

    critical_districts = sorted({
        stn.get("district", "Unknown")
        for stn in all_stns
        if stn.get("aqi_category") in ("unhealthy", "very_unhealthy", "hazardous")
    })

    trend = SESSION_MEMORY.get_aqi_trend()
    if trend.get("available") and trend.get("previous_max_aqi") is not None:
        print(f"  Memory → AQI {trend['previous_max_aqi']} → {max_aqi} "
              f"({trend['direction']}, {trend['delta_aqi']:+d} pts)")
        if trend.get("consecutive_unhealthy_runs", 0) >= 3:
            print(f"  Memory → SUSTAINED EPISODE: "
                  f"{trend['consecutive_unhealthy_runs']} consecutive runs > 150")
    else:
        print(f"  Memory → first run — establishing baseline.  max AQI={max_aqi}")

    print(f"  Reasoning → max_aqi={max_aqi}, dominant={dominant_pol}, "
          f"non_compliant={len(non_compliant)}, VC={vc} m²/s")

    # ── Phase 3: Actions (memory-gated) ───────────────────────────────────────
    print("\n  ── Phase 3: Act (memory-gated) ──")

    # 3a. Mitigation recommendations
    if emergency:
        rec = _call(log_mitigation_recommendation,
            priority               = "emergency",
            category               = "regulatory",
            target_entity          = "regulator",
            title                  = "Activate Emergency Episode Plan — Stage III",
            description            = (
                f"Network max AQI {max_aqi} triggered emergency threshold "
                f"({air_config.aqi_very_unhealthy}).  "
                "Activate Stage III: mandatory industrial curtailments, "
                "traffic restrictions, shelter-in-place advisory."
            ),
            expected_aqi_reduction  = 40.0,
            implementation_timeline = "immediate",
            regulatory_basis        = "State AQMD Air Pollution Episode Plan — Stage III",
            estimated_cost_tier     = "high",
            co_benefits             = ["reduced acute health burden", "legal compliance"],
        )
        if isinstance(rec, dict) and rec.get("recommendation_id"):
            SESSION_MEMORY.record_deferred_action(
                recommendation_id = rec["recommendation_id"],
                title             = "Activate Emergency Episode Plan — Stage III",
                target_entity     = "regulator",
                category          = "regulatory",
                priority          = "emergency",
            )
    else:
        _call(log_mitigation_recommendation,
            priority               = "high" if max_aqi > 150 else "medium",
            category               = "regulatory",
            target_entity          = "regulator",
            title                  = "Issue Compliance Orders to Non-Compliant Sources",
            description            = (
                f"AQI {max_aqi}: {len(non_compliant)} non-compliant source(s) identified.  "
                "Issue formal compliance orders requiring emission reductions within 24 hours."
            ),
            expected_aqi_reduction  = 10.0,
            implementation_timeline = "within_24h",
            regulatory_basis        = "Clean Air Act Sec. 113 / State AQMD Regulation 2-1",
            estimated_cost_tier     = "low",
            co_benefits             = ["improved public health", "regulatory compliance"],
        )

    if max_aqi > 100:
        _call(log_mitigation_recommendation,
            priority               = "high",
            category               = "traffic",
            target_entity          = "city_transport",
            title                  = "Activate Low Emission Zone — Affected Districts",
            description            = (
                f"AQI {max_aqi} with {disp_quality} dispersion: restrict Euro 4 and "
                "older diesel vehicles from critical-reading districts."
            ),
            expected_aqi_reduction  = 8.5,
            implementation_timeline = "within_24h",
            regulatory_basis        = "Local Air Quality Management Order 2022, Section 4.2",
            estimated_cost_tier     = "low",
            co_benefits             = ["noise reduction", "traffic calming"],
        )

    if vc < 6000:
        _call(log_mitigation_recommendation,
            priority               = "medium",
            category               = "infrastructure",
            target_entity          = "city_transport",
            title                  = "HGV Curfew — Poor Dispersion Conditions",
            description            = (
                f"VC={vc} m²/s ({disp_quality}): suspend HGV operations on "
                "South Highway Corridor 06:00–22:00 until VC exceeds 6000 m²/s."
            ),
            expected_aqi_reduction  = 5.5,
            implementation_timeline = "within_24h",
            regulatory_basis        = "Road Traffic Regulation Act — Air Quality Emergency Provisions",
            estimated_cost_tier     = "medium",
            co_benefits             = ["noise reduction", "road safety"],
        )

    # 3b. Public health alert — MEMORY GATED
    if max_aqi >= air_config.aqi_unhealthy_sensitive:
        severity = (
            "emergency" if max_aqi > 300 else
            "alert"     if max_aqi > 200 else
            "warning"   if max_aqi > 150 else
            "advisory"
        )
        if SESSION_MEMORY.can_issue_alert("public_health", severity, "network-wide"):
            _call(issue_public_health_alert,
                severity                 = severity,
                affected_districts       = critical_districts or ["Downtown Core"],
                aqi_level                = max_aqi,
                dominant_pollutant       = dominant_pol,
                health_message           = (
                    f"Air quality has reached {severity.upper()} level (AQI {max_aqi}).  "
                    f"Dominant pollutant: {dominant_pol}.  "
                    "Prolonged outdoor exposure poses a health risk."
                ),
                recommended_actions      = [
                    "Avoid prolonged outdoor exercise",
                    "Keep windows closed",
                    "Wear N95 mask if outdoors",
                    "Check AQI before travelling",
                ],
                sensitive_groups_warning = (
                    "Children, elderly, asthma and cardiovascular patients should "
                    "remain indoors and keep rescue inhalers accessible."
                ),
                duration_hours           = 12,
                channels                 = ["mobile_app", "sms", "website",
                                            "digital_signage", "media"],
            )
            SESSION_MEMORY.record_alert("public_health", severity, "network-wide")
        else:
            print(f"  [Memory gate] SUPPRESS public alert — {severity} already issued "
                  "or higher severity was dispatched this session.")

    # 3c. Regulatory actions — MEMORY GATED per source
    for source in non_compliant:
        action_type = "emergency_shutdown_order" if emergency else "compliance_order"
        source_id   = source.get("source_id", "SRC-UNK")
        if SESSION_MEMORY.can_issue_regulatory_action(source_id, action_type):
            primary_pol = source.get("primary_pollutants", ["PM2.5"])[0]
            rate_map    = source.get("emission_rate_kg_hr", {})
            measured    = float(
                rate_map.get(primary_pol, 50.0) if isinstance(rate_map, dict) else 50.0
            )
            permitted   = measured * 0.5  # demo: assume limit = 50% of observed
            _call(issue_regulatory_action,
                source_id              = source_id,
                source_name            = source.get("source_name", "Unknown Source"),
                violation_type         = source.get("compliance_status", "exceedance").replace(" ", "_"),
                pollutant              = primary_pol,
                measured_concentration = measured,
                permitted_limit        = permitted,
                action_type            = action_type,
                required_action        = (
                    "Immediate cessation of non-compliant emission activities.  "
                    "Submit hourly compliance reports until normalised."
                    if emergency else
                    "Reduce emission rates to permitted limits within 24 hours.  "
                    "Submit compliance confirmation within 48 hours."
                ),
                compliance_deadline    = datetime.now().strftime("%Y-%m-%d") + "T23:59:00",
                enforcement_authority  = "State Air Quality Management District",
                regulatory_basis       = "Clean Air Act Sec. 113; State AQMD Rule 1001",
            )
            SESSION_MEMORY.record_alert(
                "regulatory", action_type,
                source_id   = source_id,
                action_type = action_type,
            )
        else:
            print(f"  [Memory gate] SUPPRESS regulatory action for {source_id} — "
                  f"{action_type} already issued or higher tier was used.")

    # 3d. Hospital alert — MEMORY GATED
    if max_aqi > 100:
        hosp_level   = "critical" if emergency else "high" if max_aqi > 150 else "elevated"
        vol_increase = 45.0 if emergency else 20.0 if max_aqi > 150 else 10.0
        if SESSION_MEMORY.can_issue_alert("hospital", hosp_level, "network-wide"):
            _call(notify_hospital_network,
                alert_level                  = hosp_level,
                affected_districts           = critical_districts or ["Downtown Core"],
                primary_pollutant            = dominant_pol,
                aqi                          = max_aqi,
                expected_case_types          = [
                    "asthma exacerbation", "COPD exacerbation",
                    "chest pain", "respiratory distress",
                ],
                expected_volume_increase_pct = vol_increase,
                special_instructions         = (
                    "Pre-position nebulisers, bronchodilators, supplemental O2.  "
                    "Activate respiratory triage protocol."
                    + (" Activate emergency surge capacity." if emergency else "")
                ),
            )
            SESSION_MEMORY.record_alert("hospital", hosp_level, "network-wide")
        else:
            print(f"  [Memory gate] SUPPRESS hospital alert — {hosp_level} already issued "
                  "or higher severity was dispatched this session.")

    # 3e. Traffic restriction (no deduplication needed — re-issuing is harmless)
    if vc < 6000 and max_aqi > 100:
        _call(request_traffic_restriction,
            restriction_type = "hgv_ban",
            affected_zones   = critical_districts or ["Downtown Core", "East Industrial District"],
            vehicles_affected= "Heavy goods vehicles (>3.5t), pre-Euro 5 diesel",
            start_time       = "06:00",
            end_time         = "22:00",
            reason           = f"Air quality episode: AQI {max_aqi}, VC={vc} m²/s.",
            legal_basis      = "Road Traffic Regulation Act s.1(1) — Air Quality Emergency",
        )

    # ── Build state and update memory ─────────────────────────────────────────
    state = _extract_state(state, tool_log)
    state["situation_report"] = _build_demo_report(
        state, max_aqi, avg_aqi, dominant_pol,
        non_compliant, met, sat, tool_log, scenario,
    )

    _post_run_memory_update(state, tool_log)
    print(f"\n  [DEMO] Complete — {len(tool_log)} tool calls")
    _print_summary(state, len(tool_log))
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Post-run memory update (called after both live and demo runs)
# ══════════════════════════════════════════════════════════════════════════════

def _post_run_memory_update(state: dict, tool_log: list) -> None:
    """
    Persist this run's observations and actions into SESSION_MEMORY.

    Must be called ONCE at the end of each run.
    Handles deferred action tracking and (for the live path) reconstructs
    alert records from the tool log so memory.can_issue_alert() works next run.
    """
    SESSION_MEMORY.record_run(state)

    for entry in tool_log:
        name   = entry["tool"]
        result = entry.get("result", {})
        args   = entry.get("args", {})
        if not isinstance(result, dict):
            continue

        # Deferred actions (both paths)
        if name == "log_mitigation_recommendation" and result.get("recommendation_id"):
            SESSION_MEMORY.record_deferred_action(
                recommendation_id = result["recommendation_id"],
                title             = args.get("title", ""),
                target_entity     = args.get("target_entity", ""),
                category          = args.get("category", ""),
                priority          = args.get("priority", ""),
            )

        # Live path: reconstruct alert records from tool log
        # (demo path already called SESSION_MEMORY.record_alert() inline)
        # We check for alert_id / case_number / notification_id to confirm success
        elif name == "issue_public_health_alert" and result.get("alert_id"):
            dist     = args.get("affected_districts") or ["network-wide"]
            dist_str = dist[0] if isinstance(dist, list) else str(dist)
            # Only record if not already recorded (demo path records inline)
            already = any(
                a.alert_type == "public_health"
                and a.district == "network-wide"
                and a.severity == args.get("severity", "advisory")
                for a in SESSION_MEMORY.issued_alerts
                # within the current run (last 30 seconds)
                if a.issued_at >= state.get("analysis_timestamp", "")[:16]
            )
            if not already:
                SESSION_MEMORY.record_alert(
                    "public_health",
                    args.get("severity", "advisory"),
                    district = dist_str,
                )

        elif name == "issue_regulatory_action" and result.get("case_number"):
            src_id      = args.get("source_id")
            action_type = args.get("action_type", "notice_of_violation")
            already = any(
                a.alert_type == "regulatory" and a.source_id == src_id
                and a.action_type == action_type
                for a in SESSION_MEMORY.issued_alerts
            )
            if not already:
                SESSION_MEMORY.record_alert(
                    "regulatory", action_type,
                    source_id   = src_id,
                    action_type = action_type,
                )

        elif name == "notify_hospital_network" and result.get("notification_id"):
            hosp_level = args.get("alert_level", "elevated")
            already = any(
                a.alert_type == "hospital"
                and a.district == "network-wide"
                and a.severity == hosp_level
                for a in SESSION_MEMORY.issued_alerts
            )
            if not already:
                SESSION_MEMORY.record_alert("hospital", hosp_level, "network-wide")


# ══════════════════════════════════════════════════════════════════════════════
# State extraction (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_state(state: dict, tool_log: list) -> dict:
    """Reconstruct the structured agent state from the flat tool results log."""
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
                critical = [r for r in readings
                            if r["aqi_category"] in ("unhealthy", "very_unhealthy", "hazardous")]
                state["risk_scores"].append({
                    "domain":                 "ground_sensors",
                    "max_aqi":                max_aqi,
                    "avg_aqi":                avg_aqi,
                    "level":                  critical[0]["aqi_category"] if critical else "moderate",
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
                        "hazardous"      if max_aod > 0.8 else
                        "very_unhealthy" if max_aod > 0.6 else
                        "unhealthy"      if max_aod > 0.4 else "moderate"
                    ),
                })

        elif name == "fetch_meteorological_data":
            state["meteorological_summary"] = result

        elif name == "fetch_emission_inventory":
            state["pollution_sources"] = result.get("sources", [])

        elif name in ("issue_public_health_alert", "notify_hospital_network"):
            state["public_alerts"].append(result)

        elif name == "issue_regulatory_action":
            state["regulatory_actions"].append(result)

        elif name == "log_mitigation_recommendation":
            state["mitigation_recommendations"].append({
                **entry["args"],
                "recommendation_id":      result.get("recommendation_id", "REC-DEMO"),
                "expected_aqi_reduction": entry["args"].get("expected_aqi_reduction", 0.0),
            })

    readings = state.get("ground_readings", [])
    if readings:
        max_aqi = max(r["aqi"] for r in readings)
        avg_aqi = round(sum(r["aqi"] for r in readings) / len(readings), 1)
        state["health_impact"] = {
            "max_aqi":              max_aqi,
            "avg_aqi":              avg_aqi,
            "max_pm25_ug_m3":       max((r.get("pm25_ug_m3") or 0) for r in readings),
            "hospital_alert_level": (
                "critical" if max_aqi > 200 else
                "high"     if max_aqi > 150 else
                "elevated" if max_aqi > 100 else "normal"
            ),
            "emergency_response":   state.get("emergency_triggered", False),
        }

    return state


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_invoke(tool, args: dict) -> dict:
    """Invoke a tool with error handling.  Returns a dict in all cases."""
    try:
        fn = getattr(tool, "func", tool)
        if hasattr(tool, "invoke"):
            return tool.invoke(args) if args else tool.invoke({})
        return fn(**args) if args else fn()
    except Exception as ex:
        return {"error": str(ex)}


def _make_empty_state(scenario: str) -> dict:
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
    if not args:
        return ""
    short = {
        k: (v if len(str(v)) < 40 else str(v)[:37] + "...")
        for k, v in args.items()
    }
    return ", ".join(f"{k}={repr(v)}" for k, v in list(short.items())[:3])


def _one_liner(tool_name: str, compressed: str) -> str:
    """Extract a concise one-liner from a compressed tool result string."""
    try:
        d = json.loads(compressed)
        if "network_max_aqi"          in d:
            return (f"max_aqi={d['network_max_aqi']}, avg={d.get('network_avg_aqi')}, "
                    f"critical={d.get('critical_stations', [])}")
        if "max_aerosol_optical_depth" in d:
            return (f"max_aod={d['max_aerosol_optical_depth']}, "
                    f"fires={d.get('total_active_fire_count')}, "
                    f"plumes={d.get('plume_detections')}")
        if "ventilation_coefficient_m2_s" in d:
            return (f"VC={d['ventilation_coefficient_m2_s']} m²/s, "
                    f"dispersion={d.get('dispersion_quality')}")
        if "non_compliant_count"       in d:
            return (f"sources={d.get('total_sources')}, "
                    f"non_compliant={d.get('non_compliant_count')}")
        if "total_population"          in d:
            return (f"pop={d.get('total_population')}, "
                    f"sensitive={d.get('total_sensitive_population')}")
    except Exception:
        pass
    return compressed[:120]


def _build_demo_report(state, max_aqi, avg_aqi, dominant_pol,
                       non_compliant, met, sat, tool_log, scenario) -> str:
    """Build a structured situation report for demo mode."""
    n_rec   = sum(1 for e in tool_log if e["tool"] == "log_mitigation_recommendation")
    n_alert = sum(1 for e in tool_log if e["tool"] == "issue_public_health_alert")
    n_reg   = sum(1 for e in tool_log if e["tool"] == "issue_regulatory_action")
    n_hosp  = sum(1 for e in tool_log if e["tool"] == "notify_hospital_network")

    risk = (
        "CRITICAL" if max_aqi > 300 else "HIGH"     if max_aqi > 200 else
        "ELEVATED" if max_aqi > 150 else "MODERATE" if max_aqi > 100 else "LOW"
    )

    trend      = SESSION_MEMORY.get_aqi_trend()
    trend_note = ""
    if trend.get("available") and trend.get("previous_max_aqi") is not None:
        trend_note = (
            f"\n  AQI trend (memory): {trend['previous_max_aqi']} → {max_aqi} "
            f"({trend['direction']}, {trend['delta_aqi']:+d} pts)"
        )

    aqi_forecast = (
        max(50,  max_aqi - 20)
        if met.get("dispersion_quality") not in ("very_poor", "poor")
        else min(500, max_aqi + 30)
    )
    ah = SESSION_MEMORY.get_alert_history_summary()

    # run_count is incremented inside record_run(), called from _post_run_memory_update()
    # which is called AFTER this function.  So at this point run_count is still the
    # PREVIOUS value.  Add 1 for the display label.
    run_label = SESSION_MEMORY.run_count + 1

    return (
        f"[DEMO MODE — set OPENAI_API_KEY in .env for live LLM reasoning]\n\n"
        f"SITUATION REPORT — {air_config.city} Air Quality Monitoring\n"
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Scenario: {scenario.upper()} | Run #{run_label}\n\n"
        f"SECTION 1 — EXECUTIVE SUMMARY\n"
        f"  Max AQI {max_aqi} (avg {avg_aqi}) | dominant: {dominant_pol} | "
        f"{len(non_compliant)} non-compliant source(s) | "
        f"{len(tool_log)} ReAct tool calls{trend_note}\n\n"
        f"SECTION 2 — CURRENT AIR QUALITY\n"
        f"  Max AQI: {max_aqi} | Avg AQI: {avg_aqi} | Dominant: {dominant_pol}\n"
        f"  Dispersion: {met.get('dispersion_quality','N/A').upper()} "
        f"(VC={met.get('ventilation_coefficient_m2_s','N/A')} m²/s)\n"
        f"  Satellite AOD: {sat.get('max_aerosol_optical_depth','N/A')} | "
        f"Active fires: {sat.get('total_active_fire_count', 0)}\n\n"
        f"SECTION 3 — SOURCE ATTRIBUTION\n"
        f"  {len(non_compliant)} non-compliant source(s) in inventory.\n\n"
        f"SECTION 4 — HEALTH IMPACT\n"
        f"  Hospital alert: "
        f"{'CRITICAL' if max_aqi > 200 else 'HIGH' if max_aqi > 150 else 'ELEVATED'}\n\n"
        f"SECTION 5 — METEOROLOGICAL\n"
        f"  {met.get('forecast_12h', 'Forecast unavailable')}\n\n"
        f"SECTION 6 — ACTIONS THIS RUN\n"
        f"  Mitigations logged   : {n_rec}\n"
        f"  Public alerts issued : {n_alert}  (session total: {ah['public_alerts_issued']})\n"
        f"  Regulatory actions   : {n_reg}  (session total: {ah['regulatory_actions_issued']})\n"
        f"  Hospital alerts      : {n_hosp}  (session total: {ah['hospital_alerts_issued']})\n\n"
        f"SECTION 7 — OUTSTANDING ACTIONS\n"
        f"  Deferred pending: {ah['deferred_actions_pending']}\n"
        + (f"  *** EPISODE DECLARED — peak AQI {SESSION_MEMORY.episode_peak_aqi}\n"
           if SESSION_MEMORY.episode_declared else "")
        + f"\nOVERALL RISK: {risk}\n"
        f"AQI FORECAST (6h): {aqi_forecast}\n\n"
        f"Disclaimer: {air_config.disclaimer}"
    )


def _print_summary(state: dict, iterations: int) -> None:
    readings = state.get("ground_readings", [])
    max_aqi  = max((r["aqi"] for r in readings), default=0)
    risk = (
        "CRITICAL"  if max_aqi > 300 else "VERY HIGH" if max_aqi > 200 else
        "HIGH"      if max_aqi > 150 else "ELEVATED"  if max_aqi > 100 else "MODERATE"
    )
    ah = SESSION_MEMORY.get_alert_history_summary()
    print("\n" + "=" * 70)
    print("  REACT RUN COMPLETE")
    print(f"  Risk level         : {risk}")
    print(f"  Tool calls         : {iterations}")
    print(f"  Session run #      : {SESSION_MEMORY.run_count}")
    print(f"  Ground stations    : {len(readings)}")
    print(f"  Satellite obs      : {len(state.get('satellite_observations', []))}")
    print(f"  Sources identified : {len(state.get('pollution_sources', []))}")
    print(f"  Mitigations (run)  : {len(state.get('mitigation_recommendations', []))}")
    print(f"  Public alerts (run): {len(state.get('public_alerts', []))} | "
          f"session total: {ah['public_alerts_issued']}")
    print(f"  Regulatory (run)   : {len(state.get('regulatory_actions', []))} | "
          f"session total: {ah['regulatory_actions_issued']}")
    print(f"  Emergency flag     : {state.get('emergency_triggered', False)}")
    print(f"  Episode declared   : {SESSION_MEMORY.episode_declared}")
    print(f"  Deferred pending   : {ah['deferred_actions_pending']}")
    print("=" * 70)
