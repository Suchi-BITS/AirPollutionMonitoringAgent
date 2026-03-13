# context/context_engineer.py  — AirGuard v3
#
# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
#
# An LLM's context window is finite and expensive.  Sending everything raw
# (8 stations × 20 fields, 3 satellites × 15 fields, 7 inventory sources,
# plus the full growing message history) creates three problems:
#
#   1. NOISE
#      The LLM wastes attention on irrelevant fields like permit_number,
#      last_inspection_date, or pressure_hpa when deciding alert severity.
#
#   2. STALENESS
#      By iteration 10 the LLM is re-reading its own AQI reasoning from
#      8 tool calls ago.  That context is dead weight and can confuse
#      current decisions.
#
#   3. LOST EMPHASIS
#      AQI=500 and SO2 at 22× the limit can be buried in 3 000 raw-JSON
#      tokens.  The LLM may not weight the signal appropriately.
#
# Context engineering solves all three:
#
#   A. COMPRESS tool results before they enter the message history.
#      Raw tool payloads are 600-2500 tokens each.
#      Compressed versions are 80-400 tokens — an ~80% reduction —
#      while preserving every safety-critical number.
#
#   B. BUILD the system prompt dynamically each run.
#      Injects the current memory state (AQI trend, alert history,
#      episode metadata) and severity-calibrated response guidance.
#
#   C. TRIM the message history when it exceeds a token budget.
#      Collapses stale tool-exchange blocks into a one-line summary.
#      The LLM always sees its most recent reasoning verbatim.
#
# ══════════════════════════════════════════════════════════════════════════════
# COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
#
#   ContextEngineer
#     ├── build_system_prompt(scenario, memory_summary, max_aqi_hint) → str
#     │     Assembles the per-run system prompt:
#     │       static domain instructions
#     │       + live SESSION MEMORY block
#     │       + severity-calibrated response guidance
#     │
#     ├── compress_tool_result(tool_name, result) → str
#     │     Tool-specific compressors: raw dict → compact JSON string.
#     │     Keeps only decision-relevant fields; discards metadata noise.
#     │
#     ├── trim_message_history(messages) → List[messages]
#     │     Keeps SystemMessage + last MESSAGE_HISTORY_LIMIT messages.
#     │     Collapses older tool exchanges into a single summary ToolMessage.
#     │
#     └── build_context_for_run(scenario, memory) → dict
#           Entry point called once per run by react_agent.py.
#           Returns: {system_prompt, initial_human_message, token_estimate}

import json
from typing import List, Any, Optional
from datetime import datetime


class ContextEngineer:
    """
    Manages what enters the LLM context window at each ReAct iteration.
    One instance (CONTEXT_ENGINEER) is shared across all runs.
    """

    # Max verbatim messages to keep in history before trimming
    MESSAGE_HISTORY_LIMIT = 14

    # Station fields kept in compressed ground-sensor output
    _STATION_KEEP = {
        "station_id", "district", "station_type",
        "aqi", "aqi_category", "dominant_pollutant",
        "pm25_ug_m3", "so2_ug_m3", "no2_ug_m3", "o3_ug_m3",
        "data_quality_flag",
    }

    # Source fields kept in compressed inventory output
    _SOURCE_KEEP = {
        "source_id", "source_name", "source_category", "district",
        "primary_pollutants", "compliance_status", "operating_status",
        "emission_rate_kg_hr",
    }

    def __init__(self):
        self._token_estimate = 0

    # ── System prompt builder ─────────────────────────────────────────────────

    def build_system_prompt(self, scenario: str,
                            memory_summary: str,
                            max_aqi_hint: Optional[int] = None) -> str:
        """
        Build the per-run system prompt.

        Args:
            scenario:       'standard' | 'episode'
            memory_summary: ShortTermMemory.context_summary() text block
            max_aqi_hint:   last known max AQI — selects severity guidance tier
        """
        from config.settings import air_config

        if max_aqi_hint is not None and max_aqi_hint > 300:
            severity_block = _EMERGENCY_GUIDANCE
        elif max_aqi_hint is not None and max_aqi_hint > 150:
            severity_block = _HIGH_GUIDANCE
        else:
            severity_block = _STANDARD_GUIDANCE

        return (
            f"You are the AirGuard ReAct Agent — air quality monitoring and response "
            f"system for {air_config.city}, {air_config.region}.\n\n"
            f"You operate in a continuous monitoring loop.  Each run you must:\n"
            f"  1. Gather current data (observation tools)\n"
            f"  2. Reason about what changed since last run (use SESSION MEMORY below)\n"
            f"  3. Issue only NEW or ESCALATED actions — never duplicate prior alerts\n"
            f"  4. Write a focused situation report\n\n"
            f"══════════════════════════════════════════════\n"
            f"{memory_summary}\n"
            f"══════════════════════════════════════════════\n\n"
            f"OBSERVATION TOOLS (call all five every run):\n"
            f"  fetch_ground_sensor_data(scenario)   → AQI, PM2.5, SO2, NO2 per station\n"
            f"  fetch_satellite_imagery(scenario)    → AOD, fire count, plume detection\n"
            f"  fetch_meteorological_data(scenario)  → wind, mixing height, VC, stability\n"
            f"  fetch_emission_inventory()           → registered sources, compliance\n"
            f"  fetch_health_risk_tables()           → population, vulnerable groups\n\n"
            f"ACTION TOOLS:\n"
            f"  log_mitigation_recommendation(priority, category, target_entity, title,\n"
            f"      description, expected_aqi_reduction, implementation_timeline,\n"
            f"      regulatory_basis, estimated_cost_tier, co_benefits)\n"
            f"    priority: 'emergency'|'high'|'medium'|'low'\n\n"
            f"  issue_public_health_alert(severity, affected_districts, aqi_level,\n"
            f"      dominant_pollutant, health_message, recommended_actions,\n"
            f"      sensitive_groups_warning, duration_hours, channels)\n"
            f"    severity: 'advisory'(101-150)|'warning'(151-200)|'alert'(201-300)|'emergency'(>300)\n\n"
            f"  issue_regulatory_action(source_id, source_name, violation_type,\n"
            f"      pollutant, measured_concentration, permitted_limit,\n"
            f"      action_type, required_action, compliance_deadline,\n"
            f"      enforcement_authority, regulatory_basis)\n"
            f"    action_type: 'notice_of_violation'|'compliance_order'|'emergency_shutdown_order'\n\n"
            f"  notify_hospital_network(alert_level, affected_districts, primary_pollutant,\n"
            f"      aqi, expected_case_types, expected_volume_increase_pct, special_instructions)\n"
            f"    alert_level: 'normal'|'elevated'|'high'|'critical'\n\n"
            f"  request_traffic_restriction(restriction_type, affected_zones,\n"
            f"      vehicles_affected, start_time, end_time, reason, legal_basis)\n\n"
            f"  get_action_log(limit)   → review actions already logged this run\n\n"
            f"{severity_block}\n\n"
            f"STOPPING CRITERION — stop calling tools when ALL are met:\n"
            f"  ✓ All 5 observation tools called at least once\n"
            f"  ✓ All NEW or ESCALATED actions issued\n"
            f"  ✓ At least 1 mitigation recommendation logged\n\n"
            f"Disclaimer: {air_config.disclaimer}"
        )

    # ── Tool result compressor ────────────────────────────────────────────────

    def compress_tool_result(self, tool_name: str, result: Any) -> str:
        """
        Convert a raw tool result into a compact, high-signal string.
        Preserves all safety-critical numbers; discards metadata noise.

        Token savings:
          fetch_ground_sensor_data   ~2500 → ~300  (88%)
          fetch_satellite_imagery    ~1800 → ~250  (86%)
          fetch_meteorological_data   ~700 → ~200  (71%)
          fetch_emission_inventory   ~2000 → ~400  (80%)
          fetch_health_risk_tables   ~1200 → ~300  (75%)
          action tools               ~150  → ~150  (unchanged — already short)
        """
        if not isinstance(result, dict):
            return str(result)[:600]

        dispatch = {
            "fetch_ground_sensor_data":  self._compress_ground,
            "fetch_satellite_imagery":   self._compress_satellite,
            "fetch_meteorological_data": self._compress_met,
            "fetch_emission_inventory":  self._compress_inventory,
            "fetch_health_risk_tables":  self._compress_health,
        }
        fn = dispatch.get(tool_name)
        return fn(result) if fn else json.dumps(result, default=str)

    def _compress_ground(self, r: dict) -> str:
        stations = r.get("stations", [])
        slim = [
            {k: v for k, v in s.items() if k in self._STATION_KEEP}
            for s in sorted(stations, key=lambda x: -x.get("aqi", 0))
        ]
        return json.dumps({
            "network_max_aqi":   r.get("network_max_aqi"),
            "network_avg_aqi":   r.get("network_avg_aqi"),
            "critical_stations": r.get("critical_stations", []),
            "moderate_stations": r.get("moderate_stations", []),
            "stations":          slim,
            "scenario":          r.get("scenario_description", ""),
        }, default=str)

    def _compress_satellite(self, r: dict) -> str:
        obs = r.get("observations", [])
        slim_obs = [
            {
                "satellite":                o.get("satellite"),
                "aerosol_optical_depth":    o.get("aerosol_optical_depth"),
                "active_fire_count":        o.get("active_fire_count"),
                "plume_detected":           o.get("plume_detected"),
                "plume_origin_description": o.get("plume_origin_description"),
                "pollution_hotspots": [
                    {k: v for k, v in h.items()
                     if k in ("source_name", "intensity_index", "primary_pollutant")}
                    for h in (o.get("pollution_hotspots") or [])
                ],
            }
            for o in obs
        ]
        return json.dumps({
            "max_aerosol_optical_depth": r.get("max_aerosol_optical_depth"),
            "total_active_fire_count":   r.get("total_active_fire_count"),
            "plume_detections":          r.get("plume_detections"),
            "plume_sources":             r.get("plume_sources"),
            "agricultural_burn_alert":   r.get("agricultural_burn_alert"),
            "unique_hotspots":           r.get("unique_hotspots", []),
            "observations":              slim_obs,
        }, default=str)

    def _compress_met(self, r: dict) -> str:
        keep = {
            "wind_speed_ms", "wind_direction_deg", "wind_direction_label",
            "mixing_height_m", "stability_class", "stability_label",
            "ventilation_coefficient_m2_s", "dispersion_quality",
            "pollution_accumulation_risk", "visibility_km",
            "forecast_12h", "temperature_c",
        }
        return json.dumps({k: v for k, v in r.items() if k in keep}, default=str)

    def _compress_inventory(self, r: dict) -> str:
        sources = r.get("sources", [])
        slim = [
            {k: v for k, v in s.items() if k in self._SOURCE_KEEP}
            for s in sources
        ]
        # non-compliant first
        slim.sort(key=lambda s: 0 if s.get("compliance_status") != "compliant" else 1)
        return json.dumps({
            "total_sources":          r.get("total_sources"),
            "non_compliant_count":    r.get("non_compliant_count"),
            "non_compliant_sources":  r.get("non_compliant_sources", []),
            "sources":                slim,
        }, default=str)

    def _compress_health(self, r: dict) -> str:
        districts = r.get("districts", {})
        ranked = sorted(
            districts.items(),
            key=lambda kv: kv[1].get("sensitive_population", 0),
            reverse=True,
        )
        total_pop       = sum(d.get("population", 0)           for d in districts.values())
        total_sensitive = sum(d.get("sensitive_population", 0) for d in districts.values())
        top3 = [
            {
                "district":       name,
                "population":     data["population"],
                "sensitive":      data["sensitive_population"],
                "hospitals":      data["hospitals_nearby"],
                "vulnerable_pct": data["vulnerable_pct"],
            }
            for name, data in ranked[:3]
        ]
        cr = r.get("concentration_response_coefficients", {})
        return json.dumps({
            "total_population":                     total_pop,
            "total_sensitive_population":           total_sensitive,
            "top_3_vulnerable_districts":           top3,
            "c_r_pm25_respiratory_per_10ugm3":      cr.get("respiratory_hospitalizations_per_10ugm3_pm25"),
            "c_r_pm25_cardiovascular_per_10ugm3":   cr.get("cardiovascular_hospitalizations_per_10ugm3_pm25"),
            "c_r_pm25_mortality_per_10ugm3":        cr.get("all_cause_mortality_per_10ugm3_pm25"),
        }, default=str)

    # ── Message history trimmer ───────────────────────────────────────────────

    def trim_message_history(self, messages: List[Any]) -> List[Any]:
        """
        Keep message history within a manageable token budget.

        Strategy:
          Keep: SystemMessage(s) + last MESSAGE_HISTORY_LIMIT non-system messages
          Collapse: older messages into a single compact summary ToolMessage

        This ensures the LLM always has its current reasoning verbatim while
        stale intermediate observations are summarised and shed.
        """
        if len(messages) <= self.MESSAGE_HISTORY_LIMIT + 1:
            return messages

        try:
            from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
        except ImportError:
            return messages     # cannot trim without LangChain

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system  = [m for m in messages if not isinstance(m, SystemMessage)]

        cutoff      = max(0, len(non_system) - self.MESSAGE_HISTORY_LIMIT)
        to_collapse = non_system[:cutoff]
        to_keep     = non_system[cutoff:]

        if not to_collapse:
            return messages

        # Build compact summary of collapsed block
        lines = ["[CONTEXT TRIM — earlier observations collapsed to summary]"]
        for m in to_collapse:
            if isinstance(m, ToolMessage):
                try:
                    data = json.loads(m.content)
                    lines.append(f"  {self._one_liner(data)}")
                except Exception:
                    lines.append(f"  tool: {str(m.content)[:100]}")
            elif isinstance(m, AIMessage) and m.content:
                lines.append(f"  agent: {str(m.content)[:100]}...")

        summary_msg = ToolMessage(
            content     = "\n".join(lines),
            tool_call_id = "context-trim-summary",
        )

        self._token_estimate = len(str(system_msgs + [summary_msg] + to_keep)) // 4
        return system_msgs + [summary_msg] + to_keep

    def _one_liner(self, data: dict) -> str:
        if "network_max_aqi" in data:
            return f"ground: max_aqi={data['network_max_aqi']}, avg={data.get('network_avg_aqi')}"
        if "max_aerosol_optical_depth" in data:
            return f"satellite: aod={data['max_aerosol_optical_depth']}, fires={data.get('total_active_fire_count')}"
        if "ventilation_coefficient_m2_s" in data:
            return f"met: VC={data['ventilation_coefficient_m2_s']} m²/s, disp={data.get('dispersion_quality')}"
        if "non_compliant_count" in data:
            return f"inventory: non_compliant={data['non_compliant_count']}/{data.get('total_sources')}"
        if "total_population" in data:
            return f"health: pop={data['total_population']}, sensitive={data.get('total_sensitive_population')}"
        return str(data)[:100]

    # ── Full context builder ──────────────────────────────────────────────────

    def build_context_for_run(self, scenario: str, memory) -> dict:
        """
        Entry point called once per run by react_agent.py before the ReAct loop.

        Returns:
            {
              "system_prompt":         str   — dynamic, includes live memory state
              "initial_human_message": str   — task description for this run
              "token_estimate":        int   — rough token count of system_prompt
            }
        """
        memory_summary = memory.context_summary()
        trend          = memory.get_aqi_trend()
        max_aqi_hint   = trend.get("current_max_aqi") if trend.get("available") else None

        system_prompt = self.build_system_prompt(
            scenario       = scenario,
            memory_summary = memory_summary,
            max_aqi_hint   = max_aqi_hint,
        )

        run_num = memory.run_count + 1
        if max_aqi_hint and max_aqi_hint > 200:
            opening = (
                f"Run #{run_num} — scenario='{scenario}'.  "
                f"Last run: max AQI {max_aqi_hint} (EMERGENCY).  "
                f"Re-fetch ground sensors first to confirm current status, "
                f"then update actions as needed."
            )
        elif max_aqi_hint:
            opening = (
                f"Run #{run_num} — scenario='{scenario}'.  "
                f"Last run: max AQI {max_aqi_hint}.  "
                f"Fetch updated readings and adjust response to any changes."
            )
        else:
            opening = (
                f"Run #{run_num} — scenario='{scenario}'.  "
                f"Begin baseline assessment: fetch all observation data, "
                f"issue appropriate actions, write situation report."
            )

        self._token_estimate = len(system_prompt) // 4
        return {
            "system_prompt":         system_prompt,
            "initial_human_message": opening,
            "token_estimate":        self._token_estimate,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Severity-calibrated response guidance blocks
# Injected verbatim into the system prompt based on last-known max AQI
# ─────────────────────────────────────────────────────────────────────────────

_STANDARD_GUIDANCE = """\
RESPONSE CALIBRATION — STANDARD (AQI ≤ 150):
  → Issue ADVISORY alert if AQI > 100 and no advisory issued yet this session
  → Issue NOTICE_OF_VIOLATION for each non-compliant source not yet actioned
  → Log at least 1 mitigation recommendation
  → Hospital alert: ELEVATED if AQI 101-150, else NORMAL
  → Traffic restriction: HGV_BAN if ventilation coefficient < 6000 m²/s"""

_HIGH_GUIDANCE = """\
RESPONSE CALIBRATION — HIGH (AQI 151-300):
  → Issue WARNING or ALERT public alert (escalate from advisory if already issued)
  → Issue COMPLIANCE_ORDER for non-compliant sources (escalate from NOV if issued)
  → Notify hospital network at HIGH level
  → Log LEZ + HGV_BAN traffic restrictions
  → Log voluntary industrial curtailment recommendation
  → If 3+ consecutive runs above 150: flag sustained episode in report"""

_EMERGENCY_GUIDANCE = """\
RESPONSE CALIBRATION — EMERGENCY (AQI > 300):  *** CRITICAL ***
  → Issue EMERGENCY public alert (escalate from any prior lower severity)
  → Issue EMERGENCY_SHUTDOWN_ORDER for all non-compliant sources
  → Notify hospital network at CRITICAL level — include surge capacity instruction
  → Request all available traffic restrictions
  → Log Stage III Emergency Episode Plan mitigation
  → If episode already declared: report episode duration + whether peak is climbing"""


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────
CONTEXT_ENGINEER = ContextEngineer()
