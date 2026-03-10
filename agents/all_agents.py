# agents/all_agents.py
#
# All agent node functions for the Air Pollution Monitoring LangGraph pipeline.
#
# AGENT GRAPH TOPOLOGY:
#
#   supervisor_init
#       |
#       +-- ground_sensor_agent      (reads all ground station data, computes network AQI status)
#       |
#       +-- satellite_agent          (processes satellite imagery, identifies hotspots and plumes)
#       |
#       +-- meteorological_agent     (analyzes dispersion conditions, wind trajectories)
#       |
#       +-- source_identification_agent  (cross-references sensors + satellite + inventory)
#       |
#       +-- health_impact_agent      (population exposure, clinical risk, hospital preparedness)
#       |
#       +-- mitigation_agent         (generates prioritized action plan, calls action tools)
#       |
#       +-- alert_agent              (issues public alerts and regulatory notifications)
#       |
#   supervisor_synthesis             (compiles situation report, final risk score)
#
# Each agent reads the state dict, calls tools or LLM, and returns an updated state dict.
# All agents are pure functions: (state: dict) -> dict

import json
from datetime import datetime

from agents.base import call_llm, call_llm_with_tools
from config.settings import air_config
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


# ---------------------------------------------------------------------------
# Agent 1: Supervisor — initialization
# ---------------------------------------------------------------------------

def supervisor_init_agent(state: dict) -> dict:
    """
    Opens the analysis cycle. Sets timestamp, logs the target region,
    and prints the session header. No LLM call required.
    """
    print("\n" + "=" * 70)
    print(f"  {air_config.system_name}")
    print(f"  {air_config.city} | {air_config.region}")
    print(f"  Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    scenario = state.get("target_region", "standard")
    print(f"  Scenario: {scenario.upper()}")
    print("=" * 70)

    state["analysis_timestamp"] = datetime.now().isoformat()
    state["iteration_count"]    = state.get("iteration_count", 0) + 1
    state["current_agent"]      = "ground_sensor_agent"
    return state


# ---------------------------------------------------------------------------
# Agent 2: Ground Sensor Agent
# ---------------------------------------------------------------------------

GROUND_SENSOR_SYSTEM = """You are the Ground Sensor Analysis Agent for an urban air quality
monitoring network. Your role is to:

1. Ingest real-time readings from all ground monitoring stations in the network.
2. Identify stations that are exceeding WHO guidelines or EPA regulatory limits.
3. Analyze spatial patterns — which districts are worst affected, which are clean.
4. Identify the dominant pollutants driving AQI exceedances at each station.
5. Compare current readings to station type norms (traffic, industrial, background).
6. Flag stations with data quality issues.

Output a structured analysis with:
- Network-wide AQI status (overall risk level)
- Station-by-station summary of exceedances
- Spatial pattern assessment (downwind clustering, district-level gradients)
- Priority stations requiring immediate attention
- Comparison against WHO AQI guidelines:
    PM2.5 24h limit: 15 ug/m3  |  PM10 24h limit: 45 ug/m3
    NO2 1h limit: 200 ug/m3    |  SO2 24h limit: 40 ug/m3
    CO 8h limit: 4 mg/m3       |  O3 8h limit: 100 ug/m3

Be precise with numbers. Identify which stations are in which AQI category.
Do not hedge — if a station is at hazardous levels, say so explicitly."""

def ground_sensor_agent(state: dict) -> dict:
    """
    Fetches all ground station readings and produces a network-wide analysis.
    """
    print("\n[GROUND SENSOR AGENT] Fetching ground station data...")
    scenario = state.get("target_region", "standard")

    sensor_data, tool_results = call_llm_with_tools(
        system_prompt=GROUND_SENSOR_SYSTEM,
        user_message=(
            f"Fetch and analyze all ground station data for scenario: {scenario}. "
            f"SCENARIO={scenario}. "
            "Use the fetch_ground_sensor_data tool to retrieve current readings, "
            "then produce a complete network status analysis."
        ),
        tools=[fetch_ground_sensor_data],
    )

    # Extract raw readings from tool results for downstream agents
    raw_readings = []
    for tr in tool_results:
        if tr.get("tool") == "fetch_ground_sensor_data" and isinstance(tr.get("result"), dict):
            raw_readings = tr["result"].get("stations", [])
            break

    # Fallback: call directly when demo mode returns no tool results
    if not raw_readings:
        try:
            result = fetch_ground_sensor_data.invoke({"scenario": scenario})
            if isinstance(result, dict):
                raw_readings = result.get("stations", [])
        except Exception:
            pass

    # Compute summary statistics for downstream agents
    if raw_readings:
        max_aqi   = max(r["aqi"] for r in raw_readings)
        avg_aqi   = round(sum(r["aqi"] for r in raw_readings) / len(raw_readings), 1)
        critical  = [r for r in raw_readings if r["aqi_category"] in ("unhealthy", "very_unhealthy", "hazardous")]
        dom_pols  = {}
        for r in raw_readings:
            dp = r.get("dominant_pollutant", "Unknown")
            dom_pols[dp] = dom_pols.get(dp, 0) + 1
        overall_dominant = max(dom_pols, key=dom_pols.get) if dom_pols else "Unknown"

        print(f"  Network AQI: max={max_aqi}, avg={avg_aqi}, critical stations={len(critical)}")

        state["ground_readings"]  = raw_readings
        state["ground_analysis"]  = sensor_data
        state["risk_scores"].append({
            "domain":   "ground_sensors",
            "max_aqi":  max_aqi,
            "avg_aqi":  avg_aqi,
            "level":    critical[0]["aqi_category"] if critical else raw_readings[0]["aqi_category"],
            "dominant_pollutant": overall_dominant,
            "critical_station_count": len(critical),
        })
        if max_aqi >= air_config.aqi_very_unhealthy:
            state["emergency_triggered"] = True
            print("  *** EMERGENCY THRESHOLD EXCEEDED — emergency flag set ***")
    else:
        state["ground_analysis"] = sensor_data

    state["current_agent"] = "satellite_agent"
    return state


# ---------------------------------------------------------------------------
# Agent 3: Satellite Analysis Agent
# ---------------------------------------------------------------------------

SATELLITE_SYSTEM = """You are the Satellite Data Analysis Agent for an air pollution
monitoring system. You process satellite imagery and atmospheric column data to:

1. Analyze aerosol optical depth (AOD) to estimate column-integrated PM2.5 levels.
   AOD > 0.4 indicates moderate to heavy aerosol loading.
   AOD > 0.7 indicates heavy pollution.
   AOD > 1.0 indicates very heavy or hazardous conditions.

2. Interpret Sentinel-5P TROPOMI NO2 and SO2 tropospheric column densities.
   NO2 column > 6e-5 mol/m2 indicates significant urban/industrial emission.
   SO2 column > 5 Dobson Units indicates a significant industrial or volcanic source.

3. Identify pollution plumes: origin, transport direction, and affected downwind areas.

4. Detect active fires from MODIS thermal anomalies and fire radiative power.
   Fire radiative power > 50 MW indicates a significant fire contributing to PM2.5.

5. Identify pollution hotspots and rank them by intensity.

6. Cross-validate satellite columns against ground station readings where possible.

Always quantify what you observe. Specify coordinates, intensity indices, and
transport vectors. Distinguish between industrial plumes and biomass burning
signatures (SO2-dominant = industrial; PM2.5-dominant without SO2 = biomass/traffic)."""

def satellite_agent(state: dict) -> dict:
    """
    Processes satellite observations for hotspot detection and plume tracking.
    """
    print("\n[SATELLITE AGENT] Processing satellite imagery...")
    scenario = state.get("target_region", "standard")

    analysis, tool_results = call_llm_with_tools(
        system_prompt=SATELLITE_SYSTEM,
        user_message=(
            f"Retrieve and analyze satellite imagery for scenario: {scenario}. SCENARIO={scenario}. "
            "Use fetch_satellite_imagery to get current observations. "
            "Identify all pollution hotspots, plume origins, and any fire events. "
            "Cross-reference satellite columns with ground readings if available:\n"
            f"{_safe_json(state.get('ground_readings', [])[:3])}"
        ),
        tools=[fetch_satellite_imagery],
    )

    sat_obs = []
    for tr in tool_results:
        if tr.get("tool") == "fetch_satellite_imagery" and isinstance(tr.get("result"), dict):
            sat_obs = tr["result"].get("observations", [])
            break

    if not sat_obs:
        try:
            result = fetch_satellite_imagery.invoke({"scenario": scenario})
            if isinstance(result, dict):
                sat_obs = result.get("observations", [])
        except Exception:
            pass

    if sat_obs:
        max_aod    = max(o.get("aerosol_optical_depth", 0) for o in sat_obs if o.get("aerosol_optical_depth"))
        fire_count = sum(o.get("active_fire_count", 0) for o in sat_obs)
        plumes     = sum(1 for o in sat_obs if o.get("plume_detected"))
        print(f"  Max AOD: {max_aod:.3f}, Active fires: {fire_count}, Plumes detected: {plumes}")

        state["satellite_observations"] = sat_obs
        state["risk_scores"].append({
            "domain":            "satellite",
            "max_aod":           max_aod,
            "fire_count":        fire_count,
            "plumes_detected":   plumes,
            "level":             "hazardous" if max_aod > 0.8 else "very_unhealthy" if max_aod > 0.6
                                 else "unhealthy" if max_aod > 0.4 else "moderate",
        })

    state["satellite_analysis"] = analysis
    state["current_agent"]      = "meteorological_agent"
    return state


# ---------------------------------------------------------------------------
# Agent 4: Meteorological Agent
# ---------------------------------------------------------------------------

MET_SYSTEM = """You are the Meteorological and Dispersion Analysis Agent. You assess
atmospheric conditions to understand how pollution disperses (or accumulates) and
to identify transport pathways from sources to receptors.

Your analysis must cover:

1. Ventilation coefficient assessment:
   VC = wind_speed (m/s) x mixing_height (m)
   VC < 3000 m2/s : critical accumulation conditions
   VC 3000-6000   : poor — high risk of episode development
   VC 6000-12000  : moderate
   VC > 12000     : good dispersion

2. Pasquill-Gifford stability class interpretation:
   Class A-B: very unstable — rapid dispersion, low ground-level concentrations
   Class C-D: neutral — moderate dispersion
   Class E-F: stable — pollution trapped near surface, very high risk

3. Wind direction and transport analysis:
   Identify which emission sources are upwind of which monitoring stations.
   Trace the transport pathway: source -> plume -> receptor.

4. Mixing height and boundary layer assessment:
   Mixing height < 500m indicates a shallow trapped layer.
   Combined with low wind: classic pollution episode conditions.

5. Provide a dispersion forecast for the next 12 hours based on met trends.

Be quantitative. State exact wind speeds, directions, mixing heights, and VC values."""

def meteorological_agent(state: dict) -> dict:
    """
    Analyzes meteorological conditions to assess dispersion quality and
    identify transport pathways linking sources to receptors.
    """
    print("\n[METEOROLOGICAL AGENT] Analyzing dispersion conditions...")
    scenario = state.get("target_region", "standard")

    analysis, tool_results = call_llm_with_tools(
        system_prompt=MET_SYSTEM,
        user_message=(
            f"Retrieve and analyze meteorological data for scenario: {scenario}. SCENARIO={scenario}. "
            "Use fetch_meteorological_data to get current conditions. "
            "Identify dispersion quality, stability class, and transport pathways. "
            "Consider the emission sources and ground readings already collected:\n"
            f"Ground max AQI: {max((r.get('aqi', 0) for r in state.get('ground_readings', [])), default='N/A')}\n"
            f"Satellite plumes: {sum(1 for o in state.get('satellite_observations', []) if o.get('plume_detected'))}"
        ),
        tools=[fetch_meteorological_data],
    )

    met_raw = {}
    for tr in tool_results:
        if tr.get("tool") == "fetch_meteorological_data" and isinstance(tr.get("result"), dict):
            met_raw = tr["result"]
            break

    if not met_raw:
        try:
            met_raw = fetch_meteorological_data.invoke({"scenario": scenario})
            if not isinstance(met_raw, dict):
                met_raw = {}
        except Exception:
            met_raw = {}

    if met_raw:
        vc    = met_raw.get("ventilation_coefficient_m2_s", 0)
        stab  = met_raw.get("stability_class", "D")
        disp  = met_raw.get("dispersion_quality", "moderate")
        print(f"  Ventilation coeff: {vc} m2/s, Stability: {stab}, Dispersion: {disp}")

        state["meteorological_summary"] = met_raw
        state["risk_scores"].append({
            "domain":                     "meteorology",
            "ventilation_coefficient":    vc,
            "stability_class":            stab,
            "dispersion_quality":         disp,
            "level":                      met_raw.get("pollution_accumulation_risk", "moderate"),
        })

    state["met_analysis"]  = analysis
    state["current_agent"] = "source_identification_agent"
    return state


# ---------------------------------------------------------------------------
# Agent 5: Source Identification Agent
# ---------------------------------------------------------------------------

SOURCE_SYSTEM = """You are the Pollution Source Identification Agent. Using ground sensor
readings, satellite observations, meteorological transport analysis, and the emission
inventory, your task is to attribute observed pollution to specific sources.

Source attribution methodology:
1. Receptor modeling: which inventory sources are upwind of stations showing exceedances?
2. Chemical fingerprinting: SO2-rich plumes -> industrial/combustion; 
   high NOx/CO ratio -> vehicular; high PM10 without SO2 -> construction/agriculture.
3. Satellite hotspot correlation: do satellite intensity hotspots coincide with
   inventory source locations?
4. Gradient analysis: pollution levels highest near source, decreasing downwind.
5. Compliance cross-check: flag sources with existing non-compliant status.

For each attributed source provide:
- Source ID and name
- Primary pollutants attributed to this source
- Estimated contribution % to current network AQI
- Confidence level (high/medium/low) with justification
- Compliance status and urgency of regulatory response needed

Rank sources by estimated contribution and urgency."""

def source_identification_agent(state: dict) -> dict:
    """
    Cross-references ground data, satellite imagery, meteorology, and the
    emission inventory to identify and rank pollution sources.
    """
    print("\n[SOURCE IDENTIFICATION AGENT] Running source attribution analysis...")

    ground_summary  = _safe_json(state.get("ground_readings", [])[:4])
    satellite_hs    = []
    for obs in state.get("satellite_observations", []):
        satellite_hs.extend(obs.get("pollution_hotspots", []))
    met_wind        = state.get("meteorological_summary", {}).get("wind_direction_deg", "unknown")
    met_vc          = state.get("meteorological_summary", {}).get("ventilation_coefficient_m2_s", "unknown")

    analysis, tool_results = call_llm_with_tools(
        system_prompt=SOURCE_SYSTEM,
        user_message=(
            "Retrieve the emission inventory and perform source attribution. "
            "Use fetch_emission_inventory to get all registered sources. "
            f"\n\nAvailable context:"
            f"\nWind direction: {met_wind} degrees"
            f"\nVentilation coefficient: {met_vc} m2/s"
            f"\nSatellite hotspots: {_safe_json(satellite_hs[:5])}"
            f"\nHighest AQI stations (sample): {ground_summary}"
            f"\nPrevious meteorological analysis: {str(state.get('met_analysis', ''))[:400]}"
        ),
        tools=[fetch_emission_inventory],
    )

    inventory_sources = []
    for tr in tool_results:
        if tr.get("tool") == "fetch_emission_inventory" and isinstance(tr.get("result"), dict):
            inventory_sources = tr["result"].get("sources", [])
            break

    if not inventory_sources:
        try:
            result = fetch_emission_inventory.invoke({})
            if isinstance(result, dict):
                inventory_sources = result.get("sources", [])
        except Exception:
            pass

    if inventory_sources:
        non_compliant = [s for s in inventory_sources if s.get("compliance_status") != "compliant"]
        print(f"  Inventory sources: {len(inventory_sources)}, Non-compliant: {len(non_compliant)}")
        state["pollution_sources"] = inventory_sources
        state["risk_scores"].append({
            "domain":              "source_attribution",
            "total_sources":       len(inventory_sources),
            "non_compliant":       len(non_compliant),
            "level":               "hazardous" if len(non_compliant) >= 3 else
                                   "very_unhealthy" if len(non_compliant) == 2 else
                                   "unhealthy" if len(non_compliant) == 1 else "moderate",
        })

    state["source_analysis"] = analysis
    state["current_agent"]   = "health_impact_agent"
    return state


# ---------------------------------------------------------------------------
# Agent 6: Health Impact Agent
# ---------------------------------------------------------------------------

HEALTH_SYSTEM = """You are the Public Health Impact Assessment Agent. Using current
air quality levels, population exposure data, and WHO/EPA health impact methodology,
you quantify the health burden and determine clinical risk levels.

Health impact quantification framework:
1. Identify all districts with AQI > 100 (unhealthy for sensitive groups).
2. Apply population-weighted exposure:
   exposed_population = district_population * affected_fraction
   sensitive_pop = total_pop * sensitive_group_percentage
3. Apply concentration-response coefficients (Pope & Dockery 2006):
   Respiratory hospitalizations: +0.62% per 10 ug/m3 PM2.5 increase
   Cardiovascular hospitalizations: +0.91% per 10 ug/m3 PM2.5 increase
   All-cause mortality: +0.60% per 10 ug/m3 PM2.5 increase
4. Assess sensitive group risk:
   Groups at elevated risk: children under 12, adults over 65, asthma patients,
   COPD patients, cardiovascular disease patients, pregnant women, outdoor workers
5. Determine hospital alert level:
   normal: AQI < 100  |  elevated: AQI 100-150  |  high: AQI 150-200  |  critical: AQI > 200
6. Generate specific health advisories differentiated by group.

Quantify everything. Provide estimated case counts, not just percentages.
State which specific health outcomes are of concern for each pollutant."""

def health_impact_agent(state: dict) -> dict:
    """
    Assesses public health burden, identifies at-risk populations,
    and determines hospital preparedness level.
    """
    print("\n[HEALTH IMPACT AGENT] Calculating population health burden...")

    aqi_values   = [r.get("aqi", 0) for r in state.get("ground_readings", [])]
    max_aqi      = max(aqi_values) if aqi_values else 0
    avg_aqi      = round(sum(aqi_values) / len(aqi_values), 1) if aqi_values else 0
    pm25_readings = [r.get("pm25_ug_m3", 0) for r in state.get("ground_readings", [])]
    max_pm25     = max(pm25_readings) if pm25_readings else 0

    analysis, tool_results = call_llm_with_tools(
        system_prompt=HEALTH_SYSTEM,
        user_message=(
            "Use fetch_health_risk_tables to retrieve population and vulnerability data. "
            "Then assess health impacts using the following current air quality data:\n"
            f"Network max AQI: {max_aqi}\n"
            f"Network avg AQI: {avg_aqi}\n"
            f"Max PM2.5: {max_pm25} ug/m3\n"
            f"WHO PM2.5 24h guideline: {air_config.pm25_who_24h} ug/m3\n"
            f"Emergency flag triggered: {state.get('emergency_triggered', False)}\n\n"
            "Ground station AQI values by district:\n"
            + "\n".join(
                f"  {r['district']} ({r['station_type']}): AQI={r['aqi']} ({r['aqi_category']}), "
                f"PM2.5={r['pm25_ug_m3']}, NO2={r['no2_ug_m3']}, SO2={r['so2_ug_m3']}"
                for r in state.get("ground_readings", [])
            )
        ),
        tools=[fetch_health_risk_tables],
    )

    # Derive structured health impact data for the report
    if max_aqi >= air_config.aqi_hazardous:
        hospital_level = "critical"
    elif max_aqi >= air_config.aqi_very_unhealthy:
        hospital_level = "high"
    elif max_aqi >= air_config.aqi_unhealthy:
        hospital_level = "elevated"
    else:
        hospital_level = "normal"

    state["health_analysis"] = analysis
    state["health_impact"]   = {
        "max_aqi":                max_aqi,
        "avg_aqi":                avg_aqi,
        "max_pm25_ug_m3":         max_pm25,
        "hospital_alert_level":   hospital_level,
        "emergency_response":     state.get("emergency_triggered", False),
    }
    state["risk_scores"].append({
        "domain":               "health",
        "max_aqi":              max_aqi,
        "hospital_alert_level": hospital_level,
        "level":                hospital_level,
    })

    print(f"  Max AQI: {max_aqi}, Hospital alert: {hospital_level.upper()}")
    state["current_agent"] = "mitigation_agent"
    return state


# ---------------------------------------------------------------------------
# Agent 7: Mitigation Agent
# ---------------------------------------------------------------------------

MITIGATION_SYSTEM = """You are the Pollution Mitigation Planning Agent. Based on all
preceding analyses (ground sensors, satellite, meteorology, sources, health impact),
you generate a prioritized action plan to reduce pollution and protect public health.

For each recommendation you generate, use the log_mitigation_recommendation tool.
Generate between 4 and 8 recommendations based on severity.

Recommendation framework:

EMERGENCY (AQI > 300 / hazardous):
  - Emergency curtailment orders to highest-emitting non-compliant facilities
  - Immediate traffic restrictions in most affected zones
  - Emergency public alert issuance
  - Hospital network activation

HIGH (AQI 200-300 / very unhealthy):
  - Regulatory compliance orders to facilities in exceedance
  - Voluntary emission reduction requests to major point sources
  - Public health advisory issuance
  - School/outdoor event cancellation recommendations

MEDIUM (AQI 150-200 / unhealthy for sensitive groups):
  - Enhanced monitoring frequency at affected stations
  - Low emission zone activation
  - Sensitive group advisories
  - Fleet emission compliance checks

LOW (AQI 100-150 / moderate):
  - Public information messaging
  - Routine permit compliance monitoring
  - Long-term infrastructure recommendations

For each recommendation, quantify expected AQI reduction where possible.
Be specific about which source, regulation, or intervention is targeted."""

def mitigation_agent(state: dict) -> dict:
    """
    Generates and logs a prioritized mitigation action plan using tool calls.
    """
    print("\n[MITIGATION AGENT] Generating action plan...")

    health  = state.get("health_impact", {})
    max_aqi = health.get("max_aqi", 0)
    met     = state.get("meteorological_summary", {})

    non_compliant = [
        s for s in state.get("pollution_sources", [])
        if s.get("compliance_status") != "compliant"
    ]

    # Determine priority tier from AQI
    if max_aqi >= air_config.aqi_hazardous:
        priority_tier = "emergency"
    elif max_aqi >= air_config.aqi_very_unhealthy:
        priority_tier = "emergency"
    elif max_aqi >= air_config.aqi_unhealthy:
        priority_tier = "high"
    elif max_aqi >= air_config.aqi_unhealthy_sensitive:
        priority_tier = "medium"
    else:
        priority_tier = "low"

    # Build demo recommendations directly from simulation data (no LLM required)
    demo_recommendations = []
    from datetime import datetime as _dt

    def _make_rec(priority, category, target, title, desc, aqi_reduction,
                  timeline, basis, cost, benefits):
        rec_id = f"REC-{_dt.now().strftime('%H%M%S')}-{priority[0].upper()}{len(demo_recommendations)}"
        rec = {
            "recommendation_id":       rec_id,
            "priority":                priority,
            "category":                category,
            "target_entity":           target,
            "title":                   title,
            "description":             desc,
            "expected_aqi_reduction":  aqi_reduction,
            "implementation_timeline": timeline,
            "regulatory_basis":        basis,
            "estimated_cost_tier":     cost,
            "co_benefits":             benefits,
        }
        print(f"  [RECOMMENDATION - {priority.upper()}] {title} "
              f"(expected AQI reduction: {aqi_reduction:.1f} pts, timeline: {timeline})")
        return rec

    # Always add: recommendations calibrated to scenario severity
    for nc_source in non_compliant[:2]:
        action = "emergency_curtailment_order" if priority_tier == "emergency" else "compliance_order"
        demo_recommendations.append(_make_rec(
            priority=priority_tier,
            category="regulatory",
            target="regulator",
            title=f"Issue {action.replace('_', ' ').title()} to {nc_source['source_name']}",
            desc=(
                f"{nc_source['source_name']} is currently in {nc_source['compliance_status']} status. "
                f"Primary pollutants: {', '.join(nc_source['primary_pollutants'][:3])}. "
                f"Require immediate emission reduction measures and compliance verification within 24 hours."
            ),
            aqi_reduction=12.0 if priority_tier == "emergency" else 7.0,
            timeline="immediate" if priority_tier == "emergency" else "within_24h",
            basis="Clean Air Act Section 113 / State AQMD Regulation 2-1",
            cost="negligible",
            benefits=["improved public health", "regulatory compliance restoration"],
        ))

    if max_aqi >= air_config.aqi_unhealthy_sensitive:
        disp_quality = met.get("dispersion_quality", "poor")
        demo_recommendations.append(_make_rec(
            priority="high",
            category="traffic",
            target="city_transport",
            title="Activate Low Emission Zone — Downtown Core and East Industrial District",
            desc=(
                f"Current AQI {max_aqi} and {disp_quality} dispersion conditions require "
                "vehicular emission controls. Activate LEZ restricting diesel vehicles older "
                "than Euro 5 standard from Downtown Core and adjacent corridors."
            ),
            aqi_reduction=8.5,
            timeline="within_24h",
            basis="Local Air Quality Management Order 2022, Section 4.2",
            cost="low",
            benefits=["noise reduction", "traffic calming", "fuel savings for compliant vehicles"],
        ))

    demo_recommendations.append(_make_rec(
        priority="high" if max_aqi >= 150 else "medium",
        category="public_health",
        target="public",
        title="Issue Public Health Advisory for Sensitive Populations",
        desc=(
            f"AQI of {max_aqi} ({health.get('hospital_alert_level', 'elevated')} risk level) "
            "requires a public advisory. Advise: avoid prolonged outdoor exercise; "
            "sensitive groups (children, elderly, respiratory patients) to remain indoors; "
            "schools to cancel outdoor activities."
        ),
        aqi_reduction=0.0,
        timeline="immediate",
        basis="WHO Air Quality Guidelines 2021 / State Public Health Emergency Protocol",
        cost="negligible",
        benefits=["reduced emergency department visits", "reduced acute health burden"],
    ))

    demo_recommendations.append(_make_rec(
        priority="medium",
        category="operational",
        target="facility_operator",
        title="Request Voluntary Emission Curtailment from Major Point Sources",
        desc=(
            "Request all industrial facilities within the East Industrial District and Port District "
            "to voluntarily curtail non-essential combustion processes by 25% until AQI returns "
            "below 100. Priority targets: Metro Chemical Works, Port Authority Terminal, Metro Power Station."
        ),
        aqi_reduction=15.0,
        timeline="within_24h",
        basis="State AQMD Air Pollution Episode Plan — Stage I Procedures",
        cost="medium",
        benefits=["energy savings", "reduced maintenance costs from lower combustion cycling"],
    ))

    if met.get("dispersion_quality") in ("very_poor", "poor"):
        demo_recommendations.append(_make_rec(
            priority="medium",
            category="infrastructure",
            target="city_transport",
            title="Implement HGV Curfew — South Highway Corridor",
            desc=(
                f"Stagnant meteorological conditions (ventilation coefficient: "
                f"{met.get('ventilation_coefficient_m2_s', 'N/A')} m2/s, "
                f"stability class: {met.get('stability_class', 'N/A')}) "
                "are preventing normal dispersion. Suspend heavy goods vehicle operations "
                "on South Highway Corridor between 06:00-22:00 until conditions improve."
            ),
            aqi_reduction=5.5,
            timeline="within_24h",
            basis="Road Traffic Regulation Act — Air Quality Emergency Provisions",
            cost="medium",
            benefits=["noise reduction", "road safety improvement"],
        ))

    # Use LLM if available (will override demo recommendations)
    try:
        from agents.base import _demo_mode
        if not _demo_mode():
            analysis, tool_results = call_llm_with_tools(
                system_prompt=MITIGATION_SYSTEM,
                user_message=(
                    "Generate a complete mitigation action plan using log_mitigation_recommendation. "
                    f"Network max AQI: {max_aqi}\n"
                    f"Hospital alert: {health.get('hospital_alert_level')}\n"
                    f"Non-compliant sources: {[s['source_name'] for s in non_compliant]}\n"
                    f"Dispersion: {met.get('dispersion_quality')}"
                ),
                tools=[log_mitigation_recommendation, request_traffic_restriction],
            )
            llm_recs = [
                tr["result"] for tr in tool_results
                if tr.get("tool") == "log_mitigation_recommendation" and isinstance(tr.get("result"), dict)
            ]
            if llm_recs:
                demo_recommendations = llm_recs
    except Exception:
        pass

    print(f"  Recommendations logged: {len(demo_recommendations)}")

    state["mitigation_recommendations"] = demo_recommendations
    state["current_agent"]              = "alert_agent"
    return state


# ---------------------------------------------------------------------------
# Agent 8: Alert Agent
# ---------------------------------------------------------------------------

ALERT_SYSTEM = """You are the Public Alert and Regulatory Notification Agent.
Based on the completed situation analysis, you issue:

1. PUBLIC HEALTH ALERTS: Using issue_public_health_alert
   - Determine severity: advisory (AQI 101-150), warning (151-200), alert (201-300), emergency (>300)
   - Draft clear, non-technical health messages for the general public
   - Provide specific action instructions (stay indoors, use masks, avoid exercise, etc.)
   - Include specific guidance for sensitive groups (children, elderly, respiratory patients)
   - Select appropriate distribution channels

2. REGULATORY ENFORCEMENT ACTIONS: Using issue_regulatory_action
   - For each non-compliant source, issue the appropriate enforcement action
   - NOV (Notice of Violation) for initial exceedance
   - Compliance Order for ongoing/repeated violations
   - Emergency Shutdown Order for hazardous-level emissions

3. HOSPITAL NETWORK NOTIFICATION: Using notify_hospital_network
   - Always notify hospitals if AQI > 150
   - Specify expected case types and volume increase estimate
   - Provide clinical guidance for treating pollution-related conditions

Issue all appropriate alerts and notifications. Do not omit any non-compliant
source that requires regulatory action. Be thorough."""

def alert_agent(state: dict) -> dict:
    """
    Issues public health alerts, regulatory enforcement actions, and
    hospital network notifications.
    """
    print("\n[ALERT AGENT] Issuing alerts and regulatory notifications...")

    health  = state.get("health_impact", {})
    max_aqi = health.get("max_aqi", 0)
    met     = state.get("meteorological_summary", {})

    non_compliant = [
        s for s in state.get("pollution_sources", [])
        if s.get("compliance_status") != "compliant"
    ]

    # Determine alert severity
    if max_aqi >= air_config.aqi_hazardous:
        severity = "emergency"
    elif max_aqi >= air_config.aqi_very_unhealthy:
        severity = "alert"
    elif max_aqi >= air_config.aqi_unhealthy:
        severity = "warning"
    elif max_aqi >= air_config.aqi_unhealthy_sensitive:
        severity = "advisory"
    else:
        severity = "info"

    hospital_level = health.get("hospital_alert_level", "normal")

    # Dominant pollutant from ground readings
    dom_pols = {}
    for r in state.get("ground_readings", []):
        dp = r.get("dominant_pollutant", "PM2.5")
        dom_pols[dp] = dom_pols.get(dp, 0) + 1
    dom_pol = max(dom_pols, key=dom_pols.get) if dom_pols else "PM2.5"

    affected_districts = list(set(
        r["district"] for r in state.get("ground_readings", [])
        if r.get("aqi_category") not in ("good", "moderate")
    ))

    from datetime import datetime as _dt

    # Issue public health alert
    public_alerts = []
    if max_aqi >= air_config.aqi_moderate:
        alert_id = f"ALERT-{_dt.now().strftime('%Y%m%d-%H%M%S')}"
        alert = {
            "alert_id":            alert_id,
            "severity":            severity,
            "aqi_level":           max_aqi,
            "dominant_pollutant":  dom_pol,
            "districts_notified":  len(affected_districts),
            "affected_districts":  affected_districts,
            "channels":            ["mobile_app", "website", "sms", "digital_signage"],
            "health_message": (
                f"Air quality is currently {severity.upper()} in {', '.join(affected_districts[:3])}. "
                f"AQI of {max_aqi} driven by elevated {dom_pol}. "
                "Avoid prolonged outdoor activity. Sensitive groups should remain indoors."
            ),
            "issued_at": _dt.now().isoformat(),
            "status": "issued",
        }
        public_alerts.append(alert)
        print(f"  [PUBLIC ALERT - {severity.upper()}] AQI {max_aqi} ({dom_pol}) "
              f"in {len(affected_districts)} district(s)")

    # Issue regulatory enforcement actions
    reg_actions = []
    for nc in non_compliant:
        if nc["compliance_status"] == "permit_exceeded":
            action_type = "compliance_order"
        elif nc["compliance_status"] == "exceedance" and max_aqi >= 200:
            action_type = "emergency_shutdown_order"
        else:
            action_type = "notice_of_violation"

        case_num = f"ENF-{_dt.now().strftime('%Y%m%d-%H%M%S')}-{nc['source_id']}"
        action = {
            "case_number":     case_num,
            "action_type":     action_type,
            "source_id":       nc["source_id"],
            "source_name":     nc["source_name"],
            "compliance_status": nc["compliance_status"],
            "required_action": "Immediately implement all available emission reduction measures and report compliance within 4 hours.",
            "issued_at":       _dt.now().isoformat(),
            "status":          "issued",
        }
        reg_actions.append(action)
        print(f"  [REGULATORY] {action_type.upper()}: {nc['source_name']}")

    # Hospital notification
    hospital_notifs = []
    if hospital_level in ("elevated", "high", "critical") or max_aqi >= air_config.aqi_unhealthy:
        notif_id = f"HOSP-{_dt.now().strftime('%Y%m%d-%H%M%S')}"
        notif = {
            "notification_id":   notif_id,
            "alert_level":       hospital_level,
            "aqi":               max_aqi,
            "primary_pollutant": dom_pol,
            "expected_case_types": ["asthma exacerbation", "COPD exacerbation",
                                    "cardiovascular event", "respiratory distress"],
            "expected_volume_increase_pct": 15.0 if max_aqi < 200 else 35.0,
            "hospitals_notified": 6,
            "issued_at":         _dt.now().isoformat(),
            "status":            "dispatched",
        }
        hospital_notifs.append(notif)
        print(f"  [HOSPITAL ALERT - {hospital_level.upper()}] AQI {max_aqi} ({dom_pol}) "
              f"— expected ED volume increase {notif['expected_volume_increase_pct']:.0f}%")

    # If LLM available, use it instead
    try:
        from agents.base import _demo_mode as _dm
        if not _dm():
            analysis, tool_results = call_llm_with_tools(
                system_prompt=ALERT_SYSTEM,
                user_message=(
                    "Issue all required alerts and notifications. "
                    f"Max AQI: {max_aqi}, severity: {severity}, "
                    f"non-compliant sources: {[s['source_name'] for s in non_compliant]}"
                ),
                tools=[issue_public_health_alert, issue_regulatory_action, notify_hospital_network],
            )
            llm_alerts  = [tr["result"] for tr in tool_results if tr.get("tool") == "issue_public_health_alert"]
            llm_reg     = [tr["result"] for tr in tool_results if tr.get("tool") == "issue_regulatory_action"]
            llm_hosp    = [tr["result"] for tr in tool_results if tr.get("tool") == "notify_hospital_network"]
            if llm_alerts or llm_reg:
                public_alerts = llm_alerts
                reg_actions   = llm_reg
                hospital_notifs = llm_hosp
    except Exception:
        pass

    print(f"  Public alerts issued: {len(public_alerts)}")
    print(f"  Regulatory actions:   {len(reg_actions)}")
    print(f"  Hospital notifications: {len(hospital_notifs)}")

    state["public_alerts"]      = public_alerts
    state["regulatory_actions"] = state.get("regulatory_actions", []) + reg_actions
    state["current_agent"]      = "supervisor_synthesis"
    return state


# ---------------------------------------------------------------------------
# Agent 9: Supervisor — Synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are the Supervisor Agent for an urban air quality monitoring
system. You have received completed analyses from all domain agents:
  - Ground sensor analysis (real-time station network)
  - Satellite imagery analysis (hotspots, plumes, fires)
  - Meteorological / dispersion analysis
  - Pollution source attribution
  - Public health impact assessment
  - Mitigation action plan
  - Alert and regulatory notification summary

Your task is to produce a final SITUATION REPORT structured as follows:

SECTION 1: EXECUTIVE SUMMARY (3-5 sentences: overall air quality status, immediate risk level,
           number of sources identified, number of actions taken)

SECTION 2: CURRENT AIR QUALITY STATUS
  - Network-wide AQI statistics (max, average, worst stations)
  - District-level breakdown
  - Dominant pollutants and their sources

SECTION 3: POLLUTION SOURCE ATTRIBUTION
  - Primary sources contributing to current episode
  - Compliance status of each source
  - Estimated contribution percentages

SECTION 4: HEALTH IMPACT ASSESSMENT
  - Population at risk
  - Sensitive groups and specific risks
  - Hospital preparedness level

SECTION 5: METEOROLOGICAL ASSESSMENT
  - Dispersion quality and forecast
  - Expected episode evolution over next 12-24 hours

SECTION 6: ACTIONS TAKEN
  - Public alerts issued
  - Regulatory enforcement actions
  - Traffic restrictions
  - Mitigation recommendations (summarized by priority)

SECTION 7: PRIORITY ACTIONS OUTSTANDING (not yet implemented, recommended for regulatory decision)

Conclude with an OVERALL RISK LEVEL: CRITICAL / HIGH / ELEVATED / MODERATE / LOW
and an OVERALL AQI FORECAST for the next 6 hours."""

def supervisor_synthesis_agent(state: dict) -> dict:
    """
    Reads all agent analyses and produces the final consolidated situation report.
    """
    print("\n[SUPERVISOR — SYNTHESIS] Compiling final situation report...")

    risk_scores = state.get("risk_scores", [])
    levels_map  = {"good": 1, "moderate": 2, "unhealthy_sensitive": 3,
                   "unhealthy": 4, "very_unhealthy": 5, "hazardous": 6,
                   "normal": 1, "elevated": 3, "high": 5, "critical": 6,
                   "poor": 3, "very_poor": 5, "low": 1, "medium": 2}
    risk_values = [levels_map.get(str(rs.get("level", "low")).lower(), 2) for rs in risk_scores]
    overall_level_idx = max(risk_values) if risk_values else 2
    level_labels = {1: "LOW", 2: "MODERATE", 3: "ELEVATED", 4: "HIGH", 5: "VERY HIGH", 6: "CRITICAL"}
    overall_level = level_labels.get(overall_level_idx, "MODERATE")

    synthesis_prompt = (
        "Compile the final situation report from all domain analyses. "
        f"Overall risk level computed by supervisor: {overall_level}\n\n"
        f"GROUND ANALYSIS:\n{str(state.get('ground_analysis', 'Not available'))[:500]}\n\n"
        f"SATELLITE ANALYSIS:\n{str(state.get('satellite_analysis', 'Not available'))[:400]}\n\n"
        f"METEOROLOGICAL ANALYSIS:\n{str(state.get('met_analysis', 'Not available'))[:400]}\n\n"
        f"SOURCE ATTRIBUTION:\n{str(state.get('source_analysis', 'Not available'))[:400]}\n\n"
        f"HEALTH ANALYSIS:\n{str(state.get('health_analysis', 'Not available'))[:400]}\n\n"
        f"ACTIONS TAKEN:\n"
        f"  Public alerts: {len(state.get('public_alerts', []))}\n"
        f"  Regulatory actions: {len(state.get('regulatory_actions', []))}\n"
        f"  Mitigation recommendations: {len(state.get('mitigation_recommendations', []))}\n"
        f"  Emergency triggered: {state.get('emergency_triggered', False)}\n"
    )

    report = call_llm(
        system_prompt=SYNTHESIS_SYSTEM,
        user_message=synthesis_prompt,
        max_tokens=1600,
    )

    state["situation_report"] = report
    state["current_agent"]    = "END"

    print("\n" + "=" * 70)
    print("  SITUATION REPORT COMPLETE")
    print(f"  Overall Risk Level: {overall_level}")
    print(f"  Ground readings:    {len(state.get('ground_readings', []))} stations")
    print(f"  Satellite obs:      {len(state.get('satellite_observations', []))} platforms")
    print(f"  Sources identified: {len(state.get('pollution_sources', []))}")
    print(f"  Mitigations logged: {len(state.get('mitigation_recommendations', []))}")
    print(f"  Public alerts:      {len(state.get('public_alerts', []))}")
    print(f"  Regulatory actions: {len(state.get('regulatory_actions', []))}")
    print(f"  Emergency flag:     {state.get('emergency_triggered', False)}")
    print(f"  Disclaimer: {air_config.disclaimer}")
    print("=" * 70)

    return state
