# tools/sensor_tools.py
# LangChain tools that wrap the simulation data layer.
# In production: swap each function body for the real API call listed in simulation.py.
# All agent code and the LangGraph pipeline require zero changes when you do so.

from datetime import datetime

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):
        return fn

from data.simulation import (
    get_ground_readings,
    get_satellite_observations,
    get_meteorological_data,
    get_emission_inventory,
    describe_scenario,
)


@tool
def fetch_ground_sensor_data(scenario: str = "standard") -> dict:
    """
    Fetch real-time pollutant readings from all ground monitoring stations.

    Returns current PM2.5, PM10, NO2, SO2, CO, O3 concentrations plus computed AQI
    for all stations in the network. Also returns co-located meteorological
    measurements (temperature, wind, humidity, pressure).

    In production: queries OpenAQ API or state AQMD SCADA telemetry.

    Args:
        scenario: 'standard' for a typical day, 'episode' for severe pollution event

    Returns:
        dict with stations list and network summary statistics
    """
    readings = get_ground_readings(scenario)
    if not readings:
        return {"error": "No station data available", "stations": []}

    aqis     = [r["aqi"] for r in readings]
    max_aqi  = max(aqis)
    avg_aqi  = round(sum(aqis) / len(aqis), 1)
    critical = [r for r in readings if r["aqi_category"] in ("unhealthy", "very_unhealthy", "hazardous")]
    moderate = [r for r in readings if r["aqi_category"] in ("moderate", "unhealthy_sensitive")]

    return {
        "timestamp":           datetime.now().isoformat(timespec="minutes"),
        "station_count":       len(readings),
        "stations":            readings,
        "network_max_aqi":     max_aqi,
        "network_avg_aqi":     avg_aqi,
        "critical_stations":   [r["station_id"] for r in critical],
        "moderate_stations":   [r["station_id"] for r in moderate],
        "scenario_description": describe_scenario(scenario),
    }


@tool
def fetch_satellite_imagery(scenario: str = "standard") -> dict:
    """
    Retrieve processed satellite data products for the monitoring region.

    Includes:
      - Sentinel-5P TROPOMI: tropospheric NO2, SO2, CO columns and aerosol optical depth
      - MODIS Terra: aerosol optical depth, fire detection, fire radiative power
      - Landsat-9: urban heat island intensity, surface reflectance

    In production: Google Earth Engine Python API or direct ESA/NASA DAAC requests.

    Args:
        scenario: 'standard' or 'episode'

    Returns:
        dict with all satellite observations and cross-platform summary
    """
    observations = get_satellite_observations(scenario)
    if not observations:
        return {"error": "No satellite data available", "observations": []}

    max_aod    = max(o["aerosol_optical_depth"] for o in observations if o.get("aerosol_optical_depth"))
    total_fire = sum(o["active_fire_count"] for o in observations)
    plumes     = [o for o in observations if o.get("plume_detected")]
    all_hotspots = []
    for o in observations:
        if o.get("pollution_hotspots"):
            all_hotspots.extend(o["pollution_hotspots"])

    # Deduplicate hotspots by source_name
    seen = set()
    unique_hotspots = []
    for h in all_hotspots:
        key = h.get("source_name", "")
        if key not in seen:
            seen.add(key)
            unique_hotspots.append(h)

    return {
        "timestamp":                  datetime.now().isoformat(timespec="minutes"),
        "observations":               observations,
        "satellite_count":            len(observations),
        "max_aerosol_optical_depth":  round(max_aod, 3),
        "total_active_fire_count":    total_fire,
        "plume_detections":           len(plumes),
        "plume_sources":              [o.get("plume_origin_description") for o in plumes if o.get("plume_origin_description")],
        "unique_hotspots":            unique_hotspots,
        "agricultural_burn_alert":    total_fire > 2,
    }


@tool
def fetch_meteorological_data(scenario: str = "standard") -> dict:
    """
    Retrieve current meteorological conditions and dispersion parameters.

    Includes surface conditions, wind field, mixing height, Pasquill-Gifford
    stability class, ventilation coefficient, and derived dispersion quality.

    Key parameter — ventilation coefficient (wind_speed x mixing_height):
      < 3000 m2/s : very poor — critical pollution accumulation risk
      3000-6000   : poor — high accumulation risk
      6000-12000  : moderate
      > 12000     : good — effective dispersion

    In production: NOAA HRRR model output API or Meteomatics commercial API.

    Args:
        scenario: 'standard' or 'episode'

    Returns:
        Full meteorological and dispersion parameter dict
    """
    return get_meteorological_data(scenario)


@tool
def fetch_emission_inventory() -> dict:
    """
    Retrieve the registered emission source inventory for the monitoring region.

    Contains all permitted industrial facilities, area emission sources, and
    known non-point sources (traffic corridors, agricultural zones).
    Used to cross-reference with sensor readings for source attribution.

    In production: US EPA NEI API, state AQMD permit database, EU E-PRTR.

    Returns:
        dict with full source inventory and summary by category
    """
    sources = get_emission_inventory()
    by_category = {}
    for s in sources:
        cat = s["source_category"]
        by_category.setdefault(cat, []).append(s["source_id"])

    non_compliant = [s for s in sources if s["compliance_status"] != "compliant"]

    return {
        "timestamp":           datetime.now().isoformat(timespec="minutes"),
        "total_sources":       len(sources),
        "sources":             sources,
        "sources_by_category": by_category,
        "non_compliant_sources": non_compliant,
        "non_compliant_count": len(non_compliant),
    }


@tool
def fetch_health_risk_tables() -> dict:
    """
    Retrieve population exposure and health vulnerability data for each district.

    Provides population counts, sensitive group sizes, and concentration-response
    coefficients used in health impact calculations.

    In production:
      - Census population data by district (US Census Bureau API)
      - City health department sensitive population registry
      - WHO/EPA concentration-response functions from published epidemiology

    Returns:
        dict with district-level population and vulnerability data
    """
    return {
        "districts": {
            "Downtown Core": {
                "population":           285000,
                "sensitive_population": 48000,
                "hospitals_nearby":     3,
                "schools_nearby":       12,
                "vulnerable_pct":       16.8,
            },
            "East Industrial District": {
                "population":           42000,
                "sensitive_population": 8500,
                "hospitals_nearby":     1,
                "schools_nearby":       4,
                "vulnerable_pct":       20.2,
            },
            "Riverside West": {
                "population":           118000,
                "sensitive_population": 21000,
                "hospitals_nearby":     2,
                "schools_nearby":       8,
                "vulnerable_pct":       17.8,
            },
            "North Campus": {
                "population":           65000,
                "sensitive_population": 8200,
                "hospitals_nearby":     1,
                "schools_nearby":       6,
                "vulnerable_pct":       12.6,
            },
            "Port District": {
                "population":           28000,
                "sensitive_population": 6200,
                "hospitals_nearby":     0,
                "schools_nearby":       2,
                "vulnerable_pct":       22.1,
            },
            "North Suburbs": {
                "population":           195000,
                "sensitive_population": 38000,
                "hospitals_nearby":     2,
                "schools_nearby":       18,
                "vulnerable_pct":       19.5,
            },
            "South Highway Corridor": {
                "population":           88000,
                "sensitive_population": 16500,
                "hospitals_nearby":     1,
                "schools_nearby":       7,
                "vulnerable_pct":       18.8,
            },
        },
        # Concentration-response coefficients from WHO GBD and Pope & Dockery 2006
        # Unit: fractional increase in endpoint per 10 ug/m3 PM2.5
        "concentration_response_coefficients": {
            "respiratory_hospitalizations_per_10ugm3_pm25": 0.0062,
            "cardiovascular_hospitalizations_per_10ugm3_pm25": 0.0091,
            "all_cause_mortality_per_10ugm3_pm25": 0.0060,
            "asthma_attacks_per_10ugm3_no2": 0.0045,
        },
    }
