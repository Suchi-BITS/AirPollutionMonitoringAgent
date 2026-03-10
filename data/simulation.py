# data/simulation.py
#
# Complete simulation layer for the Air Pollution Monitoring Agent System.
# Provides all data that would come from real-world sensors and satellite feeds.
#
# SIMULATION DESIGN:
#   - Date-seeded RNG: reproducible within a calendar day, different day-to-day
#   - Two scenario profiles: STANDARD (moderate urban pollution) and EPISODE
#     (severe pollution event — industrial accident + stagnant weather)
#   - Ground station readings are physically consistent:
#       - Industrial-adjacent stations have higher SO2, PM2.5
#       - Traffic stations have higher NO2, CO
#       - Background stations are cleanest
#   - Satellite data is consistent with ground readings:
#       - High AOD when PM2.5 is elevated
#       - High NO2 column where ground NO2 is elevated
#       - Fire detection triggered in agricultural burn scenario
#   - Meteorological conditions drive dispersion logic:
#       - Low wind + stable atmosphere = accumulation (high AQI)
#       - High wind + unstable = dispersion (lower AQI)
#
# PRODUCTION REPLACEMENT GUIDE:
#
#   get_ground_readings()        -> OpenAQ platform API (openaq.org/api)
#                                   US EPA AirNow API (airnowapi.org)
#                                   State AQMD SCADA/telemetry feeds
#                                   Vaisala / Thermo Fisher analyzer data streams
#
#   get_satellite_observations() -> ESA Copernicus Open Access Hub (Sentinel-5P)
#                                   NASA LAADS DAAC (MODIS Terra/Aqua)
#                                   USGS EarthExplorer (Landsat-9)
#                                   Google Earth Engine Python API (all three above)
#
#   get_meteorological_data()    -> NOAA NDFD / HRRR model output
#                                   Meteomatics API
#                                   DTN WeatherSentry API
#                                   On-site met towers (Campbell Scientific)
#
#   get_emission_inventory()     -> US EPA NEI (api.epa.gov/FACT/1.0)
#                                   EU E-PRTR database
#                                   State AQMD permit database APIs
#                                   Global Power Plant Database (WRI)

import random
import math
from datetime import datetime, date
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Seeded RNG
# ---------------------------------------------------------------------------

def _rng(extra: int = 0) -> random.Random:
    """
    Date-seeded RNG. extra allows independent streams for different domains.
    Same values all day; different values tomorrow.
    """
    seed = int(date.today().strftime("%Y%m%d")) + extra
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Network geography
# Two pollution scenarios available: "standard" and "episode"
# ---------------------------------------------------------------------------

SCENARIOS = {
    "standard": {
        "label":       "Standard Urban Pollution Day",
        "description": "Typical weekday with elevated traffic emissions and moderate industrial activity. "
                       "AQI ranges from moderate to unhealthy-for-sensitive-groups across districts.",
        "met_wind_ms":    3.2,
        "met_stability":  "slightly_unstable",
        "pm25_baseline":  28.0,
        "no2_multiplier": 1.0,
        "so2_multiplier": 1.0,
        "episode_source": None,
    },
    "episode": {
        "label":       "Severe Pollution Episode",
        "description": "Industrial facility malfunction combined with stagnant high-pressure system. "
                       "SO2 and PM2.5 at hazardous levels downwind of industrial zone. "
                       "Agricultural burns detected in satellite imagery. Emergency response warranted.",
        "met_wind_ms":    0.8,
        "met_stability":  "very_stable",
        "pm25_baseline":  95.0,
        "no2_multiplier": 2.4,
        "so2_multiplier": 6.5,
        "episode_source": "Chemical Plant Sector 4 — uncontrolled SO2 release",
    },
}


# ---------------------------------------------------------------------------
# Station definitions
# Each station has a fixed location, type, and characteristic pollution profile.
# ---------------------------------------------------------------------------

STATION_DEFINITIONS = {
    "STN-001": {
        "name":     "Central Traffic Monitor",
        "lat":       40.7128, "lon": -74.0060,
        "district": "Downtown Core",
        "type":     "traffic",
        "pm25_base": 35.0, "no2_base": 85.0, "so2_base":  8.0,
        "co_base":    4.5, "o3_base":  62.0, "pm10_base": 60.0,
    },
    "STN-002": {
        "name":     "Industrial Zone East",
        "lat":       40.7282, "lon": -73.9842,
        "district": "East Industrial District",
        "type":     "industrial",
        "pm25_base": 52.0, "no2_base": 72.0, "so2_base": 48.0,
        "co_base":    6.2, "o3_base":  38.0, "pm10_base": 95.0,
    },
    "STN-003": {
        "name":     "Riverside Residential",
        "lat":       40.6892, "lon": -74.0445,
        "district": "Riverside West",
        "type":     "residential",
        "pm25_base": 18.0, "no2_base": 38.0, "so2_base":  4.0,
        "co_base":    2.1, "o3_base":  78.0, "pm10_base": 32.0,
    },
    "STN-004": {
        "name":     "University Background",
        "lat":       40.7299, "lon": -74.0322,
        "district": "North Campus",
        "type":     "background",
        "pm25_base": 12.0, "no2_base": 22.0, "so2_base":  2.5,
        "co_base":    1.4, "o3_base":  88.0, "pm10_base": 22.0,
    },
    "STN-005": {
        "name":     "Port Authority Monitor",
        "lat":       40.7001, "lon": -74.0150,
        "district": "Port District",
        "type":     "industrial",
        "pm25_base": 45.0, "no2_base": 95.0, "so2_base": 35.0,
        "co_base":    5.8, "o3_base":  42.0, "pm10_base": 82.0,
    },
    "STN-006": {
        "name":     "Suburban Residential North",
        "lat":       40.7580, "lon": -73.9855,
        "district": "North Suburbs",
        "type":     "residential",
        "pm25_base": 14.0, "no2_base": 28.0, "so2_base":  3.0,
        "co_base":    1.8, "o3_base":  82.0, "pm10_base": 26.0,
    },
    "STN-007": {
        "name":     "Highway Corridor South",
        "lat":       40.6743, "lon": -73.9442,
        "district": "South Highway Corridor",
        "type":     "traffic",
        "pm25_base": 42.0, "no2_base": 110.0, "so2_base": 12.0,
        "co_base":    5.2, "o3_base":   55.0, "pm10_base": 75.0,
    },
    "STN-008": {
        "name":     "Chemical Works Fence Line",
        "lat":       40.7355, "lon": -73.9612,
        "district": "East Industrial District",
        "type":     "industrial",
        "pm25_base": 58.0, "no2_base": 65.0, "so2_base": 85.0,
        "co_base":    7.5, "o3_base":  28.0, "pm10_base": 105.0,
    },
}


# ---------------------------------------------------------------------------
# Emission inventory — known sources in the network area
# ---------------------------------------------------------------------------

EMISSION_INVENTORY = [
    {
        "source_id":    "SRC-001",
        "source_name":  "Metro Chemical Works Unit A",
        "source_category": "industrial",
        "lat": 40.7360, "lon": -73.9605,
        "district":     "East Industrial District",
        "primary_pollutants": ["SO2", "PM2.5", "NOx", "VOC"],
        "emission_rate_kg_hr": {"SO2": 85.0, "PM2.5": 12.0, "NOx": 28.0, "VOC": 18.0},
        "permit_holder":  "Metro Chemical Corp",
        "permit_number":  "AQM-2021-00441",
        "compliance_status": "exceedance",
        "last_inspection_date": "2024-09-14",
        "operating_status": "active",
    },
    {
        "source_id":    "SRC-002",
        "source_name":  "Port Authority Container Terminal",
        "source_category": "industrial",
        "lat": 40.7005, "lon": -74.0140,
        "district":     "Port District",
        "primary_pollutants": ["PM2.5", "PM10", "NOx", "SO2"],
        "emission_rate_kg_hr": {"PM2.5": 8.5, "PM10": 22.0, "NOx": 45.0, "SO2": 18.0},
        "permit_holder":  "Port Authority of Metro City",
        "permit_number":  "AQM-2019-00218",
        "compliance_status": "compliant",
        "last_inspection_date": "2025-02-28",
        "operating_status": "active",
    },
    {
        "source_id":    "SRC-003",
        "source_name":  "Downtown Traffic Corridor (I-495 / Route 1 Junction)",
        "source_category": "vehicular",
        "lat": 40.7115, "lon": -74.0080,
        "district":     "Downtown Core",
        "primary_pollutants": ["NOx", "CO", "PM2.5", "VOC"],
        "emission_rate_kg_hr": {"NOx": 62.0, "CO": 120.0, "PM2.5": 5.5, "VOC": 35.0},
        "permit_holder":  "N/A (area source)",
        "permit_number":  "N/A",
        "compliance_status": "compliant",
        "last_inspection_date": "N/A",
        "operating_status": "active",
    },
    {
        "source_id":    "SRC-004",
        "source_name":  "North Valley Agricultural Zone — Seasonal Burns",
        "source_category": "agricultural",
        "lat": 40.7680, "lon": -73.9720,
        "district":     "North Agricultural Zone",
        "primary_pollutants": ["PM2.5", "PM10", "CO", "NOx"],
        "emission_rate_kg_hr": {"PM2.5": 45.0, "PM10": 95.0, "CO": 210.0, "NOx": 18.0},
        "permit_holder":  "Regional Farming Cooperative",
        "permit_number":  "BURN-2025-0042",
        "compliance_status": "permit_exceeded",
        "last_inspection_date": "2025-01-10",
        "operating_status": "intermittent",
    },
    {
        "source_id":    "SRC-005",
        "source_name":  "South Highway Diesel Fleet Corridor",
        "source_category": "vehicular",
        "lat": 40.6750, "lon": -73.9450,
        "district":     "South Highway Corridor",
        "primary_pollutants": ["NOx", "PM2.5", "CO", "BC"],
        "emission_rate_kg_hr": {"NOx": 88.0, "PM2.5": 9.2, "CO": 95.0, "BC": 2.8},
        "permit_holder":  "N/A (area source)",
        "permit_number":  "N/A",
        "compliance_status": "compliant",
        "last_inspection_date": "N/A",
        "operating_status": "active",
    },
    {
        "source_id":    "SRC-006",
        "source_name":  "Metro Power Station Unit 2",
        "source_category": "industrial",
        "lat": 40.7420, "lon": -73.9580,
        "district":     "East Industrial District",
        "primary_pollutants": ["SO2", "NOx", "PM10", "CO2"],
        "emission_rate_kg_hr": {"SO2": 38.0, "NOx": 52.0, "PM10": 15.0},
        "permit_holder":  "Metro Energy Utilities",
        "permit_number":  "AQM-2018-00095",
        "compliance_status": "compliant",
        "last_inspection_date": "2025-01-22",
        "operating_status": "active",
    },
    {
        "source_id":    "SRC-007",
        "source_name":  "Riverside Construction Site — Phase 3",
        "source_category": "construction",
        "lat": 40.6900, "lon": -74.0420,
        "district":     "Riverside West",
        "primary_pollutants": ["PM10", "PM2.5", "CO"],
        "emission_rate_kg_hr": {"PM10": 18.0, "PM2.5": 7.5, "CO": 12.0},
        "permit_holder":  "Riverside Development LLC",
        "permit_number":  "CONST-2024-0178",
        "compliance_status": "compliant",
        "last_inspection_date": "2024-11-05",
        "operating_status": "intermittent",
    },
]


# ---------------------------------------------------------------------------
# AQI calculation
# ---------------------------------------------------------------------------

def _compute_aqi(pm25: float, pm10: float, no2: float,
                 so2: float, co: float, o3: float) -> tuple:
    """
    Compute US EPA AQI from pollutant concentrations.
    Returns (aqi_value, aqi_category, dominant_pollutant).
    Simplified linear interpolation within breakpoint table.
    """
    # (Cp_low, Cp_high, AQI_low, AQI_high) breakpoints
    def _linear(cp, breakpoints):
        for i in range(len(breakpoints) - 1):
            cp_lo, cp_hi, aqi_lo, aqi_hi = breakpoints[i]
            if cp <= cp_hi:
                return round(((aqi_hi - aqi_lo) / (cp_hi - cp_lo)) * (cp - cp_lo) + aqi_lo)
        return 500

    pm25_bp  = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
                (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)]
    pm10_bp  = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
                (255,354,151,200),(355,424,201,300),(425,604,301,500)]
    no2_bp   = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
                (361,649,151,200),(650,1249,201,300),(1250,2049,301,500)]
    so2_bp   = [(0,35,0,50),(36,75,51,100),(76,185,101,150),
                (186,304,151,200),(305,604,201,300),(605,1004,301,500)]
    co_bp    = [(0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),
                (12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,50.4,301,500)]
    o3_bp    = [(0,54,0,50),(55,70,51,100),(71,85,101,150),
                (86,105,151,200),(106,200,201,300)]

    scores = {
        "PM2.5": _linear(pm25, pm25_bp),
        "PM10":  _linear(pm10, pm10_bp),
        "NO2":   _linear(no2,  no2_bp),
        "SO2":   _linear(so2,  so2_bp),
        "CO":    _linear(co,   co_bp),
        "O3":    _linear(o3,   o3_bp),
    }

    dominant = max(scores, key=scores.get)
    aqi      = scores[dominant]

    if aqi <= 50:    cat = "good"
    elif aqi <= 100: cat = "moderate"
    elif aqi <= 150: cat = "unhealthy_sensitive"
    elif aqi <= 200: cat = "unhealthy"
    elif aqi <= 300: cat = "very_unhealthy"
    else:            cat = "hazardous"

    return aqi, cat, dominant


# ---------------------------------------------------------------------------
# Public API: get_ground_readings()
# ---------------------------------------------------------------------------

def get_ground_readings(scenario: str = "standard") -> List[dict]:
    """
    Simulate ground station readings for all 8 stations.

    In production:
      Replace with: OpenAQ API call
        GET https://api.openaq.org/v2/locations?limit=8&country=US&...
        or EPA AirNow API: https://www.airnowapi.org/aq/observation/...
      Or direct SCADA/telemetry pull from state AQMD data management system.

    Simulation:
      Each station has documented baseline concentrations calibrated to its type.
      Episode scenario multiplies SO2 and PM2.5 at industrial stations.
      Diurnal pattern applied: morning traffic peak, afternoon ozone peak.
      Day-to-day variation via seeded RNG (+-15%).
    """
    rng  = _rng(extra=10)
    scen = SCENARIOS.get(scenario, SCENARIOS["standard"])
    hour = datetime.now().hour
    ts   = datetime.now().isoformat(timespec="minutes")

    # Diurnal multipliers — matches published urban pollution diurnal profiles
    pm25_diurnal  = 1.3 if  6 <= hour < 10 else 1.0 if 10 <= hour < 16 else 1.2 if 16 <= hour < 20 else 0.8
    no2_diurnal   = 1.5 if  7 <= hour < 10 else 0.9 if 10 <= hour < 15 else 1.4 if 16 <= hour < 20 else 0.7
    o3_diurnal    = 0.4 if  hour < 10       else 1.6 if 11 <= hour < 17 else 1.0 if hour < 20          else 0.5
    co_diurnal    = 1.4 if  7 <= hour < 10  else 0.9 if 10 <= hour < 15 else 1.3 if 16 <= hour < 19 else 0.8
    so2_diurnal   = 1.2 if  8 <= hour < 18  else 0.8   # industrial hours

    episode_so2_mult  = scen["so2_multiplier"]
    episode_pm_offset = scen["pm25_baseline"] - 28.0  # additive offset from standard baseline

    readings = []
    for stn_id, stn in STATION_DEFINITIONS.items():
        var = rng.uniform(0.88, 1.14)   # day-to-day variation

        pm25 = round(max(2.0, (stn["pm25_base"] + episode_pm_offset * 0.6) * pm25_diurnal * var), 1)
        pm10 = round(max(5.0, stn["pm10_base"] * pm25_diurnal * var), 1)
        no2  = round(max(5.0, stn["no2_base"]  * no2_diurnal  * var * scen["no2_multiplier"]), 1)
        so2  = round(max(1.0, stn["so2_base"]  * so2_diurnal  * var * episode_so2_mult), 1)
        co   = round(max(0.3, stn["co_base"]   * co_diurnal   * var), 2)
        o3   = round(max(5.0, stn["o3_base"]   * o3_diurnal   * var), 1)

        # Stagnant episode: accumulate (no dispersion)
        if scen["met_wind_ms"] < 1.5:
            pm25 = round(pm25 * 1.6, 1)
            so2  = round(so2  * 1.8, 1)

        aqi, cat, dom = _compute_aqi(pm25, pm10, no2, so2, co, o3)

        # Met sensors at each station
        temp     = round(rng.gauss(18.0, 4.0), 1)
        humidity = round(rng.uniform(45, 80), 1)
        wind_spd = round(max(0.1, rng.gauss(scen["met_wind_ms"], 0.5)), 1)
        wind_dir = round(rng.uniform(180, 260), 0)   # SW prevailing in scenario
        pressure = round(rng.gauss(1013.0, 4.0), 1)

        dq = "valid"
        if pm25 > 300 or no2 > 600:
            dq = "suspect"   # extreme values flagged for QC

        readings.append({
            "station_id":         stn_id,
            "station_name":       stn["name"],
            "latitude":           stn["lat"],
            "longitude":          stn["lon"],
            "district":           stn["district"],
            "station_type":       stn["type"],
            "timestamp":          ts,
            "pm25_ug_m3":         pm25,
            "pm10_ug_m3":         pm10,
            "no2_ug_m3":          no2,
            "so2_ug_m3":          so2,
            "co_mg_m3":           co,
            "o3_ug_m3":           o3,
            "temperature_c":      temp,
            "humidity_percent":   humidity,
            "wind_speed_ms":      wind_spd,
            "wind_direction_deg": wind_dir,
            "pressure_hpa":       pressure,
            "aqi":                aqi,
            "aqi_category":       cat,
            "dominant_pollutant": dom,
            "data_quality_flag":  dq,
            "scenario":           scen["label"],
        })

    return readings


# ---------------------------------------------------------------------------
# Public API: get_satellite_observations()
# ---------------------------------------------------------------------------

def get_satellite_observations(scenario: str = "standard") -> List[dict]:
    """
    Simulate processed satellite data products for the monitoring region.

    In production:
      Sentinel-5P: ESA Copernicus Open Access Hub or Google Earth Engine
        import ee; ee.Initialize(); ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
      MODIS AOD: NASA LAADS DAAC
        https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/...
      Landsat-9: USGS EarthExplorer API
      Processing: ESA SNAP toolbox, Sen2Cor, or GEE Python client

    Simulation:
      AOD correlates with PM2.5 levels (standard: moderate, episode: high).
      NO2 tropospheric column elevated where ground NO2 is elevated.
      Fire detection triggered in agricultural zone in episode scenario.
      Plume detection uses simulated wind trajectory from industrial source.
    """
    rng  = _rng(extra=20)
    scen = SCENARIOS.get(scenario, SCENARIOS["standard"])
    ts   = datetime.now().isoformat(timespec="minutes")

    is_episode = (scenario == "episode")

    # Sentinel-5P TROPOMI
    s5p = {
        "observation_id":                    f"S5P-{date.today().strftime('%Y%m%d')}-001",
        "satellite":                         "Sentinel-5P",
        "overpass_time":                     ts,
        "spatial_resolution_m":              3500,
        "cloud_cover_percent":               round(rng.uniform(5, 25), 1),
        "no2_tropospheric_column_mol_m2":    round(rng.gauss(8.5e-5 if is_episode else 4.2e-5, 0.5e-5), 6),
        "so2_column_du":                     round(rng.gauss(12.0 if is_episode else 1.8, 0.5), 2),
        "aerosol_optical_depth":             round(rng.gauss(0.72 if is_episode else 0.31, 0.05), 3),
        "co_total_column_mol_m2":            round(rng.gauss(0.048 if is_episode else 0.028, 0.003), 4),
        "ch4_column_ppb":                    round(rng.gauss(1895.0, 8.0), 1),
        "active_fire_count":                 rng.randint(4, 9) if is_episode else rng.randint(0, 1),
        "fire_radiative_power_mw":           round(rng.uniform(85, 320) if is_episode else rng.uniform(0, 15), 1),
        "urban_heat_island_intensity_c":     round(rng.gauss(4.2, 0.8), 1),
        "wind_transport_direction_deg":      225.0,
        "plume_detected":                    True if is_episode else rng.random() < 0.3,
        "plume_origin_lat":                  40.7360 if is_episode else None,
        "plume_origin_lon":                  -73.9605 if is_episode else None,
        "plume_origin_description":          "Metro Chemical Works Unit A — SO2 plume confirmed" if is_episode else None,
        "pollution_hotspots":                _build_hotspots(rng, is_episode),
    }

    # MODIS-Terra
    modis = {
        "observation_id":                    f"MODIS-{date.today().strftime('%Y%m%d')}-001",
        "satellite":                         "MODIS-Terra",
        "overpass_time":                     ts,
        "spatial_resolution_m":              1000,
        "cloud_cover_percent":               round(rng.uniform(8, 30), 1),
        "no2_tropospheric_column_mol_m2":    None,
        "so2_column_du":                     None,
        "aerosol_optical_depth":             round(rng.gauss(0.68 if is_episode else 0.29, 0.06), 3),
        "co_total_column_mol_m2":            round(rng.gauss(0.045 if is_episode else 0.026, 0.003), 4),
        "ch4_column_ppb":                    None,
        "active_fire_count":                 rng.randint(3, 8) if is_episode else 0,
        "fire_radiative_power_mw":           round(rng.uniform(75, 280) if is_episode else 0, 1),
        "urban_heat_island_intensity_c":     round(rng.gauss(4.5, 0.9), 1),
        "wind_transport_direction_deg":      228.0,
        "plume_detected":                    is_episode,
        "plume_origin_lat":                  40.7680 if is_episode else None,
        "plume_origin_lon":                  -73.9720 if is_episode else None,
        "plume_origin_description":          "Agricultural burn plume — North Valley" if is_episode else None,
        "pollution_hotspots":                _build_hotspots(rng, is_episode, offset=5),
    }

    # Landsat-9 (land surface temperature + urban heat island)
    landsat = {
        "observation_id":                    f"L9-{date.today().strftime('%Y%m%d')}-001",
        "satellite":                         "Landsat-9",
        "overpass_time":                     ts,
        "spatial_resolution_m":              30,
        "cloud_cover_percent":               round(rng.uniform(5, 20), 1),
        "no2_tropospheric_column_mol_m2":    None,
        "so2_column_du":                     None,
        "aerosol_optical_depth":             round(rng.gauss(0.70 if is_episode else 0.30, 0.04), 3),
        "co_total_column_mol_m2":            None,
        "ch4_column_ppb":                    None,
        "active_fire_count":                 0,
        "fire_radiative_power_mw":           0.0,
        "urban_heat_island_intensity_c":     round(rng.gauss(5.1, 0.7), 1),
        "wind_transport_direction_deg":      222.0,
        "plume_detected":                    False,
        "plume_origin_lat":                  None,
        "plume_origin_lon":                  None,
        "plume_origin_description":          None,
        "pollution_hotspots":                [],
    }

    return [s5p, modis, landsat]


def _build_hotspots(rng: random.Random, episode: bool, offset: int = 0) -> List[dict]:
    """Build a list of pollution hotspot dicts consistent with the scenario."""
    base = [
        {"lat": 40.7360, "lon": -73.9605, "intensity_index": 9.2 if episode else 3.1,
         "primary_pollutant": "SO2", "source_name": "Metro Chemical Works Unit A"},
        {"lat": 40.7005, "lon": -74.0140, "intensity_index": 6.8 if episode else 2.4,
         "primary_pollutant": "NOx", "source_name": "Port Authority Terminal"},
        {"lat": 40.7128, "lon": -74.0060, "intensity_index": 5.5 if episode else 2.8,
         "primary_pollutant": "NO2", "source_name": "Downtown Traffic Corridor"},
        {"lat": 40.6750, "lon": -73.9450, "intensity_index": 4.9 if episode else 2.2,
         "primary_pollutant": "PM2.5", "source_name": "South Highway Corridor"},
    ]
    if episode:
        base.append(
            {"lat": 40.7680, "lon": -73.9720, "intensity_index": 8.5,
             "primary_pollutant": "PM2.5", "source_name": "Agricultural Burns — North Valley"}
        )
    for h in base:
        h["intensity_index"] = round(h["intensity_index"] + rng.gauss(0, 0.2 + offset * 0.01), 2)
    return base


# ---------------------------------------------------------------------------
# Public API: get_meteorological_data()
# ---------------------------------------------------------------------------

def get_meteorological_data(scenario: str = "standard") -> dict:
    """
    Simulate current meteorological conditions relevant to dispersion modeling.

    In production:
      NOAA HRRR (High-Resolution Rapid Refresh) model API:
        https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/
      NOAA NDFD REST: https://graphical.weather.gov/xml/rest.php
      Meteomatics API (commercial, 90m resolution)
      Vaisala WXT sensor at each monitoring station

    Key dispersion parameters explained:
      mixing_height_m:    Height of the atmospheric boundary layer.
                          Low (<500m) = trapped pollution near surface.
                          High (>2000m) = good vertical dispersion.
      stability_class:    Pasquill-Gifford stability classes A-F.
                          A = very unstable (good dispersion),
                          F = very stable (pollution traps, worst case).
      ventilation_coeff:  wind_speed * mixing_height. Values <6000 m2/s indicate
                          poor ventilation (pollution accumulation likely).
    """
    rng  = _rng(extra=30)
    scen = SCENARIOS.get(scenario, SCENARIOS["standard"])

    is_episode = (scenario == "episode")

    mixing_height = rng.gauss(280, 40) if is_episode else rng.gauss(1450, 150)
    wind_ms       = scen["met_wind_ms"]
    vent_coeff    = round(wind_ms * mixing_height, 0)

    return {
        "timestamp":                 datetime.now().isoformat(timespec="minutes"),
        "scenario":                  scen["label"],
        # Surface conditions
        "temperature_c":             round(rng.gauss(18.0, 3.0), 1),
        "humidity_percent":          round(rng.uniform(55 if is_episode else 40, 85), 1),
        "pressure_hpa":              round(rng.gauss(1025 if is_episode else 1008, 3), 1),
        "dew_point_c":               round(rng.gauss(10.0, 2.0), 1),
        # Wind
        "wind_speed_ms":             round(wind_ms + rng.gauss(0, 0.2), 1),
        "wind_direction_deg":        round(rng.gauss(225, 10), 0),
        "wind_gust_ms":              round(wind_ms * rng.uniform(1.2, 1.8), 1),
        "wind_direction_label":      "SW",
        # Vertical structure
        "mixing_height_m":           round(mixing_height, 0),
        "stability_class":           "F" if is_episode else "C",
        "stability_label":           scen["met_stability"],
        "ventilation_coefficient_m2_s": vent_coeff,
        # Precipitation / solar
        "precipitation_mm_hr":       0.0,
        "solar_radiation_w_m2":      round(rng.uniform(150, 350), 0),
        "uv_index":                  round(rng.uniform(3.0, 7.5), 1),
        "visibility_km":             round(rng.gauss(4.5 if is_episode else 12.0, 1.0), 1),
        # Forecast
        "forecast_12h":              "Stagnant conditions persisting — no frontal passage expected" if is_episode
                                     else "Moderate winds, gradual mixing improvement expected",
        # Derived dispersion assessment
        "dispersion_quality":        "very_poor" if vent_coeff < 3000
                                     else "poor" if vent_coeff < 6000
                                     else "moderate" if vent_coeff < 12000
                                     else "good",
        "pollution_accumulation_risk": "critical" if vent_coeff < 3000
                                        else "high" if vent_coeff < 6000
                                        else "moderate" if vent_coeff < 12000
                                        else "low",
    }


# ---------------------------------------------------------------------------
# Public API: get_emission_inventory()
# ---------------------------------------------------------------------------

def get_emission_inventory() -> List[dict]:
    """
    Return the known emission sources inventory for cross-referencing with sensor data.

    In production:
      US EPA NEI: https://api.epa.gov/FACT/1.0/
      EU E-PRTR: https://prtr.eea.europa.eu/
      State AQMD permit database (varies by jurisdiction)
      WRI Global Power Plant Database: https://datasets.wri.org/dataset/globalpowerplantdatabase
    """
    return [s.copy() for s in EMISSION_INVENTORY]


# ---------------------------------------------------------------------------
# Utility: scenario description
# ---------------------------------------------------------------------------

def describe_scenario(scenario: str) -> str:
    scen = SCENARIOS.get(scenario, SCENARIOS["standard"])
    return scen["description"]
