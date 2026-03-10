# data/models.py
# All data models use stdlib dataclasses — no Pydantic required.
# Agent state is a plain dict that flows through LangGraph unchanged.

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class GroundStationReading:
    """
    One polling cycle from a ground-based air quality monitoring station.

    Production source:
      - US EPA AirNow API / state regulatory network SCADA systems
      - OpenAQ platform (aggregates 100+ country networks)
      - Vaisala AQT Series sensors (road-side, compact)
      - Met One BAM-1022 Beta Attenuation Monitor (PM2.5/PM10 reference)
      - Thermo Fisher 42i NOx analyzer, 43i SO2 analyzer, 48i CO analyzer
      - Aeroqual O3 monitor
    All transmitted via 4G/cellular or fiber to central data management system.
    Typical data latency: 1 hour (regulatory) or 1 minute (real-time low-cost sensors).
    """
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    district: str
    station_type: str            # traffic | industrial | background | residential
    timestamp: str
    # Criteria pollutants
    pm25_ug_m3: float
    pm10_ug_m3: float
    no2_ug_m3: float
    so2_ug_m3: float
    co_mg_m3: float              # CO in mg/m3 (not ug/m3)
    o3_ug_m3: float
    # Co-located meteorological sensors
    temperature_c: float
    humidity_percent: float
    wind_speed_ms: float
    wind_direction_deg: float    # meteorological convention: 0=N, 90=E, 180=S, 270=W
    pressure_hpa: float
    # Computed fields
    aqi: int
    aqi_category: str            # good | moderate | unhealthy_sensitive | unhealthy | very_unhealthy | hazardous
    dominant_pollutant: str
    data_quality_flag: str       # valid | suspect | invalid


@dataclass
class SatelliteObservation:
    """
    Processed satellite data product for the monitoring region.

    Production sources:
      - Sentinel-5P TROPOMI: NO2, SO2, CO, O3, aerosol tropospheric columns
          Resolution: 3.5x5.5 km; daily global coverage
          API: ESA Copernicus Open Access Hub / Google Earth Engine
      - MODIS Terra/Aqua: AOD (aerosol optical depth), active fire detection
          Resolution: 500m (surface reflectance), 1km (AOD)
          API: NASA LAADS DAAC, NASA Worldview
      - Landsat-9: Land surface temperature, urban heat island mapping
          Resolution: 30m multispectral, 100m thermal
          API: USGS EarthExplorer, Google Earth Engine
      - VIIRS (Suomi-NPP): Nighttime lights (industrial activity proxy), fire detection
    All processed via Google Earth Engine or ESA SNAP toolbox.
    """
    observation_id: str
    satellite: str
    overpass_time: str
    spatial_resolution_m: int
    cloud_cover_percent: float
    # Sentinel-5P TROPOMI retrievals
    no2_tropospheric_column_mol_m2: float
    so2_column_du: float                 # Dobson Units
    aerosol_optical_depth: float         # 550nm AOD; correlates with PM2.5 column
    co_total_column_mol_m2: float
    ch4_column_ppb: float                # methane — agricultural/landfill indicator
    # MODIS fire products
    active_fire_count: int
    fire_radiative_power_mw: float
    # Landsat-derived
    urban_heat_island_intensity_c: float
    # Hotspot analysis results
    pollution_hotspots: List[Dict]       # [{lat, lon, intensity_index, primary_pollutant}]
    wind_transport_direction_deg: float
    plume_detected: bool
    plume_origin_lat: Optional[float]
    plume_origin_lon: Optional[float]
    plume_origin_description: Optional[str]


@dataclass
class PollutionSource:
    """
    An identified or suspected pollution emission source.
    Derived by cross-referencing ground measurements, satellite hotspots,
    emission inventory databases, and dispersion back-trajectory analysis.

    Production emission inventory sources:
      - US EPA National Emissions Inventory (NEI)
      - EU E-PRTR (European Pollutant Release and Transfer Register)
      - Global Power Plant Database (WRI)
      - OpenStreetMap industrial/commercial layer
      - State/regional air quality management district permit databases
    """
    source_id: str
    source_name: str
    source_category: str         # industrial | vehicular | construction | agricultural | natural | residential
    latitude: float
    longitude: float
    district: str
    primary_pollutants: List[str]
    emission_rate_kg_hr: Dict[str, float]
    operating_status: str        # active | intermittent | shutdown | unknown
    permit_holder: str
    permit_number: str
    compliance_status: str       # compliant | exceedance | permit_exceeded | no_permit
    last_inspection_date: str
    estimated_contribution_percent: float
    confidence: str              # high | medium | low
    detection_method: str        # satellite | ground_gradient | inventory | combined


@dataclass
class HealthImpactAssessment:
    """
    Estimated public health burden from current air quality levels.

    Methodology:
      - Exposure-response functions from WHO Global Burden of Disease study
      - Concentration-response coefficients from Pope & Dockery (2006) for PM2.5
      - Population data from census district overlays
      - Sensitive population registry (asthma, COPD, cardiovascular disease lists
        held by city health department)
    """
    assessment_id: str
    timestamp: str
    affected_districts: List[str]
    current_aqi: int
    aqi_category: str
    population_exposed: int
    sensitive_population_count: int
    # Risk estimates per 100,000 population per day of exposure at current levels
    respiratory_cases_per_100k: float
    cardiovascular_cases_per_100k: float
    mortality_risk_per_million: float
    groups_at_risk: List[str]
    health_advisories: List[str]
    hospital_alert_level: str    # normal | elevated | high | critical


@dataclass
class MitigationRecommendation:
    """
    A specific recommended action to reduce pollution levels or protect public health.
    """
    recommendation_id: str
    priority: str                # emergency | high | medium | low
    category: str                # regulatory | operational | public_health | infrastructure
    target_entity: str           # who implements: regulator | facility_operator | city_transport | public
    title: str
    description: str
    expected_aqi_reduction: float
    implementation_timeline: str
    regulatory_basis: str
    estimated_cost_tier: str     # negligible | low | medium | high | very_high
    co_benefits: List[str]


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

def make_agent_state() -> dict:
    """
    Create the empty agent state dict that flows through the LangGraph pipeline.
    All agents read from and write back to this dict.
    """
    return {
        # Inputs
        "target_region":               None,
        "analysis_timestamp":          None,

        # Raw data collected by sensor / satellite agents
        "ground_readings":             [],
        "satellite_observations":      [],
        "meteorological_summary":      None,

        # Identified sources
        "pollution_sources":           [],

        # Analysis strings produced by each monitoring agent
        "ground_analysis":             None,
        "satellite_analysis":          None,
        "source_analysis":             None,
        "met_analysis":                None,
        "health_analysis":             None,
        "dispersion_analysis":         None,

        # Structured assessments
        "health_impact":               None,
        "risk_scores":                 [],

        # Outputs from action agents
        "mitigation_recommendations":  [],
        "public_alerts":               [],
        "regulatory_actions":          [],

        # Final report
        "situation_report":            None,

        # Control
        "current_agent":               "supervisor",
        "iteration_count":             0,
        "emergency_triggered":         False,
        "errors":                      [],
    }
