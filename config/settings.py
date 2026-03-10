# config/settings.py
# All configuration for the Air Pollution Monitoring Agent System.
# Uses stdlib dataclass only — no Pydantic dependency.

import os
from dataclasses import dataclass, field
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class AirQualityConfig:

    # LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4o"
    temperature: float = 0.05

    # Network identity
    system_name: str = "AirGuard Pollution Monitoring System"
    city: str = "Metro City"
    region: str = "Central Metropolitan Region"

    # Ground station IDs managed by this network
    ground_stations: List[str] = field(default_factory=lambda: [
        "STN-001", "STN-002", "STN-003", "STN-004",
        "STN-005", "STN-006", "STN-007", "STN-008",
    ])

    # Satellite platforms tracked
    satellite_sources: List[str] = field(default_factory=lambda: [
        "Sentinel-5P", "MODIS-Terra", "Landsat-9",
    ])

    # WHO Air Quality Guidelines (2021) — all concentrations in ug/m3 unless noted
    # PM2.5
    pm25_who_24h: float = 15.0
    pm25_unhealthy_sensitive: float = 35.4
    pm25_unhealthy: float = 55.4
    pm25_very_unhealthy: float = 150.4
    pm25_hazardous: float = 250.4

    # PM10
    pm10_who_24h: float = 45.0
    pm10_unhealthy: float = 154.0
    pm10_hazardous: float = 424.0

    # NO2
    no2_who_1h: float = 200.0
    no2_unhealthy: float = 360.0

    # SO2
    so2_who_24h: float = 40.0
    so2_unhealthy: float = 185.0
    so2_hazardous: float = 604.0

    # CO (mg/m3)
    co_who_8h: float = 4.0
    co_unhealthy: float = 15.4
    co_hazardous: float = 30.4

    # O3
    o3_who_8h: float = 100.0
    o3_unhealthy: float = 168.0

    # US EPA AQI category breakpoints (AQI score)
    aqi_good: int = 50
    aqi_moderate: int = 100
    aqi_unhealthy_sensitive: int = 150
    aqi_unhealthy: int = 200
    aqi_very_unhealthy: int = 300
    aqi_hazardous: int = 500

    disclaimer: str = (
        "This system provides decision support for air quality management. "
        "All mitigation orders and public health advisories must be reviewed "
        "and issued by authorized regulatory personnel."
    )


air_config = AirQualityConfig()
