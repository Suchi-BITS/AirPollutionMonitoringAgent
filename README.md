# Air Pollution Monitoring Agent System

Multi-agent AI system built with LangGraph and LangChain that analyzes real-time
ground sensor data and satellite imagery to identify pollution sources and recommend
mitigation actions.

---

## Architecture

```
supervisor_init
    |
    +-- ground_sensor_agent          Reads all 8 ground stations (PM2.5, PM10, NO2, SO2, CO, O3, AQI)
    |
    +-- satellite_agent              Processes Sentinel-5P, MODIS-Terra, Landsat-9 imagery
    |
    +-- meteorological_agent         Assesses dispersion conditions, ventilation coefficient, stability
    |
    +-- source_identification_agent  Cross-references sensors + satellite + emission inventory
    |
    +-- health_impact_agent          Population exposure, clinical risk, hospital preparedness
    |
    +-- mitigation_agent             Generates prioritized action plan, logs recommendations
    |
    +-- alert_agent                  Issues public alerts, regulatory enforcement, hospital notifications
    |
supervisor_synthesis                 Compiles final situation report
```

---

## Quick Start

### Mode 1: Demo (no dependencies beyond stdlib)
```bash
python main.py standard
python main.py episode
```

### Mode 2: With LangGraph (no API key required)
```bash
pip install langgraph langchain langchain-core langchain-openai python-dotenv
python main.py standard
```

### Mode 3: With live LLM reasoning
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here
python main.py episode
```

---

## Scenarios

| Scenario   | Description |
|------------|-------------|
| `standard` | Typical weekday urban pollution. Moderate AQI across most districts. Traffic and industrial sources active. |
| `episode`  | Severe pollution event. Industrial facility SO2 release + agricultural burns + stagnant high-pressure system. AQI at hazardous levels downwind. |

---

## File Structure

```
air_pollution_agents/
    main.py                         Entry point
    requirements.txt
    .env.example
    config/
        settings.py                 All WHO guidelines, AQI thresholds, station list
    data/
        models.py                   Dataclass definitions + agent state factory
        simulation.py               Complete simulation layer (replaces real APIs in production)
    tools/
        sensor_tools.py             LangChain tools: ground sensors, satellite, met, inventory, health
        action_tools.py             LangChain tools: public alerts, regulatory actions, hospital notifications
    agents/
        base.py                     LLM call helpers with demo mode fallback
        all_agents.py               All 9 agent node functions
    graph/
        pollution_graph.py          LangGraph StateGraph definition + direct pipeline fallback
```

---

## Data Sources (simulation -> production replacement)

| Simulation Function          | Production Replacement |
|------------------------------|------------------------|
| `get_ground_readings()`      | OpenAQ Platform API / US EPA AirNow API / State AQMD SCADA |
| `get_satellite_observations()` | ESA Copernicus Sentinel-5P TROPOMI / NASA MODIS / USGS Landsat-9 via Google Earth Engine |
| `get_meteorological_data()`  | NOAA HRRR Model API / Meteomatics API / On-site met towers |
| `get_emission_inventory()`   | US EPA NEI API / EU E-PRTR / State AQMD permit database |
| `fetch_health_risk_tables()` | US Census Bureau API / City health department registry |

All production replacements are drop-in: swap the function body in `simulation.py` or
`tools/sensor_tools.py`. No changes required to agents or graph.

---

## WHO Air Quality Guidelines Applied

| Pollutant | WHO Guideline (24h) | Unhealthy Threshold |
|-----------|---------------------|---------------------|
| PM2.5     | 15 ug/m3            | 55.4 ug/m3          |
| PM10      | 45 ug/m3            | 154 ug/m3           |
| NO2       | 200 ug/m3 (1h)      | 360 ug/m3           |
| SO2       | 40 ug/m3            | 185 ug/m3           |
| CO        | 4 mg/m3 (8h)        | 15.4 mg/m3          |
| O3        | 100 ug/m3 (8h)      | 168 ug/m3           |

---

## Disclaimer

This system provides decision support for air quality management.
All mitigation orders and public health advisories must be reviewed and issued
by authorized regulatory personnel.
