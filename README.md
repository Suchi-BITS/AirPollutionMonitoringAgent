# AirGuard Air Pollution Monitoring Agent System v4

A complete multi-agent AI system built with LangGraph and LangChain that continuously
monitors urban air quality through ground sensor networks and satellite imagery,
identifies pollution sources, assesses public health risk, and generates prioritized
enforcement and mitigation actions. Version 4 adds a ReAct (Reason + Act) agent
loop, context engineering with RAG-grounded regulatory knowledge, in-session
short-term memory with alert deduplication and trend tracking, and MCP integrations
with Gmail and Google Calendar for external notification and incident tracking.

The system runs fully end-to-end in demo mode with no external dependencies, and
upgrades automatically to live GPT-4o reasoning when an OpenAI API key is present.

---

## Table of Contents

1. [Use Case and Problem Statement](#use-case-and-problem-statement)
2. [System Architecture](#system-architecture)
3. [Version History and What Changed](#version-history)
4. [Architectural Pattern — ReAct Agent](#architectural-pattern)
5. [Context Engineering](#context-engineering)
6. [Short-Term Memory](#short-term-memory)
7. [RAG Knowledge Base](#rag-knowledge-base)
8. [MCP Integrations](#mcp-integrations)
9. [Agent Graph Topology](#agent-graph-topology)
10. [Step-by-Step Workflow](#step-by-step-workflow)
11. [File-by-File Explanation](#file-by-file-explanation)
12. [Data Layer and Simulation Design](#data-layer-and-simulation-design)
13. [Ground Station Network](#ground-station-network)
14. [Pollution Source Inventory](#pollution-source-inventory)
15. [WHO Guidelines and AQI Framework](#who-guidelines-and-aqi-framework)
16. [Tool Reference](#tool-reference)
17. [Production Deployment Guide](#production-deployment-guide)
18. [Run Modes and Quick Start](#run-modes-and-quick-start)
19. [Sample Output](#sample-output)
20. [Disclaimer](#disclaimer)

---

## Use Case and Problem Statement

### The Domain

Urban air quality management requires simultaneous analysis of multiple data streams:
ground-level sensor networks, satellite imagery, meteorological models, emission
inventories, and regulatory compliance records. The gap between raw data availability
and actionable enforcement and public health decisions is where this system operates.

Air quality episodes — periods of hazardous pollution levels — require coordinated
responses across several authorities: the Air Quality Management District (enforcement),
city transport (traffic restrictions), hospital networks (surge capacity preparation),
and the public (health advisories). These responses must be timely, legally grounded
in verified regulations, and non-redundant across a multi-hour monitoring session
that may involve dozens of monitoring cycles.

### The Problem

Four specific gaps make this an appropriate agentic AI problem.

First, the multi-domain nature of pollution analysis — sensor data interpretation,
satellite imagery analysis, atmospheric dispersion modeling, source attribution, and
health impact quantification — requires reasoning that spans five technical domains
simultaneously. No rule-based system handles all five correctly across the full range
of episode types.

Second, a real monitoring session involves multiple analysis cycles (every 30 minutes)
and the agent must not repeat the same public alert or regulatory order on every
cycle. Alert deduplication across cycles is a hard requirement.

Third, enforcement actions must be grounded in specific, verifiable regulatory
thresholds. An LLM that cites an incorrect permit limit (e.g., SO2 must be below 30
ug/m3 when the applicable rule specifies 100 mg/m3) produces a legally invalid
enforcement action. RAG retrieval of verified regulatory documents is required.

Fourth, when emergency orders are issued, human authorities must be notified
immediately through official communication channels — not just logged in the system.

### What This System Delivers

Each monitoring cycle produces: a complete network AQI assessment, source attribution
with compliance status, health impact quantification, a memory-deduplicated set of
public alerts and regulatory enforcement actions, RAG-grounded regulatory basis for
each enforcement action, and Gmail and Google Calendar MCP integrations for official
notifications and incident tracking.

---

## System Architecture

```
+------------------------------------------------------------------+
|  DATA LAYER                                                      |
|  fetch_ground_sensor_data()  OpenAQ / EPA AirNow API             |
|  fetch_satellite_imagery()   Sentinel-5P / MODIS / Landsat-9     |
|  fetch_meteorological_data() NOAA HRRR / Meteomatics API         |
|  fetch_emission_inventory()  EPA NEI / State AQMD permit DB      |
|  fetch_health_risk_tables()  Census + FHIR sensitive pop. list   |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  SHORT-TERM MEMORY (memory/short_term_memory.py)                |
|  AQI rolling window (12 passes), trend direction                |
|  Alert deduplication registry (type + district + severity)      |
|  Regulatory action log (source_id + action_type)                |
|  Deferred action tracker                                        |
|  Episode metadata (declared, start time, peak AQI)             |
|  context_summary() -> text injected into every system prompt    |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  CONTEXT ENGINEER (context/context_engineer.py)                 |
|  build_system_prompt() — dynamic per run: memory + RAG + guidance|
|  compress_tool_result() — ~80% token reduction per tool call    |
|  trim_message_history() — keeps context within token budget     |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  RAG KNOWLEDGE BASE (rag/knowledge_base.py)                     |
|  22 regulatory/health document chunks:                          |
|    WHO AQG 2021 (PM2.5, PM10, NO2, SO2, O3, CO)                |
|    US EPA NAAQS and AQI breakpoints                             |
|    State AQMD Rules 1001, 2-1, 4-12                             |
|    Metro City Episode Plan Stage I/II/III + de-escalation       |
|    PM2.5 and O3 concentration-response functions                |
|    Hospital surge capacity guidance                             |
|    Historical episode lessons learned (2019, 2022)              |
|  Retrieved and injected into system prompt each run             |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  REACT AGENT LOOP (agents/react_agent.py)                       |
|  16 tools: 5 observation + 6 action + 5 knowledge/MCP          |
|  LLM.bind_tools(ALL_TOOLS) — decides tool call sequence         |
|  Every tool result compressed before entering message history   |
|  Message history trimmed before each LLM call                  |
|  Memory gate checked before each alert/enforcement dispatch     |
|  _post_run_memory_update() — updates SESSION_MEMORY after run  |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  MCP INTEGRATIONS (mcp/mcp_client.py, mcp/mcp_tools.py)        |
|  Gmail MCP:           send_incident_notification_email          |
|                       check_compliance_inbox                    |
|  Google Calendar MCP: create_response_calendar_event           |
|                       get_compliance_deadlines                  |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  OUTPUT LAYER                                                    |
|  7-section situation report                                      |
|  Public health alerts (memory-deduplicated)                     |
|  Regulatory enforcement actions (memory-deduplicated)           |
|  Hospital network notifications                                 |
|  Traffic restriction requests                                   |
|  Gmail notifications to enforcement officials                   |
|  Google Calendar incident timeline events                       |
+------------------------------------------------------------------+
```

---

## Version History

### v1 — Linear Pipeline (9 Sequential Agents)

The original architecture used a strictly sequential LangGraph pipeline: supervisor
init → ground sensor agent → satellite agent → meteorological agent → source
identification agent → health impact agent → mitigation agent → alert agent →
supervisor synthesis → END. All 9 agents ran on every cycle regardless of severity.
No memory, no context engineering, no RAG, no external integrations.

### v2 — ReAct Pattern

The 9-agent linear pipeline was replaced with a single ReAct (Reason + Act) agent
loop. One LLM bound to all 11 tools (5 observation + 6 action) decides which tools
to call and in what order based on what it observes. The linear `add_edge()` graph
was replaced with a single `add_conditional_edges()` loop: tool_calls present →
execute tools → loop; no tool_calls → END. The agent calls only the tools relevant
to the current situation rather than running all 9 agents unconditionally.

### v3 — Context Engineering + Short-Term Memory

Short-term memory (`memory/short_term_memory.py`) was added to persist AQI trend data,
alert dispatch history, and regulatory action records across monitoring cycles. The
context engineer (`context/context_engineer.py`) was added to: rebuild the system
prompt dynamically each run with live memory state injected, compress raw tool
results ~80% before they enter the message history, and trim the message history
when it exceeds the token budget. The memory gate prevents duplicate alert dispatches
and supports escalation (advisory → warning → emergency allowed; same severity → suppressed).

### v4 — RAG + MCP (Current Version)

The RAG knowledge base (`rag/knowledge_base.py`) was added with 22 curated regulatory
and health documents. The RAG retriever (`rag/rag_retriever.py`) automatically builds
a context block injected into every system prompt, and exposes `query_knowledge_base`
as an on-demand tool. The MCP client (`mcp/mcp_client.py`) integrates the Anthropic
API's MCP server protocol with connected Gmail and Google Calendar servers. Five new
tools were added to `ALL_TOOLS`: `query_knowledge_base`, `send_incident_notification_
email`, `check_compliance_inbox`, `create_response_calendar_event`,
`get_compliance_deadlines`. Total tools: 16.

---

## Architectural Pattern

### ReAct Agent

The ReAct (Reason + Act) pattern interleaves Thought, Action, and Observation steps.
The LLM generates reasoning text, decides which tool to call next based on what it
has observed, receives the tool result as a compressed observation, and continues
until it determines no more tool calls are needed and produces a final response.

This is appropriate for air quality monitoring because:

The sequence of observations matters and varies by situation. In a standard day the
agent might observe ground sensors (AQI=140), confirm with satellite (low AOD), check
meteorology (moderate dispersion), and stop — no emergency actions needed. In an
episode day it observes ground sensors (AQI=487), sees satellite confirm a plume
from the chemical works, checks meteorology (very poor dispersion), queries RAG for
the Stage III episode protocol, issues an emergency shutdown order, sends a Gmail
notification, and creates a calendar event. The ReAct agent adapts its tool call
sequence to the situation; the linear 9-agent pipeline cannot.

The agent does not waste calls on irrelevant tools. If no sources are non-compliant,
the agent does not call `issue_regulatory_action`. If the calendar shows no pending
deadlines, the agent does not need to act on them. The linear pipeline would have
called all 9 agents regardless.

---

## Context Engineering

Context engineering controls what enters the LLM's context window at each iteration.
Without it, three problems arise as the ReAct loop runs many tool call iterations:

**Token noise**: Raw ground sensor data is 2,500 tokens per call (8 stations × 20
fields). After 5 tool calls, the message history contains 12,000 tokens of raw JSON
that the LLM processes again on every subsequent call. Most of those tokens are
irrelevant to the current decision.

**Staleness**: By iteration 10, the LLM is re-reading its own AQI reasoning from 8
tool calls ago. That context is dead weight and can confuse current decisions.

**Lost emphasis**: AQI=500 and SO2 at 22× the permit limit can be buried in 3,000
tokens of raw JSON. The LLM may not weight the signal appropriately when choosing
the next action.

The context engineer solves all three:

**Compression** (`compress_tool_result()`): Each tool result is compressed before
entering the message history. Ground sensor data: 2,500 → 300 tokens (88% reduction).
Satellite data: 1,800 → 250 tokens (86%). Emission inventory: 2,000 → 400 tokens
(80%). Only decision-relevant fields are preserved.

**Dynamic system prompt** (`build_system_prompt()`): The system prompt is rebuilt
every run, injecting the live memory state, RAG-retrieved regulatory documents, and
severity-calibrated response guidance. The LLM starts each run knowing what happened
last run, what regulations apply, and what level of response is expected.

**History trimming** (`trim_message_history()`): When the message history exceeds 14
messages, stale tool exchange blocks are collapsed into a compact one-line summary.
The LLM always sees its most recent reasoning verbatim while old observations are
summarised and shed.

---

## Short-Term Memory

The `ShortTermMemory` class (`memory/short_term_memory.py`) persists state across
all calls to `run_react_agent()` within a Python process (a monitoring session).

### AQI Trend Window

A rolling list of up to 12 `AQISnapshot` objects, one per monitoring cycle. Each
snapshot records: max AQI, average AQI, dominant pollutant, emergency flag, critical
station count, and dispersion quality. `get_aqi_trend()` computes direction (worsening
if delta > 5 pts, improving if delta < -5 pts, stable otherwise), consecutive
unhealthy run count, and session peak AQI. These values are injected into every system
prompt via `context_summary()`.

### Alert Deduplication

Every dispatched alert is recorded in `issued_alerts` with type, severity, district,
and timestamp. Before any alert or regulatory action is issued, `can_issue_alert()` or
`can_issue_regulatory_action()` is called. The logic: same severity → suppress;
strictly higher severity → allow (escalation). This prevents the agent from issuing
the same EMERGENCY public alert 5 times across 5 monitoring cycles while still allowing
it to escalate from ADVISORY to WARNING to EMERGENCY as conditions worsen.

### Episode Tracking

When `emergency_triggered=True` appears in state for the first time, the memory
records the episode start timestamp and peak AQI. Subsequent cycles update the peak
if a new maximum is reached. The `*** EPISODE DECLARED` annotation appears in the
system prompt context block from that point forward.

### Deferred Action Tracker

Mitigation recommendations logged via `log_mitigation_recommendation()` are recorded
in `deferred_actions`. The context summary reports how many deferred items are still
pending, prompting the LLM to check whether prior recommendations were implemented
before logging new ones.

---

## RAG Knowledge Base

The knowledge base (`rag/knowledge_base.py`) contains 22 curated document chunks
across 6 categories. Retrieval uses pure-Python TF-IDF cosine similarity.

| Category | Documents | Coverage |
|---|---|---|
| `who_guidelines` | WHO-01 to WHO-06 | 2021 guidelines for PM2.5, PM10, NO2, SO2, O3, CO |
| `naaqs` | EPA-01 to EPA-03 | US EPA NAAQS standards + AQI breakpoints |
| `state_regulation` | AQMD-01 to AQMD-03 | AQMD Rules 1001, 2-1, 4-12 |
| `episode_plan` | EPPLAN-01 to EPPLAN-04 | Stage I/II/III triggers + de-escalation |
| `health_tables` | HEALTH-01 to HEALTH-03 | PM2.5/O3 C-R functions + hospital surge |
| `historical_episodes` | HIST-01 to HIST-03 | 2019 industrial, 2022 ozone lessons |

The `RAGRetriever` (`rag/rag_retriever.py`) builds an automatic context block before
each run by retrieving the most relevant chunks for the current AQI level, dominant
pollutant, and episode stage. This block is injected into the system prompt. The LLM
can also call `query_knowledge_base()` during the ReAct loop for on-demand lookups
before citing specific threshold values.

---

## MCP Integrations

The MCP client (`mcp/mcp_client.py`) calls the Anthropic `/v1/messages` API with the
`mcp_servers` parameter, routing specific requests to the connected Gmail and Google
Calendar MCP servers.

| Tool | MCP Server | Trigger Condition |
|---|---|---|
| `send_incident_notification_email` | Gmail | After emergency shutdown order issued |
| `check_compliance_inbox` | Gmail | Start of run 2+ to check inspector reports |
| `create_response_calendar_event` | Google Calendar | Episode declaration or compliance deadline |
| `get_compliance_deadlines` | Google Calendar | Run 2+ to check outstanding enforcement deadlines |
| `query_knowledge_base` | RAG (no MCP) | Before citing any regulatory threshold |

In demo mode (no Anthropic API key), all MCP calls return structured mock responses
so the pipeline runs end-to-end. In live mode, Gmail and Google Calendar operations
execute against the connected accounts.

---

## Agent Graph Topology

```
START
  |
  v
inject_context_node
  | CONTEXT_ENGINEER.build_context_for_run()
  | RAG_RETRIEVER.build_rag_context()
  | SESSION_MEMORY.context_summary()
  | Builds: dynamic system prompt with memory + RAG + severity guidance
  |
  v
react_agent_node
  | CONTEXT_ENGINEER.trim_message_history()
  | LLM.bind_tools(ALL_TOOLS[16]).invoke(trimmed_messages)
  |
  v
should_continue()
  | tool_calls present AND iteration < 20 -> "tools"
  | no tool_calls OR max iterations -> "end"
  |
  +-- "tools" --> compressed_tool_node
  |               CONTEXT_ENGINEER.compress_tool_result()
  |               Execute tool, compress result, add ToolMessage
  |               -> react_agent_node (loop)
  |
  +-- "end"   --> END
                  _post_run_memory_update()
                  SESSION_MEMORY.record_run()
```

---

## Step-by-Step Workflow

### Step 1: Context Injection

Before the ReAct loop starts, the context engineer calls `build_context_for_run()`,
which: reads the current session memory state, retrieves RAG documents relevant to
the last known AQI and dominant pollutant, selects the severity-calibrated response
guidance tier (standard/high/emergency), and assembles the complete dynamic system
prompt. The LLM receives a system prompt that includes verified regulatory thresholds
and the session's alert history before making any tool calls.

### Step 2: Observation Phase (ReAct Iterations 1-5)

The LLM's first five tool calls are typically the five observation tools. It calls
`fetch_ground_sensor_data()` first (always), then satellite, meteorological, inventory,
and health tables. Each tool result is compressed by the context engineer before being
appended to the message history. The LLM sees: network max AQI, critical station list,
satellite AOD, plume detection flags, ventilation coefficient, dispersion quality,
non-compliant source count, and population exposure data — all in compact, signal-dense
format.

### Step 3: Reasoning and Knowledge Retrieval

After the observations, the LLM reasons about what actions are required. If it needs
to verify a regulatory threshold before citing it in an enforcement action, it calls
`query_knowledge_base()` with a specific query (e.g., "State AQMD Rule 1001 SO2 emission
limit for industrial sources"). The retrieved document chunk confirms the exact threshold
before the enforcement action is constructed.

### Step 4: Action Phase (Memory-Gated)

Before issuing any alert or regulatory action, the session memory gate is checked.
In the live LLM path the memory check runs in `_post_run_memory_update()` after the
run. In the demo path the memory gate is checked inline before each `_call()` — if
`SESSION_MEMORY.can_issue_alert()` or `SESSION_MEMORY.can_issue_regulatory_action()`
returns False, the action is skipped and a `[Memory gate] SUPPRESS` message is printed.

### Step 5: MCP External Integrations

After core enforcement actions, the LLM (or demo path) calls `check_compliance_inbox()`
to check whether inspectors have filed field reports confirming prior orders were
implemented. If an emergency order was issued, `send_incident_notification_email()`
notifies the AQMD enforcement director. `create_response_calendar_event()` timestamps
the incident declaration in the AirGuard Incident Tracking calendar. On run 2+,
`get_compliance_deadlines()` retrieves upcoming enforcement deadlines.

### Step 6: Situation Report

The LLM produces a 7-section situation report covering: executive summary, current air
quality status, source attribution, health impact, meteorological assessment, actions
taken (including RAG lookup count and MCP calls made), and outstanding items. The
`_post_run_memory_update()` function is called to update the session memory with this
run's snapshot and dispatched alerts.

---

## File-by-File Explanation

### main.py

Entry point. Accepts `scenario` (standard or episode) and `--runs N` arguments.
`run_monitoring_loop()` calls `run_react_agent()` N times with a configurable delay
between cycles, demonstrating memory deduplication and trend detection. After all
cycles, the final session memory summary and AQI history bar chart are printed.
`--reset-memory` clears the session memory.

### config/settings.py

`AirQualityConfig` stdlib dataclass. All WHO and EPA thresholds, AQI breakpoints,
station list, satellite sources, and disclaimer. Single `air_config` singleton.

### data/simulation.py

Complete simulation layer for all data streams. Date-seeded `_rng(extra)` for
reproducibility. Two scenario profiles: standard (moderate urban day) and episode
(industrial accident + stagnant weather). Eight station definitions with type-specific
baseline concentrations. Full US EPA AQI computation with all six pollutant breakpoint
tables. Physical self-consistency between ground readings, satellite AOD, and
meteorological ventilation coefficient.

### data/models.py

Stdlib dataclasses for all domain objects: `GroundStationReading`, `SatelliteObservation`,
`PollutionSource`, `HealthImpactAssessment`, `MitigationRecommendation`. Detailed
docstrings document production data source and instrument specifics. `make_agent_state()`
factory function for the initial empty state dict.

### tools/sensor_tools.py

Five LangChain `@tool` functions wrapping the simulation layer. Each tool adds
network summary statistics (max AQI, critical station list, etc.) on top of the raw
simulation data. Production replacement points documented in docstrings.

### tools/action_tools.py

Six LangChain `@tool` functions for issuing operational outputs. All append to an
in-memory `_action_log`. In production: public health alerts dispatch via Everbridge
or Twilio; regulatory actions write to the AQMD enforcement database; hospital
notifications dispatch via HL7 FHIR alerts to hospital EHR systems; traffic
restrictions submit to the city ATMS via API.

### memory/short_term_memory.py

`ShortTermMemory` class with `AQISnapshot`, `IssuedAlert`, and `DeferredAction`
dataclasses. Methods: `record_run()`, `record_alert()`, `record_deferred_action()`,
`can_issue_alert()`, `can_issue_regulatory_action()`, `get_aqi_trend()`,
`get_alert_history_summary()`, `context_summary()`, `reset()`. Module-level
`SESSION_MEMORY` singleton.

### context/context_engineer.py

`ContextEngineer` class. `build_system_prompt()` now accepts a `rag_context` parameter
and includes the new MCP tool descriptions. `compress_tool_result()` dispatches to
five tool-specific compressors. `trim_message_history()` collapses stale blocks into
one-line summaries. `build_context_for_run()` calls `RAG_RETRIEVER.build_rag_context()`
and injects the result before building the system prompt. Module-level `CONTEXT_ENGINEER`
singleton.

### rag/knowledge_base.py

`Chunk` dataclass. 22-chunk `CORPUS` across 6 categories. `KnowledgeBase` pure-Python
TF-IDF vector store with `_build()`, `retrieve(query, top_k, category)`, and
`retrieve_text()`. Module-level `KNOWLEDGE_BASE` singleton.

### rag/rag_retriever.py

`RAGRetriever` class. `build_rag_context()` auto-retrieves 4-6 chunks per run based
on AQI, dominant pollutant, and episode stage. `retrieve_for_enforcement()` grounds
regulatory basis strings. `on_demand_query()` backs the `query_knowledge_base` tool.
Module-level `RAG_RETRIEVER` singleton.

### mcp/mcp_client.py

`MCPClient` class. `send_incident_email()`, `get_recent_incident_emails()`,
`create_incident_calendar_event()`, `get_upcoming_compliance_deadlines()`. All methods
detect demo mode and return structured mock responses when no Anthropic API key is
present. `_call_anthropic_with_mcp()` sends requests to the Anthropic API with
`mcp_servers` and `anthropic-beta: mcp-client-2025-04-04` header. Module-level
`MCP_CLIENT` singleton.

### mcp/mcp_tools.py

Five LangChain `@tool` wrappers: `send_incident_notification_email`,
`check_compliance_inbox`, `create_response_calendar_event`, `get_compliance_deadlines`,
`query_knowledge_base`. `MCP_AND_RAG_TOOLS` export list. All tools print a one-line
status line to console showing whether they ran in demo or live MCP mode.

### agents/react_agent.py

`ALL_TOOLS` list: 16 tools (5 observation + 6 action + 5 knowledge/MCP).
`run_react_agent()` entry point: calls `CONTEXT_ENGINEER.build_context_for_run()`,
routes to `_live_react_run()` or `_demo_react_run()`. Live path: full LangChain
`bind_tools()` ReAct loop with compression and trimming. Demo path: deterministic
6-phase execution (observe, reason, mitigations, public alerts, regulatory, MCP/RAG)
with memory gate inline. `_post_run_memory_update()` called after both paths.

### graph/react_graph.py

Three nodes: `inject_context_node` (dynamic system prompt), `react_agent_node` (LLM
call with trimming), `compressed_tool_node` (tool execution with compression).
`should_continue()` conditional edge. `build_react_graph()` compiles the graph.
`run_react_pipeline()` fallback without LangGraph.

---

## Data Layer and Simulation Design

The simulation is physically self-consistent across all four data streams. In the
episode scenario: ground PM2.5 reaches 150 to 200 ug/m3 (hazardous); satellite AOD
reaches 0.68 to 0.75 (consistent with column-integrated PM2.5 at that concentration);
fire count reaches 7 to 17 across platforms with high fire radiative power (consistent
with multiple simultaneously burning agricultural fields); meteorological VC drops to
200 to 260 m2/s (well below the Stage III trigger of any positive number, reflecting
class F stability and 0.8 m/s wind).

---

## Ground Station Network

| Station ID | District | Type | Key Pollutants |
|---|---|---|---|
| STN-001 | Downtown Core | traffic | NO2, CO |
| STN-002 | East Industrial District | industrial | SO2, PM10 |
| STN-003 | Riverside West | residential | PM2.5, NO2 |
| STN-004 | North Campus | background | baseline reference |
| STN-005 | Port District | industrial | SO2, PM10 |
| STN-006 | North Suburbs | residential | PM2.5 |
| STN-007 | South Highway Corridor | traffic | NO2, CO |
| STN-008 | East Industrial (fence line) | industrial | SO2, PM2.5 |

---

## Pollution Source Inventory

| Source ID | Name | Category | Compliance |
|---|---|---|---|
| SRC-001 | Metro Chemical Works Unit A | industrial | exceedance |
| SRC-002 | Port Authority Container Terminal | industrial | compliant |
| SRC-003 | Downtown Traffic Corridor (I-495) | vehicular | compliant |
| SRC-004 | North Valley Agricultural Zone Burns | agricultural | permit_exceeded |
| SRC-005 | South Highway Diesel Fleet Corridor | vehicular | compliant |
| SRC-006 | Metro Power Station Unit 2 | industrial | compliant |
| SRC-007 | Riverside Construction Site Phase 3 | construction | compliant |

---

## WHO Guidelines and AQI Framework

| Pollutant | WHO Limit | Unhealthy Threshold | Hazardous |
|---|---|---|---|
| PM2.5 | 15 ug/m3 (24h) | 55.4 ug/m3 | 250.4 ug/m3 |
| PM10 | 45 ug/m3 (24h) | 154 ug/m3 | 424 ug/m3 |
| NO2 | 200 ug/m3 (1h) | 360 ug/m3 | 1249 ug/m3 |
| SO2 | 40 ug/m3 (24h) | 185 ug/m3 | 604 ug/m3 |
| CO | 4 mg/m3 (8h) | 15.4 mg/m3 | 30.4 mg/m3 |
| O3 | 100 ug/m3 (8h) | 168 ug/m3 | 374 ug/m3 |

AQI categories: Good (0-50), Moderate (51-100), Unhealthy for Sensitive (101-150),
Unhealthy (151-200), Very Unhealthy (201-300), Hazardous (301-500).

---

## Tool Reference

### Observation Tools

| Tool | Arguments | Returns |
|---|---|---|
| `fetch_ground_sensor_data` | `scenario: str` | Stations list, network AQI summary |
| `fetch_satellite_imagery` | `scenario: str` | AOD, fire count, plume detections |
| `fetch_meteorological_data` | `scenario: str` | Full met dict with VC and stability |
| `fetch_emission_inventory` | none | Sources list, non-compliant subset |
| `fetch_health_risk_tables` | none | District population, C-R coefficients |

### Action Tools

| Tool | Key Arguments | Returns |
|---|---|---|
| `issue_public_health_alert` | severity, districts, aqi, channels | alert_id |
| `issue_regulatory_action` | source_id, action_type, pollutant | case_number |
| `log_mitigation_recommendation` | priority, title, aqi_reduction | recommendation_id |
| `notify_hospital_network` | alert_level, expected_volume_increase_pct | notification_id |
| `request_traffic_restriction` | restriction_type, zones, times | reference |
| `get_action_log` | limit: int | list of records |

### Knowledge and MCP Tools

| Tool | Backend | When Called |
|---|---|---|
| `query_knowledge_base` | RAG TF-IDF | Before citing regulatory threshold |
| `send_incident_notification_email` | Gmail MCP | After emergency shutdown order |
| `check_compliance_inbox` | Gmail MCP | Run 2+ to check inspector reports |
| `create_response_calendar_event` | Google Calendar MCP | Episode declaration |
| `get_compliance_deadlines` | Google Calendar MCP | Run 2+ deadline check |

---

## Production Deployment Guide

### Ground Sensor Data

Replace `fetch_ground_sensor_data()` with OpenAQ Platform API
(`GET https://api.openaq.org/v2/locations`) or US EPA AirNow API
(`GET https://www.airnowapi.org/aq/observation/`) for US deployments.

### Satellite Imagery

Replace `fetch_satellite_imagery()` with Google Earth Engine Python API.
Collections: `COPERNICUS/S5P/OFFL/L3_NO2`, `COPERNICUS/S5P/OFFL/L3_AER_AI`,
`MODIS/006/MOD04_3K` for AOD, `MODIS/006/MOD14A1` for active fires.

### Meteorological Data

Replace `fetch_meteorological_data()` with NOAA HRRR model output (3km, hourly,
continental US) or Meteomatics API for global coverage including mixing height and
stability class.

### Emission Inventory

Replace `fetch_emission_inventory()` with US EPA NEI API (`api.epa.gov/FACT/1.0`)
for annual inventory data plus state AQMD permit database for real-time compliance.

### MCP API Keys

Set `ANTHROPIC_API_KEY=sk-ant-...` in `.env` for live Gmail and Google Calendar
MCP calls. The Gmail and Google Calendar servers are already connected at
`gmail.mcp.claude.com` and `gcal.mcp.claude.com` for Claude.ai users.

### RAG Vector Store

Replace `rag/knowledge_base.KnowledgeBase` with ChromaDB backed by
`sentence-transformers/all-MiniLM-L6-v2` for production semantic retrieval.
The `retrieve(query, top_k, category)` interface is unchanged.

---

## Run Modes and Quick Start

### Demo Mode — Single Run

```bash
cd airguard_v4
python main.py standard
python main.py episode
```

### Demo Mode — Multi-Run Monitoring Loop (demonstrates memory)

```bash
python main.py standard --runs 3
python main.py episode  --runs 2
```

Pass 1 establishes baseline and issues all alerts. Pass 2 shows memory context
injected in system prompt, deduplication suppressing repeat alerts, and MCP deadline
check. Pass 3 shows sustained episode detection if AQI remains above 150.

### Live LLM Mode

```bash
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY=sk-your-key to .env
python main.py episode --runs 2
```

### Reset Session Memory

```bash
python main.py --reset-memory
```

---

## Sample Output

```
AirGuard Pollution Monitoring System
Metro City | Central Metropolitan Region
Run #1 | Scenario: STANDARD | Mode: DEMO (no API key)
Context tokens: ~1,650

[RAG] Context injected: 3,391 chars | AQI=None, dominant=PM2.5, stage=0

[REACTIVE AGENT] Phase 1: Observation (compressed)
  fetch_ground_sensor_data  -> max_aqi=142, critical=['STN-007','STN-008']
  fetch_satellite_imagery   -> max_aod=0.328, fires=0, plumes=0
  fetch_meteorological_data -> VC=3492 m²/s, dispersion=poor
  fetch_emission_inventory  -> sources=7, non_compliant=2
  fetch_health_risk_tables  -> pop=821000, sensitive=146400

Phase 3f: RAG on-demand lookup
  [RAG] PM2.5 compliance order WHO guideline -> AQMD-01 (score: 0.271)
  [RAG] PM2.5 concentration response health  -> HEALTH-01 (score: 0.389)

Phase 3g: MCP Gmail — check compliance inbox
  [MCP-Gmail] Inbox search -> 1 email found (DEMO)

Phase 3i: MCP Google Calendar — timestamp incident
  [MCP-GCal] Event: 'Stage I Watch — AQI 142' created (DEMO)

SECTION 6b — RAG KNOWLEDGE RETRIEVAL
  Regulatory lookups   : 2 queries against knowledge base
  RAG injected to sys  : WHO guidelines + AQMD rules + Episode Plan + Health C-R

SECTION 6c — MCP EXTERNAL INTEGRATIONS
  Gmail notifications  : 0 emails (AQI < 300, no emergency)
  Calendar events      : 1 event created
  Inbox checks         : 1 compliance search

OVERALL RISK: ELEVATED
```

---

## Disclaimer

This system provides decision support for air quality management. All mitigation
orders, public health advisories, and regulatory enforcement actions produced by
this system must be reviewed and issued by authorized regulatory personnel before
operational use. Model outputs are based on simulated or processed sensor data and
do not constitute an official regulatory determination. RAG-retrieved regulatory
documents are provided for informational context only and should be verified against
current published regulations before legal use.
