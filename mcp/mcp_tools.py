# mcp/mcp_tools.py  — AirGuard v4
#
# ══════════════════════════════════════════════════════════════════════════════
# MCP-BACKED LANGCHAIN TOOLS
# ══════════════════════════════════════════════════════════════════════════════
#
# This module exposes the MCP operations as standard LangChain @tool functions
# so they integrate seamlessly into ALL_TOOLS in react_agent.py and the LLM
# can call them during the ReAct loop exactly like any other tool.
#
# TOOL INVENTORY:
#   send_incident_notification_email  → Gmail MCP: email to enforcement/health officials
#   create_response_calendar_event    → Google Calendar MCP: timestamp incidents & deadlines
#   check_compliance_inbox            → Gmail MCP: read inspector field reports
#   get_compliance_deadlines          → Google Calendar MCP: upcoming enforcement deadlines
#   query_knowledge_base              → RAG (no MCP): on-demand regulatory/health lookup
#
# The last tool (query_knowledge_base) is included here because it is
# logically "external knowledge retrieval" alongside the MCP tools.
#
# ══════════════════════════════════════════════════════════════════════════════
# WHEN THE LLM SHOULD CALL THESE TOOLS
# ══════════════════════════════════════════════════════════════════════════════
#
#   send_incident_notification_email:
#     → After issuing an emergency_shutdown_order
#     → After declaring a Stage III episode
#     → When hospital alert level reaches CRITICAL
#
#   create_response_calendar_event:
#     → When an episode is first declared (timestamp it)
#     → When a compliance deadline is set (track it)
#     → When a de-escalation checkpoint is scheduled
#
#   check_compliance_inbox:
#     → At the start of run 2+ to check if inspectors filed compliance confirmations
#
#   get_compliance_deadlines:
#     → When reviewing outstanding deferred actions
#
#   query_knowledge_base:
#     → Before citing a specific regulatory limit to verify the threshold
#     → When determining which episode stage applies
#     → When estimating health impact of a specific pollutant level

from datetime import datetime
from typing import Optional, List

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):
        return fn

from mcp.mcp_client  import MCP_CLIENT
from rag.rag_retriever import RAG_RETRIEVER


# ─────────────────────────────────────────────────────────────────────────────
# Gmail tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def send_incident_notification_email(
    recipient_role:    str,
    recipient_email:   str,
    incident_type:     str,
    subject:           str,
    message_body:      str,
    cc_emails:         Optional[List[str]] = None,
) -> dict:
    """
    Send an incident notification email via Gmail MCP to enforcement officers,
    public health authorities, or regulatory directors.

    Call this tool AFTER issuing an emergency_shutdown_order, Stage III episode
    declaration, or CRITICAL hospital alert to ensure human authorities are
    immediately informed via official communication channels.

    In production: emails are delivered via the connected Gmail account to real
    enforcement officers.  In demo mode: returns a confirmation without sending.

    Args:
        recipient_role:  human role of the recipient (e.g. 'AQMD Enforcement Director')
        recipient_email: email address
        incident_type:   'emergency_shutdown' | 'episode_declaration' | 'hospital_critical' | 'compliance_deadline'
        subject:         email subject line
        message_body:    full email body with incident details
        cc_emails:       optional list of CC email addresses

    Returns:
        dict with success flag, message_id, and delivery confirmation
    """
    result = MCP_CLIENT.send_incident_email(
        to      = recipient_email,
        subject = subject,
        body    = message_body,
        cc      = cc_emails,
    )
    print(f"  [MCP-Gmail] Email → {recipient_role} <{recipient_email}> | "
          f"Subject: {subject[:60]} | "
          f"{'SENT' if result.get('success') else 'FAILED'} "
          f"({'DEMO' if result.get('mode') == 'demo' else 'LIVE'})")
    return result


@tool
def check_compliance_inbox(
    search_query: str = "air quality compliance inspection",
    max_results:  int = 5,
) -> dict:
    """
    Search the Gmail inbox for recent compliance reports, field inspection
    emails, or operator-submitted compliance confirmations.

    Call this tool at the start of run 2+ to check whether AQMD inspectors
    have confirmed that previous enforcement actions have been implemented
    before issuing duplicate orders.

    Args:
        search_query: Gmail search terms (e.g. 'compliance Metro Chemical' or 'AQI inspection')
        max_results:  maximum emails to retrieve

    Returns:
        dict with list of matching emails (subject, sender, date, snippet)
    """
    result = MCP_CLIENT.get_recent_incident_emails(
        query       = search_query,
        max_results = max_results,
    )
    emails = result.get("emails", [])
    print(f"  [MCP-Gmail] Inbox search: '{search_query}' → {len(emails)} email(s) found "
          f"({'DEMO' if result.get('mode') == 'demo' else 'LIVE'})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Google Calendar tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def create_response_calendar_event(
    event_title:       str,
    event_type:        str,
    description:       str,
    duration_hours:    float = 2.0,
    attendee_emails:   Optional[List[str]] = None,
) -> dict:
    """
    Create a Google Calendar event in the AirGuard Incident Tracking calendar
    to timestamp an air quality incident, compliance deadline, or response checkpoint.

    Call this tool when:
      - A new episode is declared (timestamp the declaration)
      - A compliance deadline is set via issue_regulatory_action (track it)
      - A de-escalation checkpoint needs to be scheduled

    Args:
        event_title:       descriptive title (e.g. 'Stage III Episode Declaration — AQI 487')
        event_type:        'episode_declaration' | 'compliance_deadline' | 'deescalation_checkpoint' | 'hospital_surge'
        description:       full event description with incident details
        duration_hours:    expected event duration (default 2 hours)
        attendee_emails:   optional list of attendee emails (enforcement officers, health dept)

    Returns:
        dict with success flag, event_id, and calendar link
    """
    start_time = datetime.now().isoformat()
    result = MCP_CLIENT.create_incident_calendar_event(
        title          = event_title,
        start_time     = start_time,
        duration_hours = duration_hours,
        description    = description,
        attendees      = attendee_emails,
    )
    print(f"  [MCP-GCal] Calendar event created: '{event_title}' "
          f"[{event_type}] at {start_time[:16]} "
          f"({'DEMO' if result.get('mode') == 'demo' else 'LIVE'})")
    return result


@tool
def get_compliance_deadlines(
    days_ahead: int = 7,
) -> dict:
    """
    Retrieve upcoming compliance deadlines and incident response checkpoints
    from the Google Calendar AirGuard Incident Tracking calendar.

    Call this tool at the start of run 2+ to check whether any previously
    issued compliance deadlines are approaching or overdue before deciding
    whether to escalate enforcement.

    Args:
        days_ahead: how many days ahead to look for upcoming events

    Returns:
        dict with list of upcoming deadline events
    """
    result = MCP_CLIENT.get_upcoming_compliance_deadlines(days_ahead=days_ahead)
    deadlines = result.get("deadlines", [])
    print(f"  [MCP-GCal] Compliance deadlines: {len(deadlines)} event(s) in next "
          f"{days_ahead} day(s) "
          f"({'DEMO' if result.get('mode') == 'demo' else 'LIVE'})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RAG on-demand query tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def query_knowledge_base(
    query:    str,
    category: Optional[str] = None,
    top_k:    int = 3,
) -> str:
    """
    Search the AirGuard regulatory and health knowledge base for specific
    standards, thresholds, protocols, and historical episode data.

    Call this tool BEFORE citing a specific regulatory limit or threshold to
    ensure the value is grounded in the retrieved documents rather than
    approximated from training data.

    Available categories (leave blank to search all):
      who_guidelines      — WHO Air Quality Guidelines 2021
      naaqs               — US EPA National Ambient Air Quality Standards
      state_regulation    — State AQMD Emission Rules and LEZ procedures
      episode_plan        — Metro City Stage I/II/III Episode Plans
      health_tables       — Concentration-response functions and hospital surge guidance
      historical_episodes — Lessons learned from past Metro City episodes

    Example queries:
      "What is the WHO 24-hour PM2.5 guideline?"
      "Stage III emergency shutdown triggers and required actions"
      "SO2 emission permit limit for industrial boilers"
      "Hospital surge capacity guidance for CRITICAL air quality alert"
      "HGV curfew effectiveness data from past episodes"

    Args:
        query:    free-text search query
        category: optional category filter (see above)
        top_k:    number of document chunks to retrieve (1-5)

    Returns:
        Formatted string with retrieved document excerpts and source citations
    """
    result = RAG_RETRIEVER.on_demand_query(query=query, category=category, top_k=top_k)
    # Count results in the output for logging
    n = result.count("[") - 1 if "[" in result else 0
    print(f"  [RAG] query_knowledge_base: '{query[:60]}' → {max(n, 1)} chunk(s) retrieved")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Export list for inclusion in ALL_TOOLS
# ─────────────────────────────────────────────────────────────────────────────

MCP_AND_RAG_TOOLS = [
    send_incident_notification_email,
    check_compliance_inbox,
    create_response_calendar_event,
    get_compliance_deadlines,
    query_knowledge_base,
]
