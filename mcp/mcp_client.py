# mcp/mcp_client.py  — AirGuard v4
#
# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONTEXT PROTOCOL (MCP) CLIENT
# ══════════════════════════════════════════════════════════════════════════════
#
# WHAT IS MCP IN THIS CONTEXT?
# ══════════════════════════════════════════════════════════════════════════════
#
# The AirGuard ReAct agent can observe sensors and issue in-system actions
# (alerts, compliance orders, mitigations).  But a real incident response
# also requires reaching out to EXTERNAL services:
#
#   • Email the enforcement director when a shutdown order is issued
#   • Create a Google Calendar event to track the incident response timeline
#   • Check the inbox for recent field reports from AQMD inspectors
#   • Log the incident to a shared operational dashboard
#
# MCP (Model Context Protocol) is Anthropic's standard for connecting an LLM
# to external tools via server URLs.  We use the Anthropic /v1/messages API
# with the mcp_servers parameter to route specific tool calls to:
#
#   Gmail MCP Server    → https://gmail.mcp.claude.com/mcp
#   Google Calendar MCP → https://gcal.mcp.claude.com/mcp
#
# ══════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
#
# MCPClient._call_anthropic_with_mcp() sends a single-turn request to the
# Anthropic API with:
#   - An MCP-aware system prompt telling the LLM what tools are available
#   - The mcp_servers list pointing to the connected servers
#   - A structured user message describing the exact action to take
#
# The Anthropic API executes the MCP tool call server-side and returns the
# result.  MCPClient extracts the relevant content blocks.
#
# ══════════════════════════════════════════════════════════════════════════════
# DEMO MODE
# ══════════════════════════════════════════════════════════════════════════════
#
# If no OPENAI_API_KEY / ANTHROPIC_API_KEY is set, all MCP calls simulate
# success with structured demo responses so the pipeline runs end-to-end.

import os
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# MCP server URLs (from connected services in Claude.ai)
# ─────────────────────────────────────────────────────────────────────────────

MCP_SERVERS = {
    "gmail":    {"type": "url", "url": "https://gmail.mcp.claude.com/mcp",  "name": "gmail-mcp"},
    "gcal":     {"type": "url", "url": "https://gcal.mcp.claude.com/mcp",   "name": "gcal-mcp"},
}

# Which Anthropic model to use for MCP calls (Sonnet is faster and cheaper for tool use)
MCP_MODEL = "claude-sonnet-4-20250514"


class MCPClient:
    """
    Lightweight wrapper around the Anthropic /v1/messages API
    with MCP server integration.

    Each public method is called by one of the LangChain tool wrappers
    in mcp/mcp_tools.py.
    """

    def __init__(self):
        self._api_key   = os.getenv("ANTHROPIC_API_KEY", "")
        self._demo_mode = not bool(self._api_key.strip().startswith("sk-ant-"))

    # ── Gmail ─────────────────────────────────────────────────────────────────

    def send_incident_email(self, to: str, subject: str, body: str,
                            cc: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send an email via Gmail MCP.
        Used for: enforcement notifications, public health authority alerts,
                  post-incident reports to regulatory leadership.
        """
        if self._demo_mode:
            return {
                "success":    True,
                "message_id": f"DEMO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "to":         to,
                "subject":    subject,
                "sent_at":    datetime.now().isoformat(),
                "mode":       "demo",
            }

        prompt = (
            f"Send an email using Gmail with the following parameters:\n"
            f"To: {to}\n"
            f"Subject: {subject}\n"
            f"{'CC: ' + ', '.join(cc) + chr(10) if cc else ''}"
            f"Body:\n{body}\n\n"
            f"Send the email now and confirm the message ID."
        )
        result = self._call_anthropic_with_mcp(prompt, ["gmail"])
        return {
            "success":    True,
            "response":   result,
            "to":         to,
            "subject":    subject,
            "sent_at":    datetime.now().isoformat(),
        }

    def get_recent_incident_emails(self, query: str = "air quality incident",
                                   max_results: int = 5) -> Dict[str, Any]:
        """
        Search Gmail inbox for recent field reports or communications.
        Used to: check whether inspectors have filed compliance confirmation emails.
        """
        if self._demo_mode:
            return {
                "success": True,
                "query":   query,
                "emails":  [
                    {
                        "subject": "[DEMO] Field Report: Metro Chemical Works inspection",
                        "from":    "inspector@aqmd.demo",
                        "date":    (datetime.now() - timedelta(hours=2)).isoformat(),
                        "snippet": "Visited facility at 14:00. Scrubber still offline. "
                                   "NOV delivered in person. Operator committed to repair by 18:00.",
                    }
                ],
                "mode":    "demo",
            }

        prompt = (
            f"Search Gmail for emails matching: '{query}'. "
            f"Return the {max_results} most recent results with subject, sender, date, and snippet."
        )
        result = self._call_anthropic_with_mcp(prompt, ["gmail"])
        return {"success": True, "response": result, "query": query}

    # ── Google Calendar ───────────────────────────────────────────────────────

    def create_incident_calendar_event(self,
                                       title: str,
                                       start_time: str,
                                       duration_hours: float,
                                       description: str,
                                       attendees: Optional[List[str]] = None,
                                       location: str = "Metro City") -> Dict[str, Any]:
        """
        Create a Google Calendar event to track an incident response.
        Used for: episode declaration time-stamp, compliance deadlines,
                  hospital surge de-escalation checkpoints.
        """
        if self._demo_mode:
            event_id = f"DEMO-EVT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            return {
                "success":    True,
                "event_id":   event_id,
                "title":      title,
                "start":      start_time,
                "duration_h": duration_hours,
                "calendar":   "AirGuard Incident Tracking",
                "mode":       "demo",
            }

        end_time = (
            datetime.fromisoformat(start_time) + timedelta(hours=duration_hours)
        ).isoformat()

        prompt = (
            f"Create a Google Calendar event with these details:\n"
            f"Title: {title}\n"
            f"Start: {start_time}\n"
            f"End:   {end_time}\n"
            f"Location: {location}\n"
            f"Description: {description}\n"
            + (f"Invite attendees: {', '.join(attendees)}\n" if attendees else "")
            + "Add this to the 'AirGuard Incident Tracking' calendar. "
            "Confirm the event ID after creation."
        )
        result = self._call_anthropic_with_mcp(prompt, ["gcal"])
        return {
            "success":  True,
            "response": result,
            "title":    title,
            "start":    start_time,
        }

    def get_upcoming_compliance_deadlines(self, days_ahead: int = 7) -> Dict[str, Any]:
        """
        Retrieve upcoming compliance deadline events from Google Calendar.
        Used to: remind the agent of pending enforcement follow-ups.
        """
        if self._demo_mode:
            return {
                "success":   True,
                "deadlines": [
                    {
                        "title":    "[DEMO] SRC-001 Compliance Deadline",
                        "due":      (datetime.now() + timedelta(hours=24)).isoformat(),
                        "notes":    "Metro Chemical Works scrubber repair deadline",
                    }
                ],
                "mode": "demo",
            }

        prompt = (
            f"List all Google Calendar events in the 'AirGuard Incident Tracking' "
            f"calendar in the next {days_ahead} days. "
            "Include event title, start time, and description."
        )
        result = self._call_anthropic_with_mcp(prompt, ["gcal"])
        return {"success": True, "response": result}

    # ── Core Anthropic API call with MCP ──────────────────────────────────────

    def _call_anthropic_with_mcp(self, user_prompt: str,
                                  servers: List[str]) -> str:
        """
        Call the Anthropic /v1/messages API with MCP server integration.

        This is the actual API call that routes through the connected MCP servers.
        The Anthropic API executes the tool call and returns the result content.
        """
        import urllib.request

        server_configs = [MCP_SERVERS[s] for s in servers if s in MCP_SERVERS]
        if not server_configs:
            return "[MCP] No valid server configs found"

        payload = {
            "model":      MCP_MODEL,
            "max_tokens": 1024,
            "system": (
                "You are an air quality incident management assistant.  "
                "Use the available MCP tools to execute the requested action precisely.  "
                "Return only the result — no commentary."
            ),
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "mcp_servers": server_configs,
        }

        try:
            data    = json.dumps(payload).encode()
            headers = {
                "Content-Type":      "application/json",
                "x-api-key":         self._api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta":    "mcp-client-2025-04-04",
            }
            req  = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=data, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())

            # Extract text content from response
            contents = body.get("content", [])
            texts    = [
                c.get("text", "")
                for c in contents
                if c.get("type") == "text"
            ]
            return "\n".join(texts) or "[MCP] Empty response"

        except Exception as ex:
            return f"[MCP ERROR] {type(ex).__name__}: {ex}"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────
MCP_CLIENT = MCPClient()
