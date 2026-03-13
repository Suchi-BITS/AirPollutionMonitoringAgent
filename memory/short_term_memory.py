# memory/short_term_memory.py  — AirGuard v3
#
# ══════════════════════════════════════════════════════════════════════════════
# SHORT-TERM MEMORY
# ══════════════════════════════════════════════════════════════════════════════
#
# Without memory every analysis run is completely isolated:
#   Run 1 → AQI 140.  Advisory issued.  Compliance orders issued.
#   Run 2 → AQI 155.  No knowledge of Run 1.  Same advisory issued AGAIN.
#   Run 3 → AQI 130.  No idea whether we are improving or worsening.
#
# Short-term memory adds a ROLLING WINDOW that carries forward across runs:
#   Run 1 → AQI 140 stored in memory.
#   Run 2 → Memory: "15 pts worse (+10.7%)".
#            Agent: accelerating episode, deferred mitigations not working.
#            Memory gate: advisory already issued — suppress duplicate.
#   Run 3 → Memory: "improving -25 pts from peak 155".
#            Agent: interventions working, begin de-escalation.
#            Memory gate: emergency shutdown already issued — no re-issue.
#
# ══════════════════════════════════════════════════════════════════════════════
# WHAT IS STORED
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. AQI TREND WINDOW (last N snapshots)
#    Computes: direction, velocity, session peak/low, episode detection.
#
# 2. ALERT DEDUPLICATION REGISTRY
#    Tracks alert_type + district + severity already dispatched.
#    Suppresses duplicates; allows escalation (advisory → warning → emergency).
#
# 3. REGULATORY ACTION LOG
#    Tracks source_id + action_type already issued.
#    Suppresses duplicate NOVs; allows escalation (NOV → order → shutdown).
#
# 4. DEFERRED ACTION TRACKER
#    Mitigations recommended but not yet confirmed implemented.
#    Injected into the system prompt so the LLM can check for follow-through.
#
# 5. EPISODE METADATA
#    Declared flag, start timestamp, peak AQI reached.
#
# ══════════════════════════════════════════════════════════════════════════════
# SCOPE
# ══════════════════════════════════════════════════════════════════════════════
#
# IN-PROCESS memory.  Persists across run_react_agent() calls within the same
# Python process (a 30-min polling loop).  Does NOT survive process restart.
#
# For persistence across restarts: serialise SESSION_MEMORY to JSON/Redis/SQLite
# at the end of each run and reload at startup.

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AQISnapshot:
    """One AQI observation captured at the end of an analysis run."""
    timestamp:              str
    scenario:               str
    max_aqi:                int
    avg_aqi:                float
    dominant_pollutant:     str
    emergency_triggered:    bool
    critical_station_count: int
    dispersion_quality:     str


@dataclass
class IssuedAlert:
    """Record of an alert dispatched this session."""
    alert_type:  str            # 'public_health' | 'hospital' | 'regulatory'
    severity:    str            # severity label OR action_type for regulatory
    district:    str            # district name or 'network-wide'
    source_id:   Optional[str]  # for regulatory actions
    action_type: Optional[str]  # regulatory action tier
    issued_at:   str


@dataclass
class DeferredAction:
    """A mitigation recommendation logged but not yet confirmed implemented."""
    recommendation_id:     str
    title:                 str
    target_entity:         str
    category:              str
    priority:              str
    logged_at:             str
    confirmed_implemented: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Memory store
# ─────────────────────────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    In-process short-term memory for the AirGuard ReAct agent.

    Single shared instance SESSION_MEMORY imported by react_agent.py.
    Agent calls record_run() after each pass and reads context_summary()
    at the start of each pass.
    """

    # Severity → numeric rank for escalation comparisons
    _SEVERITY_RANK: Dict[str, int] = {
        # public health
        "advisory": 1, "warning": 2, "alert": 3, "emergency": 4,
        # hospital
        "normal": 0, "elevated": 1, "high": 2, "critical": 3,
        # regulatory
        "notice_of_violation": 1, "compliance_order": 2, "emergency_shutdown_order": 3,
    }

    def __init__(self, max_aqi_history: int = 12):
        self.max_aqi_history:  int                  = max_aqi_history
        self.aqi_history:      List[AQISnapshot]    = []
        self.issued_alerts:    List[IssuedAlert]    = []
        self.deferred_actions: List[DeferredAction] = []
        self.session_start:    str                  = datetime.now().isoformat()
        self.run_count:        int                  = 0
        self.episode_declared: bool                 = False
        self.episode_start:    Optional[str]        = None
        self.episode_peak_aqi: int                  = 0
        self._notes:           List[str]            = []

    # ── Write ─────────────────────────────────────────────────────────────────

    def record_run(self, state: dict) -> None:
        """
        Call ONCE at the end of each analysis run.
        Increments run_count and appends a new AQISnapshot.
        """
        self.run_count += 1
        readings = state.get("ground_readings", [])
        if not readings:
            return

        max_aqi  = max(r["aqi"] for r in readings)
        avg_aqi  = round(sum(r["aqi"] for r in readings) / len(readings), 1)
        critical = [r for r in readings
                    if r["aqi_category"] in ("unhealthy", "very_unhealthy", "hazardous")]

        pol_counts: Dict[str, int] = {}
        for r in readings:
            dp = r.get("dominant_pollutant", "PM2.5")
            pol_counts[dp] = pol_counts.get(dp, 0) + 1
        dominant_pol = max(pol_counts, key=pol_counts.get) if pol_counts else "PM2.5"

        met = state.get("meteorological_summary") or {}

        snap = AQISnapshot(
            timestamp              = datetime.now().isoformat(),
            scenario               = state.get("target_region", "unknown"),
            max_aqi                = max_aqi,
            avg_aqi                = avg_aqi,
            dominant_pollutant     = dominant_pol,
            emergency_triggered    = state.get("emergency_triggered", False),
            critical_station_count = len(critical),
            dispersion_quality     = met.get("dispersion_quality", "unknown"),
        )
        self.aqi_history.append(snap)

        # Rolling window trim
        if len(self.aqi_history) > self.max_aqi_history:
            self.aqi_history = self.aqi_history[-self.max_aqi_history:]

        # Episode lifecycle
        if state.get("emergency_triggered") and not self.episode_declared:
            self.episode_declared = True
            self.episode_start    = snap.timestamp
            self.episode_peak_aqi = max_aqi
        elif self.episode_declared and max_aqi > self.episode_peak_aqi:
            self.episode_peak_aqi = max_aqi

    def record_alert(self,
                     alert_type:  str,
                     severity:    str,
                     district:    str           = "network-wide",
                     source_id:   Optional[str] = None,
                     action_type: Optional[str] = None) -> None:
        """Register a dispatched alert to enable future deduplication checks."""
        self.issued_alerts.append(IssuedAlert(
            alert_type  = alert_type,
            severity    = severity,
            district    = district,
            source_id   = source_id,
            action_type = action_type,
            issued_at   = datetime.now().isoformat(),
        ))

    def record_deferred_action(self, recommendation_id: str, title: str,
                               target_entity: str, category: str, priority: str) -> None:
        """Track a mitigation recommendation that is pending implementation."""
        # Do not duplicate
        if any(d.recommendation_id == recommendation_id for d in self.deferred_actions):
            return
        self.deferred_actions.append(DeferredAction(
            recommendation_id     = recommendation_id,
            title                 = title,
            target_entity         = target_entity,
            category              = category,
            priority              = priority,
            logged_at             = datetime.now().isoformat(),
        ))

    def add_note(self, note: str) -> None:
        """Append a free-text observation to the session log."""
        self._notes.append(f"[Run {self.run_count}] {note}")

    # ── Read / query ──────────────────────────────────────────────────────────

    def can_issue_alert(self, alert_type: str, severity: str,
                        district: str = "network-wide",
                        source_id: Optional[str] = None) -> bool:
        """
        Return True if the alert may be dispatched.

        Logic:
          No prior alert of this type+district → ALLOW
          Same or lower severity than already issued → SUPPRESS (duplicate)
          Strictly higher severity → ALLOW (escalation)
        """
        matching = [
            a for a in self.issued_alerts
            if a.alert_type == alert_type
            and a.district  == district
            and (source_id is None or a.source_id == source_id)
        ]
        if not matching:
            return True
        last      = max(matching, key=lambda a: a.issued_at)
        last_rank = self._SEVERITY_RANK.get(last.severity, 0)
        new_rank  = self._SEVERITY_RANK.get(severity, 0)
        return new_rank > last_rank

    def can_issue_regulatory_action(self, source_id: str, action_type: str) -> bool:
        """
        Return True if the regulatory action may be issued for this source.
        Prevents duplicate NOVs; allows escalation to higher tiers.
        """
        matching = [
            a for a in self.issued_alerts
            if a.alert_type == "regulatory" and a.source_id == source_id
        ]
        if not matching:
            return True
        last      = max(matching, key=lambda a: a.issued_at)
        last_rank = self._SEVERITY_RANK.get(last.action_type or "", 0)
        new_rank  = self._SEVERITY_RANK.get(action_type, 0)
        return new_rank > last_rank

    def get_aqi_trend(self) -> Dict[str, Any]:
        """
        Compute trend statistics from the rolling AQI history window.
        Return value is consumed by ContextEngineer.build_system_prompt().
        """
        if not self.aqi_history:
            return {"available": False}

        recent      = self.aqi_history[-1]
        history_max = max(s.max_aqi for s in self.aqi_history)
        history_min = min(s.max_aqi for s in self.aqi_history)

        if len(self.aqi_history) >= 2:
            prev      = self.aqi_history[-2]
            delta     = recent.max_aqi - prev.max_aqi
            pct       = round((delta / max(prev.max_aqi, 1)) * 100, 1)
            direction = "worsening" if delta > 5 else "improving" if delta < -5 else "stable"
            prev_max  = prev.max_aqi
        else:
            delta, pct, direction, prev_max = 0, 0.0, "baseline (first run)", None

        # Sustained episode = 3+ consecutive runs above AQI 150
        consec = 0
        for s in reversed(self.aqi_history):
            if s.max_aqi > 150:
                consec += 1
            else:
                break

        return {
            "available":                  True,
            "run_count":                  self.run_count,
            "current_max_aqi":            recent.max_aqi,
            "previous_max_aqi":           prev_max,
            "delta_aqi":                  delta,
            "delta_pct":                  pct,
            "direction":                  direction,
            "session_peak_aqi":           history_max,
            "session_low_aqi":            history_min,
            "consecutive_unhealthy_runs": consec,
            "episode_declared":           self.episode_declared,
            "episode_peak_aqi":           self.episode_peak_aqi if self.episode_declared else None,
            "dominant_pollutant":         recent.dominant_pollutant,
            "dispersion_quality":         recent.dispersion_quality,
            "runs_in_window":             len(self.aqi_history),
        }

    def get_alert_history_summary(self) -> Dict[str, Any]:
        """Aggregate session alert counts for reporting and context injection."""
        public = [a for a in self.issued_alerts if a.alert_type == "public_health"]
        hosp   = [a for a in self.issued_alerts if a.alert_type == "hospital"]
        reg    = [a for a in self.issued_alerts if a.alert_type == "regulatory"]

        highest_public = None
        if public:
            highest_public = max(
                public, key=lambda a: self._SEVERITY_RANK.get(a.severity, 0)
            ).severity

        return {
            "public_alerts_issued":      len(public),
            "hospital_alerts_issued":    len(hosp),
            "regulatory_actions_issued": len(reg),
            "highest_public_severity":   highest_public,
            "regulated_sources":         sorted({a.source_id for a in reg if a.source_id}),
            "deferred_actions_pending":  sum(
                1 for d in self.deferred_actions if not d.confirmed_implemented
            ),
        }

    def context_summary(self) -> str:
        """
        Return a compact text block for injection into the LLM system prompt.
        Primary interface between the memory layer and ContextEngineer.
        """
        trend  = self.get_aqi_trend()
        alerts = self.get_alert_history_summary()

        if not trend["available"]:
            return (
                "SESSION MEMORY: This is run #1.  "
                "No prior analysis data available.  Establish baseline."
            )

        lines = [
            f"SESSION MEMORY (run #{self.run_count}, "
            f"{len(self.aqi_history)} runs in window):",
            f"  AQI trend      : {trend['direction'].upper()} | "
            f"current max={trend['current_max_aqi']} | "
            f"prev={trend['previous_max_aqi']} | "
            f"delta={trend['delta_aqi']:+d} ({trend['delta_pct']:+.1f}%)",
            f"  Session peak   : AQI {trend['session_peak_aqi']} | "
            f"session low: AQI {trend['session_low_aqi']}",
            f"  Dominant pol.  : {trend['dominant_pollutant']} | "
            f"dispersion: {trend['dispersion_quality']}",
        ]

        if trend["consecutive_unhealthy_runs"] >= 3:
            lines.append(
                f"  *** SUSTAINED EPISODE: {trend['consecutive_unhealthy_runs']} "
                f"consecutive runs with AQI > 150.  Escalate response."
            )
        if trend["episode_declared"]:
            lines.append(
                f"  *** EPISODE DECLARED: peak AQI {trend['episode_peak_aqi']}."
            )

        lines.append(
            f"  Alerts issued  : "
            f"public={alerts['public_alerts_issued']} "
            f"(highest: {alerts['highest_public_severity'] or 'none'}) | "
            f"hospital={alerts['hospital_alerts_issued']} | "
            f"regulatory={alerts['regulatory_actions_issued']}"
        )

        if alerts["regulated_sources"]:
            lines.append(
                f"  Regulated srcs : {', '.join(alerts['regulated_sources'])} "
                f"— do NOT re-issue same enforcement tier"
            )

        if alerts["deferred_actions_pending"] > 0:
            lines.append(
                f"  Deferred items : {alerts['deferred_actions_pending']} pending — "
                f"check whether prior recommendations were implemented "
                f"before logging new ones"
            )

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all state.  Call at the start of a new monitoring day."""
        self.__init__(max_aqi_history=self.max_aqi_history)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — imported by react_agent.py and context_engineer.py
# ─────────────────────────────────────────────────────────────────────────────
SESSION_MEMORY = ShortTermMemory(max_aqi_history=12)
