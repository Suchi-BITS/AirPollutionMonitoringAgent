# tools/action_tools.py
# Tools for issuing mitigation recommendations, public health alerts,
# and regulatory enforcement actions.
# In production: these write to incident management systems, SMS gateways,
# regulatory enforcement databases, and emergency operations centers.

from datetime import datetime

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):
        return fn

_action_log: list = []


@tool
def issue_public_health_alert(
    severity: str,
    affected_districts: list,
    aqi_level: int,
    dominant_pollutant: str,
    health_message: str,
    recommended_actions: list,
    sensitive_groups_warning: str,
    duration_hours: int,
    channels: list,
) -> dict:
    """
    Publish a public health advisory across communication channels.

    In production: integrates with city emergency notification system (Everbridge,
    Rave Mobile Safety), pushes to AirNow.gov, sends SMS via Twilio, updates
    digital signage, and posts to city social media API.

    Args:
        severity: 'advisory' | 'warning' | 'alert' | 'emergency'
        affected_districts: list of district names
        aqi_level: current AQI value
        dominant_pollutant: the pollutant driving the AQI
        health_message: clear public-facing explanation of the risk
        recommended_actions: list of specific actions the public should take
        sensitive_groups_warning: targeted message for vulnerable populations
        duration_hours: expected duration of the advisory
        channels: ['mobile_app', 'sms', 'website', 'digital_signage', 'media']

    Returns:
        Alert dispatch confirmation with alert ID
    """
    alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    record = {
        "alert_id":               alert_id,
        "type":                   "public_health_alert",
        "severity":               severity,
        "affected_districts":     affected_districts,
        "aqi_level":              aqi_level,
        "dominant_pollutant":     dominant_pollutant,
        "health_message":         health_message,
        "recommended_actions":    recommended_actions,
        "sensitive_groups":       sensitive_groups_warning,
        "duration_hours":         duration_hours,
        "channels":               channels,
        "issued_at":              datetime.now().isoformat(),
        "status":                 "issued",
    }
    _action_log.append(record)
    print(f"  [PUBLIC ALERT - {severity.upper()}] AQI {aqi_level} ({dominant_pollutant}) "
          f"in {len(affected_districts)} district(s)")
    return {"success": True, "alert_id": alert_id, "severity": severity,
            "districts_notified": len(affected_districts), "channels": channels}


@tool
def issue_regulatory_action(
    source_id: str,
    source_name: str,
    violation_type: str,
    pollutant: str,
    measured_concentration: float,
    permitted_limit: float,
    action_type: str,
    required_action: str,
    compliance_deadline: str,
    enforcement_authority: str,
    regulatory_basis: str,
) -> dict:
    """
    Issue a formal regulatory enforcement action against a non-compliant emission source.

    In production: writes to the state AQMD enforcement database, generates
    Notice of Violation (NOV) document, triggers legal notification workflow,
    and schedules follow-up inspection.

    Args:
        source_id: facility identifier from emission inventory
        source_name: facility name
        violation_type: 'permit_exceedance' | 'emergency_emission' | 'operating_without_permit'
        pollutant: pollutant in violation
        measured_concentration: observed concentration in ug/m3
        permitted_limit: permitted limit in ug/m3
        action_type: 'notice_of_violation' | 'compliance_order' | 'emergency_shutdown_order'
        required_action: specific action the operator must take
        compliance_deadline: ISO date string for compliance deadline
        enforcement_authority: issuing regulatory body
        regulatory_basis: applicable regulation / code section

    Returns:
        Enforcement action record with case number
    """
    case_number = f"ENF-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{source_id}"
    record = {
        "case_number":             case_number,
        "type":                    "regulatory_action",
        "source_id":               source_id,
        "source_name":             source_name,
        "violation_type":          violation_type,
        "pollutant":               pollutant,
        "measured_concentration":  measured_concentration,
        "permitted_limit":         permitted_limit,
        "exceedance_factor":       round(measured_concentration / max(permitted_limit, 0.01), 2),
        "action_type":             action_type,
        "required_action":         required_action,
        "compliance_deadline":     compliance_deadline,
        "enforcement_authority":   enforcement_authority,
        "regulatory_basis":        regulatory_basis,
        "issued_at":               datetime.now().isoformat(),
        "status":                  "issued",
    }
    _action_log.append(record)
    print(f"  [REGULATORY] {action_type.upper()}: {source_name} "
          f"({pollutant}: {measured_concentration} vs limit {permitted_limit} ug/m3) "
          f"[x{record['exceedance_factor']}]")
    return {"success": True, "case_number": case_number, "action_type": action_type,
            "source_name": source_name, "exceedance_factor": record["exceedance_factor"]}


@tool
def log_mitigation_recommendation(
    priority: str,
    category: str,
    target_entity: str,
    title: str,
    description: str,
    expected_aqi_reduction: float,
    implementation_timeline: str,
    regulatory_basis: str,
    estimated_cost_tier: str,
    co_benefits: list,
) -> dict:
    """
    Log a structured mitigation recommendation to the action plan.

    These recommendations are assembled by the Mitigation Agent and reviewed
    by the Supervisor before being included in the situation report.

    In production: writes to the city air quality management plan database
    and generates tasks in the regulatory agency project management system.

    Args:
        priority: 'emergency' | 'high' | 'medium' | 'low'
        category: 'regulatory' | 'operational' | 'public_health' | 'infrastructure' | 'traffic'
        target_entity: who implements: 'regulator' | 'facility_operator' | 'city_transport' | 'public' | 'agriculture'
        title: short action title (max 80 chars)
        description: full description of the recommended action
        expected_aqi_reduction: estimated AQI point reduction if implemented
        implementation_timeline: 'immediate' | 'within_24h' | 'within_week' | 'medium_term' | 'long_term'
        regulatory_basis: applicable regulation or policy reference
        estimated_cost_tier: 'negligible' | 'low' | 'medium' | 'high' | 'very_high'
        co_benefits: list of co-benefits beyond AQI improvement

    Returns:
        Recommendation record with ID
    """
    rec_id = f"REC-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{priority[0].upper()}"
    record = {
        "recommendation_id":       rec_id,
        "type":                    "mitigation_recommendation",
        "priority":                priority,
        "category":                category,
        "target_entity":           target_entity,
        "title":                   title,
        "description":             description,
        "expected_aqi_reduction":  expected_aqi_reduction,
        "implementation_timeline": implementation_timeline,
        "regulatory_basis":        regulatory_basis,
        "estimated_cost_tier":     estimated_cost_tier,
        "co_benefits":             co_benefits,
        "logged_at":               datetime.now().isoformat(),
    }
    _action_log.append(record)
    print(f"  [RECOMMENDATION - {priority.upper()}] {title} "
          f"(expected AQI reduction: {expected_aqi_reduction:.1f} pts, "
          f"timeline: {implementation_timeline})")
    return {"success": True, "recommendation_id": rec_id, "priority": priority,
            "expected_aqi_reduction": expected_aqi_reduction}


@tool
def notify_hospital_network(
    alert_level: str,
    affected_districts: list,
    primary_pollutant: str,
    aqi: int,
    expected_case_types: list,
    expected_volume_increase_pct: float,
    special_instructions: str,
) -> dict:
    """
    Send clinical preparedness alert to hospitals and emergency departments.

    In production: integrates with hospital incident command systems (HAvBED2),
    sends HL7 FHIR alerts to hospital EHR systems, notifies emergency medical
    services (EMS) dispatch centers, and updates poison control center.

    Args:
        alert_level: 'normal' | 'elevated' | 'high' | 'critical'
        affected_districts: districts with elevated pollution
        primary_pollutant: dominant pollutant causing health risk
        aqi: current AQI value
        expected_case_types: list of expected case types (e.g. ['asthma', 'COPD exacerbation'])
        expected_volume_increase_pct: estimated ED visit volume increase
        special_instructions: clinical guidance for treating physicians

    Returns:
        Hospital notification confirmation
    """
    notif_id = f"HOSP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    record = {
        "notification_id":              notif_id,
        "type":                         "hospital_network_alert",
        "alert_level":                  alert_level,
        "affected_districts":           affected_districts,
        "primary_pollutant":            primary_pollutant,
        "aqi":                          aqi,
        "expected_case_types":          expected_case_types,
        "expected_volume_increase_pct": expected_volume_increase_pct,
        "special_instructions":         special_instructions,
        "issued_at":                    datetime.now().isoformat(),
        "status":                       "dispatched",
    }
    _action_log.append(record)
    print(f"  [HOSPITAL ALERT - {alert_level.upper()}] AQI {aqi} ({primary_pollutant}) "
          f"— expected ED volume increase {expected_volume_increase_pct:.0f}%")
    return {"success": True, "notification_id": notif_id, "alert_level": alert_level,
            "hospitals_notified": 6, "ems_notified": True}


@tool
def request_traffic_restriction(
    restriction_type: str,
    affected_zones: list,
    vehicles_affected: str,
    start_time: str,
    end_time: str,
    reason: str,
    legal_basis: str,
) -> dict:
    """
    Request traffic restriction to reduce vehicular emissions during a pollution episode.

    In production: submits to city traffic management center (ATMS), updates
    variable message signs (VMS), notifies police dispatch for enforcement,
    and triggers congestion charge API modifications.

    Args:
        restriction_type: 'low_emission_zone' | 'odd_even' | 'diesel_ban' | 'speed_reduction' | 'hgv_ban'
        affected_zones: list of zone/district names
        vehicles_affected: description of vehicle types restricted
        start_time: ISO format or HH:MM
        end_time: ISO format or HH:MM
        reason: public justification
        legal_basis: applicable traffic / air quality regulation

    Returns:
        Restriction request confirmation with reference number
    """
    ref = f"TRF-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    record = {
        "reference":          ref,
        "type":               "traffic_restriction",
        "restriction_type":   restriction_type,
        "affected_zones":     affected_zones,
        "vehicles_affected":  vehicles_affected,
        "start_time":         start_time,
        "end_time":           end_time,
        "reason":             reason,
        "legal_basis":        legal_basis,
        "requested_at":       datetime.now().isoformat(),
        "status":             "submitted",
    }
    _action_log.append(record)
    print(f"  [TRAFFIC RESTRICTION] {restriction_type.upper()} in {len(affected_zones)} zone(s) "
          f"({start_time} - {end_time})")
    return {"success": True, "reference": ref, "restriction_type": restriction_type,
            "zones_affected": len(affected_zones)}


@tool
def get_action_log(limit: int = 50) -> list:
    """
    Retrieve the log of all actions taken in this analysis session.

    Returns:
        List of all action records ordered chronologically
    """
    return _action_log[-limit:]
