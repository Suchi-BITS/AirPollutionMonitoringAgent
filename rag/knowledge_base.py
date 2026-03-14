# rag/knowledge_base.py  — AirGuard v4
#
# ══════════════════════════════════════════════════════════════════════════════
# RAG KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
#
# WHY RAG IN AN AIR-QUALITY AGENT?
# ══════════════════════════════════════════════════════════════════════════════
#
# The ReAct agent makes enforcement and health decisions that must be grounded
# in specific regulatory thresholds, WHO guidelines, and historical episode
# protocols.  Without RAG:
#
#   Problem 1 — HALLUCINATION OF THRESHOLDS
#     The LLM may invent permit limits ("SO2 must be < 30 µg/m³") that differ
#     from the actual applicable standard ("State AQMD Rule 1001: 100 µg/m³").
#     A regulatory action based on wrong limits is legally invalid.
#
#   Problem 2 — STALE GUIDELINES
#     WHO updated its air quality guidelines in 2021 (PM2.5 24-hr dropped from
#     25 to 15 µg/m³).  An LLM trained before that update would cite old values.
#
#   Problem 3 — MISSING EPISODE PROTOCOLS
#     Stage I/II/III episode action plans are jurisdiction-specific.  The LLM
#     has no way to know the precise curtailment trigger thresholds for Metro
#     City's own Air Pollution Episode Plan.
#
# RAG solves all three by RETRIEVING verified documents at query time and
# injecting them into the context window, so the LLM reasons against facts
# rather than training-set approximations.
#
# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION: PURE-PYTHON IN-MEMORY VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
#
# No external libraries required (no FAISS, no ChromaDB, no sentence-transformers).
# Uses TF-IDF bag-of-words vectors with cosine similarity — sufficient for the
# small, domain-specific corpus here (~40 chunks).
#
# PRODUCTION REPLACEMENT:
#   Replace KnowledgeBase with a ChromaDB / Pinecone / pgvector backend.
#   Replace _tfidf_vector() with sentence-transformers embeddings (all-MiniLM-L6).
#   The RAGRetriever interface (query, retrieve) stays exactly the same.
#
# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE CORPUS
# ══════════════════════════════════════════════════════════════════════════════
#
# Six document categories, each chunked into paragraph-sized passages:
#
#   1. WHO Air Quality Guidelines 2021
#   2. US EPA National Ambient Air Quality Standards (NAAQS)
#   3. State AQMD Emission Rules (local regulatory basis)
#   4. Metro City Air Pollution Episode Plan (Stage I / II / III)
#   5. Health Impact Tables (concentration-response functions)
#   6. Historical Episode Lessons Learned

import math
import re
from typing import List, Dict, Tuple, Any


# ─────────────────────────────────────────────────────────────────────────────
# Document chunk data structure
# ─────────────────────────────────────────────────────────────────────────────

class DocumentChunk:
    """One retrievable passage from the knowledge corpus."""
    __slots__ = ("doc_id", "category", "title", "text", "metadata", "_vector")

    def __init__(self, doc_id: str, category: str, title: str,
                 text: str, metadata: Dict[str, Any] = None):
        self.doc_id   = doc_id
        self.category = category
        self.title    = title
        self.text     = text
        self.metadata = metadata or {}
        self._vector: Dict[str, float] = {}   # populated lazily by KnowledgeBase


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge corpus — curated domain documents
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_CORPUS: List[DocumentChunk] = [

    # ── WHO Air Quality Guidelines 2021 ──────────────────────────────────────

    DocumentChunk("WHO-01", "who_guidelines",
        "WHO AQG 2021 — PM2.5 Guidelines",
        """WHO Air Quality Guidelines 2021: PM2.5 annual mean guideline is 5 µg/m³.
PM2.5 24-hour mean guideline is 15 µg/m³ (99th percentile, i.e. 3-4 exceedance days/year allowed).
Interim targets: IT-1=35 µg/m³, IT-2=25 µg/m³, IT-3=15 µg/m³ (24-hr).
Health basis: PM2.5 causes cardiovascular disease, lung cancer, acute lower respiratory
infections. Each 10 µg/m³ increase in long-term PM2.5 exposure increases all-cause
mortality by 6-7% (Pope & Dockery 2006 cohort study).""",
        {"pollutant": "PM2.5", "standard_type": "guideline", "year": 2021}),

    DocumentChunk("WHO-02", "who_guidelines",
        "WHO AQG 2021 — PM10 Guidelines",
        """WHO Air Quality Guidelines 2021: PM10 annual mean guideline is 15 µg/m³.
PM10 24-hour mean guideline is 45 µg/m³.
Interim targets: IT-1=70 µg/m³, IT-2=50 µg/m³, IT-3=30 µg/m³ (24-hr).
PM10 includes coarse particles (2.5-10 µm) from road dust, construction, agriculture.
Epidemiological evidence links PM10 exceedance to increased respiratory hospital admissions.""",
        {"pollutant": "PM10", "standard_type": "guideline", "year": 2021}),

    DocumentChunk("WHO-03", "who_guidelines",
        "WHO AQG 2021 — NO2 Guidelines",
        """WHO Air Quality Guidelines 2021: NO2 annual mean guideline is 10 µg/m³.
NO2 24-hour mean guideline is 25 µg/m³.
Short-term (1-hour) peak guideline: 200 µg/m³ (previous value, unchanged in 2021 update).
NO2 is a marker for traffic emissions and reacts with VOCs to form ground-level ozone.
Short-term NO2 exposure exacerbates asthma; long-term linked to lung development impairment
in children. Each 10 µg/m³ increase raises asthma attack risk by 4.5%.""",
        {"pollutant": "NO2", "standard_type": "guideline", "year": 2021}),

    DocumentChunk("WHO-04", "who_guidelines",
        "WHO AQG 2021 — SO2 Guidelines",
        """WHO Air Quality Guidelines 2021: SO2 24-hour mean guideline is 40 µg/m³.
WHO no longer recommends a separate annual SO2 guideline (evidence insufficient).
SO2 10-minute peak guideline: 500 µg/m³.
SO2 is emitted by combustion of sulfur-containing fuels (coal, heavy fuel oil).
Industrial stack emissions are primary source. Reacts with water to form acid rain.
At high concentrations (>400 µg/m³), SO2 causes bronchoconstriction within minutes.""",
        {"pollutant": "SO2", "standard_type": "guideline", "year": 2021}),

    DocumentChunk("WHO-05", "who_guidelines",
        "WHO AQG 2021 — O3 Guidelines",
        """WHO Air Quality Guidelines 2021: O3 peak season 8-hour mean guideline is 60 µg/m³.
Previous (2005) guideline was 100 µg/m³; 2021 update tightened to 60 µg/m³.
O3 is formed by photochemical reactions between NOx and VOCs in sunlight.
Episodes peak in summer afternoons. O3 causes respiratory inflammation, reduces lung function,
triggers asthma attacks. Sensitive groups: children, outdoor athletes, COPD patients.""",
        {"pollutant": "O3", "standard_type": "guideline", "year": 2021}),

    DocumentChunk("WHO-06", "who_guidelines",
        "WHO AQG 2021 — CO Guidelines",
        """WHO Air Quality Guidelines 2021: CO 24-hour mean guideline is 4 mg/m³.
CO 8-hour mean guideline: 10 mg/m³. CO 1-hour mean guideline: 35 mg/m³.
CO is primarily emitted by incomplete combustion — petrol/diesel vehicles, residential heating.
CO reduces oxygen-carrying capacity of blood (carboxyhaemoglobin formation).
Sensitive groups: cardiovascular patients, pregnant women, infants.
At >30 mg/m³, healthy adults experience headache and impaired cognitive function.""",
        {"pollutant": "CO", "standard_type": "guideline", "year": 2021}),

    # ── US EPA NAAQS ──────────────────────────────────────────────────────────

    DocumentChunk("EPA-01", "naaqs",
        "US EPA NAAQS — PM2.5 Standards",
        """US EPA National Ambient Air Quality Standards (NAAQS) for PM2.5:
Primary (health-based): Annual mean 9 µg/m³ (revised 2024, down from 12 µg/m³).
Primary 24-hour: 35 µg/m³ (98th percentile averaged over 3 years).
Secondary (welfare-based): Annual 15 µg/m³, 24-hour 35 µg/m³.
Attainment designation: areas exceeding standards classified nonattainment; SIPs required.
Enforcement basis: Clean Air Act Section 109-110. Penalties up to $70,000/day for violations.""",
        {"pollutant": "PM2.5", "standard_type": "naaqs", "authority": "US EPA"}),

    DocumentChunk("EPA-02", "naaqs",
        "US EPA NAAQS — SO2 Standards",
        """US EPA NAAQS for SO2:
Primary 1-hour standard: 75 ppb (196 µg/m³) — 99th percentile, 3-year average.
Secondary (environmental): 0.5 ppm (3-hour average, for vegetation/materials protection).
SO2 nonattainment areas must implement Reasonably Available Control Technology (RACT).
Major stationary sources (>100 tons/year SO2) require Prevention of Significant Deterioration (PSD) permits.
Title IV acid rain provisions require continuous emission monitoring (CEMS) for large sources.""",
        {"pollutant": "SO2", "standard_type": "naaqs", "authority": "US EPA"}),

    DocumentChunk("EPA-03", "naaqs",
        "US EPA AQI Breakpoints and Health Communications",
        """US EPA AQI Breakpoints (PM2.5, 24-hour average):
  Good (0-50):            0-9 µg/m³
  Moderate (51-100):      9.1-35.4 µg/m³
  Unhealthy for Sensitive (101-150): 35.5-55.4 µg/m³
  Unhealthy (151-200):    55.5-150.4 µg/m³
  Very Unhealthy (201-300): 150.5-250.4 µg/m³
  Hazardous (301-500):    250.5-500.4 µg/m³
AQI > 150: sensitive groups should avoid outdoor activity.
AQI > 200: everyone should avoid prolonged outdoor exertion.
AQI > 300: health emergency — all outdoor activity should cease.""",
        {"pollutant": "PM2.5", "standard_type": "aqi", "authority": "US EPA"}),

    # ── State AQMD Emission Rules ─────────────────────────────────────────────

    DocumentChunk("AQMD-01", "state_regulation",
        "State AQMD Rule 1001 — Emission Limits for Industrial Sources",
        """State Air Quality Management District Rule 1001 (Industrial Source Emissions):
SO2 emission limit for industrial boilers: 100 mg/m³ (hourly average).
PM2.5 fugitive dust limit: 50 mg/m³ at fence line.
NOx emission limit for combustion processes: 150 mg/m³.
Continuous Emission Monitoring Systems (CEMS) required for sources > 50 tons/year.
Permit exceedance reporting: operator must notify AQMD within 2 hours of discovery.
Emergency shutdown order authority: AQMD Enforcement Division, under Section 42400.
Penalties: first violation $1,000-$10,000; repeat violations $10,000-$100,000/day.""",
        {"rule": "AQMD-1001", "standard_type": "permit", "authority": "State AQMD"}),

    DocumentChunk("AQMD-02", "state_regulation",
        "State AQMD Rule 2-1 — Agricultural Burn Permits",
        """State AQMD Rule 2-1 (Agricultural Burning):
Open burning requires a burn permit issued by AQMD at least 24 hours in advance.
Burning prohibited on Spare the Air days (AQI forecast > 100) or during air quality episodes.
Burn permit may be revoked by AQMD officer if AQI at nearest monitor exceeds 150.
Permit_exceeded compliance status means burning without permit or in excess of permitted tonnage.
Violations subject to $1,000/day fine plus cost of AQMD emergency response.
Agricultural operators must maintain 3-year records of burn activities.""",
        {"rule": "AQMD-2-1", "standard_type": "permit", "authority": "State AQMD"}),

    DocumentChunk("AQMD-03", "state_regulation",
        "State AQMD Rule 4-12 — Low Emission Zone Procedures",
        """State AQMD Rule 4-12 (Low Emission Zone — LEZ):
LEZ may be activated by city transport authority when 24-hr PM2.5 forecast > 35 µg/m³.
Pre-Euro 5 diesel vehicles (registered before 2011) prohibited in LEZ zones.
HGV curfew: vehicles > 3.5 tonnes prohibited 06:00-22:00 during active episodes.
Enforcement: ANPR cameras at zone boundaries; £200 penalty for unauthorized entry.
LEZ deactivation: when 3 consecutive hours of AQI < 100 at all monitors in zone.
Zone extension authority: AQMD Director may extend to additional districts under Section 4.2.""",
        {"rule": "AQMD-4-12", "standard_type": "traffic_control", "authority": "State AQMD"}),

    # ── Metro City Air Pollution Episode Plan ─────────────────────────────────

    DocumentChunk("EPPLAN-01", "episode_plan",
        "Metro City Air Pollution Episode Plan — Stage I (Watch)",
        """Metro City Air Pollution Episode Plan — Stage I (WATCH):
Trigger: AQI forecast 151-200 OR measured AQI > 150 at 3+ stations for 2+ consecutive hours.
Required actions:
  - AQMD issues public advisory via AirNow app and city website
  - Voluntary curtailment request to industrial facilities > 50 tons/year
  - City transport activates LEZ for downtown core
  - Emergency services placed on standby
  - Hospital network notified (ELEVATED level)
Duration: remains in effect until AQI drops below 150 at all stations for 3 consecutive hours.
Responsible authority: AQMD Director and City Emergency Management coordinator.""",
        {"stage": 1, "trigger_aqi": 150, "plan_type": "episode"}),

    DocumentChunk("EPPLAN-02", "episode_plan",
        "Metro City Air Pollution Episode Plan — Stage II (Alert)",
        """Metro City Air Pollution Episode Plan — Stage II (ALERT):
Trigger: AQI forecast 201-300 OR measured AQI > 200 at 2+ stations for 1+ consecutive hours.
Required actions:
  - AQMD issues mandatory curtailment orders to all non-compliant sources
  - City transport activates full HGV ban on all arterial roads 06:00-22:00
  - School outdoor activities suspended in affected districts
  - Public health alert issued (WARNING severity) via all channels
  - Hospital network notified (HIGH level); trauma centers increase ED capacity 20%
  - Vulnerable population sheltering guidance issued
Escalation trigger: if AQI > 250 persists 2+ hours → escalate to Stage III.
Responsible authority: Mayor's Emergency Operations Center.""",
        {"stage": 2, "trigger_aqi": 200, "plan_type": "episode"}),

    DocumentChunk("EPPLAN-03", "episode_plan",
        "Metro City Air Pollution Episode Plan — Stage III (Emergency)",
        """Metro City Air Pollution Episode Plan — Stage III (EMERGENCY):
Trigger: AQI > 300 at any station OR measured hazardous category (AQI 301-500) for any duration.
Required actions:
  - AQMD issues emergency shutdown orders to ALL non-compliant industrial sources
  - Governor's office notified for potential state emergency declaration
  - All outdoor public events cancelled
  - Schools closed in affected districts
  - Shelter-in-place advisory issued to general public
  - Hospital network at CRITICAL level; activate surge capacity, specialty wards cleared
  - Public health alert: EMERGENCY severity, all channels including EAS
  - AQMD mobile monitoring units deployed to identify emission sources
  - Air quality improvement target: AQI < 150 at all stations before de-escalation
Duration: minimum 24 hours from declaration before downgrade considered.
Responsible authority: Governor's Emergency Declaration (activates CAL OES support).""",
        {"stage": 3, "trigger_aqi": 300, "plan_type": "episode"}),

    DocumentChunk("EPPLAN-04", "episode_plan",
        "Metro City Episode Plan — De-escalation Criteria",
        """Metro City Air Pollution Episode Plan — De-escalation Criteria:
Stage III → Stage II: AQI < 200 at ALL stations for 3 consecutive hours AND
  - All emergency shutdown orders confirmed implemented (CEMS verification)
  - Wind speed > 4 m/s confirmed sustained OR mixing height > 1000 m
Stage II → Stage I: AQI < 150 at ALL stations for 3 consecutive hours AND
  - Source curtailment confirmed by AQMD inspection or CEMS data
Stage I → Normal: AQI < 100 at ALL stations for 4 consecutive hours.
Note: re-escalation is possible; de-escalation does NOT automatically lift curtailment orders.""",
        {"plan_type": "episode", "direction": "de-escalation"}),

    # ── Health Impact Tables ──────────────────────────────────────────────────

    DocumentChunk("HEALTH-01", "health_tables",
        "Concentration-Response Functions — PM2.5",
        """PM2.5 Concentration-Response Functions (WHO GBD 2019 / Pope & Dockery):
Respiratory hospitalizations: 0.62% increase per 10 µg/m³ PM2.5 increase (short-term, 24-hr).
Cardiovascular hospitalizations: 0.91% increase per 10 µg/m³ PM2.5.
All-cause mortality: 0.60% increase per 10 µg/m³ (acute, same-day + next-day).
Asthma emergency visits: 2.3% increase per 10 µg/m³ PM2.5.
COPD exacerbation: 3.1% increase per 10 µg/m³ PM2.5.
Children's lung function: permanent deficit at annual mean > 15 µg/m³.
Sensitive populations (asthma, COPD, cardiovascular disease) have 3-5× higher risk multiplier.""",
        {"pollutant": "PM2.5", "table_type": "c_r_function"}),

    DocumentChunk("HEALTH-02", "health_tables",
        "Concentration-Response Functions — O3",
        """O3 Concentration-Response Functions:
Respiratory hospitalizations: 0.52% increase per 10 µg/m³ O3 (8-hour average).
Asthma emergency visits: 3.2% per 10 µg/m³ O3 (children age 5-17).
Lung function decrement: -1.5% FEV1 per 50 µg/m³ acute O3 exposure (same-day).
All-cause mortality: 0.30% per 10 µg/m³ O3 (long-term seasonal average).
Exercise amplifies O3 health effects 2-3× (increased ventilation rate).
Protective measure: remain indoors with windows closed reduces exposure ~70%.""",
        {"pollutant": "O3", "table_type": "c_r_function"}),

    DocumentChunk("HEALTH-03", "health_tables",
        "Hospital Surge Capacity Guidance",
        """Hospital Network Surge Capacity Guidance — Air Quality Episodes:
Stage I (ELEVATED): 10-15% increase in respiratory ED visits expected.
  → Pre-position nebulisers and short-acting bronchodilators (24-hour supply).
Stage II (HIGH): 20-30% increase expected.
  → Activate internal hospital surge plan; cancel elective procedures.
  → Open respiratory overflow ward (minimum 20 extra beds across network).
Stage III (CRITICAL): 40-60% increase expected; mortality risk 2-3× baseline.
  → Declare hospital emergency; request mutual aid from regional hospitals.
  → Activate mass casualty triage protocol for respiratory cases.
  → Specialist respiratory physician on call 24/7.
Discharge criteria during episodes: do not discharge COPD/asthma patients to AQI > 150.""",
        {"table_type": "hospital_surge", "authority": "Metro City Health Department"}),

    # ── Historical Episode Lessons Learned ────────────────────────────────────

    DocumentChunk("HIST-01", "historical_episodes",
        "Metro City 2019 Industrial Episode — Lessons Learned",
        """Metro City Industrial Air Quality Episode, August 2019:
Cause: Scrubber failure at Metro Chemical Works Unit A + simultaneous agricultural burns.
Peak AQI: 487 (hazardous) at STN-001. Duration: 18 hours.
Actions taken: Emergency shutdown order issued to Chemical Works at Hour 3.
Agricultural burn permit revoked at Hour 4. Stage III episode declared at Hour 2.
Lessons learned:
  1. Emergency shutdown order was delayed 3 hours due to permit verification process
     → New protocol: verbal shutdown order may be issued; written follows within 1 hour.
  2. Hospital network was not pre-notified → 34% surge hit without preparation.
     → New protocol: hospital ELEVATED alert at Stage I (not just Stage II).
  3. Public health alert went out only on AirNow app — many residents did not see it.
     → New protocol: all 5 channels (app, SMS, signage, website, media) activated at Stage I.""",
        {"year": 2019, "max_aqi": 487, "event_type": "industrial_accident"}),

    DocumentChunk("HIST-02", "historical_episodes",
        "Metro City 2022 Summer Ozone Episode — Lessons Learned",
        """Metro City Summer Ozone Episode, July 2022:
Cause: Prolonged heat wave (7 days > 38°C), stagnant high-pressure system, elevated vehicle emissions.
Peak AQI: 312 (hazardous O3) at STN-004. Duration: 4 consecutive days.
Ventilation coefficient averaged 2,100 m²/s (very poor) for 72 hours.
Actions taken: Stage II declared on Day 2. Stage III on Day 3.
HGV ban reduced heavy truck traffic 68%; PM2.5 from road transport fell 12 µg/m³.
Lessons learned:
  1. Odd-even vehicle restriction proposed but not implemented → estimated 18% AQI reduction missed.
  2. Agricultural zone burning continued for 2 days before enforcement reached field operators.
  3. Consecutive unhealthy-run detection would have triggered Stage II 14 hours earlier
     → Recommendation: sustained episode criterion = 3+ consecutive monitoring cycles > AQI 150.""",
        {"year": 2022, "max_aqi": 312, "event_type": "ozone_episode",
         "dominant_pollutant": "O3"}),

    DocumentChunk("HIST-03", "historical_episodes",
        "Effective Mitigation Actions — Evidence from Past Episodes",
        """Evidence-Based Mitigation Effectiveness (Metro City historical data):
HGV curfew (06:00-22:00): reduces roadside PM2.5 by 8-12 µg/m³; NOx by 15-20%.
Low Emission Zone (LEZ) activation: reduces PM2.5 2-5 µg/m³ (slower effect, 12-24 hr lag).
Industrial source shutdown (emergency order): reduces SO2 and PM2.5 at proximal stations
  within 2-4 hours; network-wide improvement within 6-8 hours depending on dispersion.
Agricultural burn prohibition: PM2.5 effect realised within 4 hours of enforcement.
Voluntary industrial curtailment (Stage I): reduces emissions 15-25% if compliance is high.
Odd-even vehicle restriction: estimated 12-20% traffic volume reduction; PM2.5 effect 4-8 µg/m³.
Shelter-in-place advisory: reduces personal exposure 50-70% for those who comply.""",
        {"table_type": "mitigation_effectiveness"}),
]


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python TF-IDF vector store
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    In-memory TF-IDF vector store over the KNOWLEDGE_CORPUS.

    No external libraries required.  Uses term frequency × inverse document
    frequency with cosine similarity for retrieval.

    PRODUCTION REPLACEMENT:
      Swap _tfidf_vector() for sentence-transformer embeddings.
      Swap cosine similarity for approximate nearest-neighbour (FAISS/HNSW).
      Keep the retrieve() interface unchanged.
    """

    _STOPWORDS = {
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "by","from","is","are","was","were","be","been","has","have","had",
        "it","its","this","that","as","per","not","no","do","does",
    }

    def __init__(self, corpus: List[DocumentChunk] = None):
        self._corpus   = corpus or KNOWLEDGE_CORPUS
        self._idf: Dict[str, float] = {}
        self._build_index()

    def _tokenise(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9µ²³/]+", text.lower())
        return [t for t in tokens if t not in self._STOPWORDS and len(t) > 1]

    def _build_index(self) -> None:
        """Compute IDF weights and TF-IDF vectors for all chunks."""
        N = len(self._corpus)

        # Document frequency
        df: Dict[str, int] = {}
        for chunk in self._corpus:
            terms = set(self._tokenise(chunk.title + " " + chunk.text))
            for t in terms:
                df[t] = df.get(t, 0) + 1

        # IDF
        self._idf = {t: math.log(N / v) for t, v in df.items()}

        # TF-IDF vectors
        for chunk in self._corpus:
            tokens = self._tokenise(chunk.title + " " + chunk.text)
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            n = len(tokens) or 1
            chunk._vector = {
                t: (count / n) * self._idf.get(t, 0)
                for t, count in tf.items()
            }

    def _tfidf_vector(self, text: str) -> Dict[str, float]:
        """Compute a TF-IDF query vector (no IDF re-estimation for the query)."""
        tokens = self._tokenise(text)
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = len(tokens) or 1
        return {t: (count / n) * self._idf.get(t, 0) for t, count in tf.items()}

    @staticmethod
    def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        dot = sum(v1[t] * v2[t] for t in v1 if t in v2)
        m1  = math.sqrt(sum(x * x for x in v1.values()))
        m2  = math.sqrt(sum(x * x for x in v2.values()))
        return dot / (m1 * m2 + 1e-9)

    def retrieve(self, query: str, top_k: int = 3,
                 category_filter: str = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve the top_k most relevant chunks for the query.

        Args:
            query:           free-text question
            top_k:           maximum chunks to return
            category_filter: if set, restrict to one category
                             ('who_guidelines' | 'naaqs' | 'state_regulation' |
                              'episode_plan' | 'health_tables' | 'historical_episodes')

        Returns:
            List of (DocumentChunk, similarity_score) sorted by score descending.
        """
        q_vec   = self._tfidf_vector(query)
        pool    = [c for c in self._corpus
                   if category_filter is None or c.category == category_filter]
        scored  = [(c, self._cosine(q_vec, c._vector)) for c in pool]
        scored.sort(key=lambda x: -x[1])
        return [(c, s) for c, s in scored[:top_k] if s > 0.0]

    def retrieve_text(self, query: str, top_k: int = 3,
                      category_filter: str = None) -> str:
        """
        Convenience wrapper: returns a formatted multi-chunk context string
        ready for injection into the LLM system prompt.
        """
        results = self.retrieve(query, top_k, category_filter)
        if not results:
            return f"[RAG] No relevant regulatory/health documents found for: {query}"

        lines = [f"[RAG RETRIEVED — {len(results)} document(s) for query: '{query}']"]
        for i, (chunk, score) in enumerate(results, 1):
            lines.append(
                f"\n--- [{i}] {chunk.title}  "
                f"(category: {chunk.category}, relevance: {score:.3f}) ---"
            )
            lines.append(chunk.text)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = KnowledgeBase()
