# rag/rag_retriever.py  — AirGuard v4
#
# ══════════════════════════════════════════════════════════════════════════════
# RAG RETRIEVER — Query Routing and Context Injection
# ══════════════════════════════════════════════════════════════════════════════
#
# RAGRetriever sits between the running agent state and the KnowledgeBase.
# It answers the question: "Given what the agent has just observed, which
# regulatory documents and health tables should be injected into context?"
#
# ── THREE RETRIEVAL STRATEGIES ────────────────────────────────────────────────
#
# 1. AUTOMATIC RETRIEVAL (called by ContextEngineer)
#    Runs before each ReAct loop.  Inspects AQI levels, dominant pollutant,
#    and episode status from memory, then queries multiple categories to build
#    a comprehensive context block.
#
#    Example: AQI=350, dominant=SO2, emergency=True
#    → retrieves: WHO SO2 guideline, AQMD Rule 1001, Stage III episode plan,
#                 health tables for SO2/PM2.5, historical industrial episode
#
# 2. ON-DEMAND RETRIEVAL (LangChain tool: query_knowledge_base)
#    The LLM can call this tool during the ReAct loop when it needs to look up
#    a specific regulation, threshold, or protocol.
#
#    Example tool call: query_knowledge_base(query="Stage II de-escalation criteria",
#                                            category="episode_plan")
#
# 3. ENFORCEMENT RETRIEVAL (called before issuing regulatory actions)
#    Ensures the exact regulatory basis for any enforcement action is grounded
#    in retrieved documents, not hallucinated by the LLM.
#
# ── INTEGRATION POINTS ────────────────────────────────────────────────────────
#
#   context_engineer.py:
#     context_block = RAG_RETRIEVER.build_rag_context(max_aqi, dominant_pol, emergency)
#     → injected as a section in the system prompt
#
#   react_agent.py:
#     ALL_TOOLS includes query_knowledge_base (the on-demand tool)
#
#   agents/react_agent.py _demo_react_run():
#     prints retrieved chunks to show what context was injected

from typing import List, Optional, Tuple
from rag.knowledge_base import KNOWLEDGE_BASE, DocumentChunk


class RAGRetriever:
    """
    Orchestrates retrieval from the KnowledgeBase for different use cases.
    """

    # How many chunks to include per category in automatic retrieval
    AUTO_TOP_K = 2

    def build_rag_context(self,
                          max_aqi: int,
                          dominant_pollutant: str,
                          emergency: bool,
                          non_compliant_sources: List[dict] = None,
                          episode_stage: int = 0) -> str:
        """
        Build a multi-section RAG context block for injection into the system prompt.

        Called once per run by ContextEngineer.build_system_prompt().
        Retrieves from 4-5 categories based on current conditions.

        Returns:
            Formatted string, ready for {rag_context} placeholder in system prompt.
        """
        blocks: List[str] = [
            "══════════════════════════════════════════════",
            "RETRIEVED REGULATORY & HEALTH KNOWLEDGE (RAG)",
            "══════════════════════════════════════════════",
        ]

        # 1. WHO guideline for the dominant pollutant
        pol_query = f"{dominant_pollutant} air quality guideline concentration limit health"
        who_results = KNOWLEDGE_BASE.retrieve(pol_query, top_k=1, category_filter="who_guidelines")
        if who_results:
            blocks.append(self._format_chunk(who_results[0], "WHO Guideline"))

        # 2. EPA AQI breakpoints (always relevant)
        aqi_results = KNOWLEDGE_BASE.retrieve(
            "AQI breakpoints health category unhealthy hazardous emergency",
            top_k=1, category_filter="naaqs"
        )
        if aqi_results:
            blocks.append(self._format_chunk(aqi_results[0], "EPA AQI"))

        # 3. Applicable regulatory rule
        reg_query = (
            "industrial source emission permit limit violation enforcement shutdown"
            if any((s or {}).get("source_category") == "industrial"
                   for s in (non_compliant_sources or []))
            else "agricultural burning permit regulation violation"
        )
        aqmd_results = KNOWLEDGE_BASE.retrieve(
            reg_query, top_k=1, category_filter="state_regulation"
        )
        if aqmd_results:
            blocks.append(self._format_chunk(aqmd_results[0], "AQMD Rule"))

        # 4. Episode plan for current stage
        if emergency or max_aqi > 200:
            stage_query = "Stage III emergency shutdown hazardous AQI 300 hospital surge"
        elif max_aqi > 150:
            stage_query = "Stage II alert mandatory curtailment AQI 200 hospital HIGH"
        else:
            stage_query = "Stage I watch advisory voluntary curtailment AQI 150"
        ep_results = KNOWLEDGE_BASE.retrieve(stage_query, top_k=1, category_filter="episode_plan")
        if ep_results:
            blocks.append(self._format_chunk(ep_results[0], "Episode Protocol"))

        # 5. Health tables for concentration-response at current AQI
        health_query = (
            f"{dominant_pollutant} concentration response hospital mortality asthma COPD"
        )
        health_results = KNOWLEDGE_BASE.retrieve(
            health_query, top_k=1, category_filter="health_tables"
        )
        if health_results:
            blocks.append(self._format_chunk(health_results[0], "Health C-R Function"))

        # 6. Historical lesson if sustained emergency
        if emergency or episode_stage >= 2:
            hist_results = KNOWLEDGE_BASE.retrieve(
                "episode lessons learned mitigation effective shutdown delay",
                top_k=1, category_filter="historical_episodes"
            )
            if hist_results:
                blocks.append(self._format_chunk(hist_results[0], "Historical Lesson"))

        blocks.append("══════════════════════════════════════════════")
        blocks.append(
            "Use the retrieved documents above as the authoritative source for "
            "regulatory thresholds, enforcement powers, and health risk estimates. "
            "Do NOT invent limit values that differ from the retrieved documents."
        )
        blocks.append("══════════════════════════════════════════════")

        return "\n".join(blocks)

    def retrieve_for_enforcement(self, source_category: str,
                                 pollutant: str, action_type: str) -> str:
        """
        Retrieve the specific regulatory basis for an enforcement action.
        Called before issuing issue_regulatory_action() to ground the
        regulatory_basis argument in retrieved documents.
        """
        query = (
            f"{source_category} {pollutant} {action_type} permit limit "
            "enforcement authority regulatory rule"
        )
        results = KNOWLEDGE_BASE.retrieve(
            query, top_k=2, category_filter="state_regulation"
        )
        if not results:
            results = KNOWLEDGE_BASE.retrieve(query, top_k=2)

        lines = []
        for chunk, score in results:
            lines.append(f"[{chunk.doc_id}] {chunk.title}: {chunk.text[:200]}...")
        return " | ".join(lines) if lines else "State AQMD Rule 1001 — Emission Limits"

    def on_demand_query(self, query: str,
                        category: Optional[str] = None,
                        top_k: int = 3) -> str:
        """
        Answer an ad-hoc query from the LLM during the ReAct loop.
        Exposed as the query_knowledge_base LangChain tool.
        """
        results = KNOWLEDGE_BASE.retrieve(query, top_k=top_k, category_filter=category)
        if not results:
            return (
                f"No documents found for query: '{query}'. "
                "Available categories: who_guidelines, naaqs, state_regulation, "
                "episode_plan, health_tables, historical_episodes."
            )
        lines = [f"KNOWLEDGE BASE — {len(results)} result(s) for: '{query}'\n"]
        for chunk, score in results:
            lines.append(
                f"[{chunk.doc_id}] {chunk.title}  (score: {score:.3f})\n"
                f"{chunk.text}\n"
                f"Metadata: {chunk.metadata}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_chunk(chunk_score: Tuple[DocumentChunk, float], label: str) -> str:
        chunk, score = chunk_score
        return (
            f"\n[{label} | {chunk.doc_id} | relevance: {score:.3f}]\n"
            f"{chunk.text}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────
RAG_RETRIEVER = RAGRetriever()
