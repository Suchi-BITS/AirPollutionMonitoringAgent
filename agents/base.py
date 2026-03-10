# agents/base.py
# LLM invocation helpers shared by all agents.
# DEMO MODE: if no OpenAI API key is present, returns structured placeholder
# text so the full pipeline runs end-to-end without any API calls.

import os
import json
import inspect
from config.settings import air_config


def _demo_mode() -> bool:
    return not bool(air_config.openai_api_key.strip().startswith("sk-"))


def _call_tool(t, scenario="standard"):
    """
    Call a tool (LangChain @tool wrapper or plain function) with sensible defaults.
    The scenario parameter overrides the tool's declared default for scenario args.
    """
    t_name = getattr(t, "name", None) or getattr(t, "__name__", str(t))
    try:
        fn  = getattr(t, "func", t)
        sig = inspect.signature(fn)
        kwargs = {}
        for pname, param in sig.parameters.items():
            if pname == "scenario":
                kwargs[pname] = scenario      # always use caller-supplied scenario
            elif param.default is not inspect.Parameter.empty:
                kwargs[pname] = param.default
            elif pname == "limit":
                kwargs[pname] = 20
        if hasattr(t, "invoke"):
            result = t.invoke(kwargs)
        else:
            result = fn(**kwargs)
        return t_name, {"tool": t_name, "result": result}
    except Exception as ex:
        return t_name, {"tool": t_name, "error": str(ex)}


def call_llm(system_prompt: str, user_message: str, max_tokens: int = 1200) -> str:
    """
    Call the LLM with a system + user message pair.
    Falls back to demo mode if no API key configured.
    """
    if _demo_mode():
        return (
            "[DEMO MODE — configure OPENAI_API_KEY in .env for live LLM reasoning]\n\n"
            f"System role: {system_prompt[:120]}...\n\n"
            f"Input summary: {user_message[:300]}...\n\n"
            "Analysis: Based on the provided data, this agent would produce a detailed "
            "scientific analysis with specific pollutant readings, trend identification, "
            "source attribution, and actionable recommendations formatted as a structured report."
        )

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(
            model=air_config.model_name,
            temperature=air_config.temperature,
            api_key=air_config.openai_api_key,
            max_tokens=max_tokens,
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        return response.content
    except Exception as e:
        return f"[LLM ERROR: {e}]\n\nFallback: analysis unavailable for this agent turn."


def call_llm_with_tools(system_prompt: str, user_message: str, tools: list,
                        max_iterations: int = 5) -> tuple:
    """
    Run an agent loop: LLM with bound tools, executing tool calls until the
    model produces a final text response (no more tool calls).

    In demo mode: calls every tool directly with default args, returns their
    results so downstream state-extraction logic can find and use real data.

    Returns:
        (final_text_response: str, tool_results: list of dicts)
    """
    # Detect the scenario from user_message to pass through to tools
    scenario = "episode" if "episode" in user_message.lower() else "standard"

    if _demo_mode():
        tool_results = []
        for t in tools:
            _, record = _call_tool(t, scenario=scenario)
            tool_results.append(record)

        analysis = (
            "[DEMO MODE — configure OPENAI_API_KEY in .env for live LLM reasoning]\n\n"
            f"This agent executed {len(tool_results)} tool call(s) with the simulation data layer. "
            "With a live LLM, it would analyze the returned data to produce domain-specific findings, "
            "source attribution, risk scores, and prioritized action items."
        )
        return analysis, tool_results

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

        llm = ChatOpenAI(
            model=air_config.model_name,
            temperature=air_config.temperature,
            api_key=air_config.openai_api_key,
            max_tokens=1400,
        )
        llm_with_tools = llm.bind_tools(tools)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        tool_results = []

        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content, tool_results

            for tc in response.tool_calls:
                tool_map = {t.name: t for t in tools}
                t = tool_map.get(tc["name"])
                if t:
                    try:
                        result = t.invoke(tc["args"])
                        result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
                    except Exception as ex:
                        result_str = f"Tool error: {ex}"
                        result = {"error": str(ex)}
                    tool_results.append({"tool": tc["name"], "args": tc["args"], "result": result})
                    messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

        last_text = next(
            (m.content for m in reversed(messages)
             if hasattr(m, "content") and isinstance(m.content, str) and m.content),
            "Agent reached maximum tool iterations without producing a final response."
        )
        return last_text, tool_results

    except Exception as e:
        return f"[LLM ERROR: {e}]", []
