"""
LangGraph state machine for the Brand Voice Content Agent.

Flow:
    generate_outline -> generate_draft -> self_critique
        -> (score < threshold and revisions left) -> revise_draft -> self_critique  [loop]
        -> (else) -> generate_variants -> END

This is a genuine agentic loop (not a fixed-length chain): the graph itself
decides, based on the critique node's score, whether to revise again or move
on, up to a configurable revision cap.
"""

from __future__ import annotations

import os
import re
from typing import TypedDict

from langgraph.graph import StateGraph, END

import prompts

SCORE_THRESHOLD = 7


class ContentState(TypedDict, total=False):
    brief: str
    bullets: str
    content_format: str
    tone: str
    brand_voice: str
    max_revisions: int

    outline: str
    draft: str
    critique_score: int
    critique_feedback: str
    revision_count: int
    history: list  # list of {"step": str, "detail": str} for the UI trace

    variants: dict  # {"professional": str, "casual": str, "punchy": str}


def _get_llm(temperature: float = 0.7):
    """
    Returns a chat model based on LLM_PROVIDER (env var, default "groq").
    Both providers have a genuinely free tier — pick whichever key you have.

    LLM_PROVIDER=groq   -> needs GROQ_API_KEY   (console.groq.com)
    LLM_PROVIDER=gemini -> needs GOOGLE_API_KEY (aistudio.google.com/apikey)
    """
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and put it in your .env file "
                "(see .env.example), or paste it in the sidebar."
            )
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)

    # default: groq
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and put it in a .env file (see .env.example), or paste it in the sidebar."
        )
    model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(model=model_name, temperature=temperature, api_key=api_key)


def _log(state: ContentState, step: str, detail: str) -> list:
    history = list(state.get("history", []))
    history.append({"step": step, "detail": detail})
    return history


def _extract_text(content) -> str:
    """
    Normalize a chat model's `.content` into a plain string.

    Depending on the LangChain/provider version, `.content` can be either
    a plain string OR a list of content-block dicts, e.g.
    [{"type": "text", "text": "..."}] — this shows up especially with
    langchain-google-genai on newer langchain-core releases. Handling both
    shapes here means every node can just call llm.invoke(prompt) and get
    a string back, regardless of provider/version.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return "".join(parts)
    return str(content)


def generate_outline(state: ContentState) -> ContentState:
    llm = _get_llm()
    prompt = prompts.outline_prompt(
        brief=state["brief"],
        bullets=state.get("bullets", ""),
        content_format=state["content_format"],
        tone=state["tone"],
    )
    outline = _extract_text(llm.invoke(prompt).content).strip()
    return {
        "outline": outline,
        "history": _log(state, "generate_outline", outline),
    }


def generate_draft(state: ContentState) -> ContentState:
    llm = _get_llm()
    prompt = prompts.draft_prompt(
        outline=state["outline"],
        content_format=state["content_format"],
        tone=state["tone"],
        brand_voice=state.get("brand_voice", ""),
    )
    draft = _extract_text(llm.invoke(prompt).content).strip()
    return {
        "draft": draft,
        "revision_count": state.get("revision_count", 0),
        "history": _log(state, "generate_draft", draft),
    }


def self_critique(state: ContentState) -> ContentState:
    llm = _get_llm(temperature=0.2)
    prompt = prompts.critique_prompt(
        draft=state["draft"],
        content_format=state["content_format"],
        tone=state["tone"],
        brand_voice=state.get("brand_voice", ""),
    )
    response = _extract_text(llm.invoke(prompt).content).strip()

    score_match = re.search(r"SCORE:\s*(\d+)", response)
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", response, re.DOTALL)
    score = int(score_match.group(1)) if score_match else 5
    feedback = feedback_match.group(1).strip() if feedback_match else response

    return {
        "critique_score": score,
        "critique_feedback": feedback,
        "history": _log(state, "self_critique", f"Score: {score}/10 — {feedback}"),
    }


def revise_draft(state: ContentState) -> ContentState:
    llm = _get_llm()
    prompt = prompts.revise_prompt(
        draft=state["draft"],
        feedback=state["critique_feedback"],
        content_format=state["content_format"],
        tone=state["tone"],
    )
    revised = _extract_text(llm.invoke(prompt).content).strip()
    revision_count = state.get("revision_count", 0) + 1
    return {
        "draft": revised,
        "revision_count": revision_count,
        "history": _log(state, "revise_draft", revised),
    }


def generate_variants(state: ContentState) -> ContentState:
    llm = _get_llm()
    prompt = prompts.variants_prompt(draft=state["draft"], content_format=state["content_format"])
    response = _extract_text(llm.invoke(prompt).content).strip()

    def _extract(label: str) -> str:
        pattern = rf"{label}:\s*(.+?)(?=\n[A-Z]+:|\Z)"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else ""

    variants = {
        "professional": _extract("PROFESSIONAL"),
        "casual": _extract("CASUAL"),
        "punchy": _extract("PUNCHY"),
    }
    return {
        "variants": variants,
        "history": _log(state, "generate_variants", "3 tone variants generated"),
    }


def _route_after_critique(state: ContentState) -> str:
    score = state.get("critique_score", 10)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)
    if score < SCORE_THRESHOLD and revision_count < max_revisions:
        return "revise_draft"
    return "generate_variants"


def build_graph():
    graph = StateGraph(ContentState)

    graph.add_node("generate_outline", generate_outline)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("self_critique", self_critique)
    graph.add_node("revise_draft", revise_draft)
    graph.add_node("generate_variants", generate_variants)

    graph.set_entry_point("generate_outline")
    graph.add_edge("generate_outline", "generate_draft")
    graph.add_edge("generate_draft", "self_critique")
    graph.add_conditional_edges(
        "self_critique",
        _route_after_critique,
        {"revise_draft": "revise_draft", "generate_variants": "generate_variants"},
    )
    graph.add_edge("revise_draft", "self_critique")
    graph.add_edge("generate_variants", END)

    return graph.compile()


if __name__ == "__main__":
    # Quick smoke test: run end-to-end with a deliberately vague brief so the
    # critique/revise loop has a real chance to trigger at least once.
    from dotenv import load_dotenv

    load_dotenv()

    app = build_graph()
    initial_state: ContentState = {
        "brief": "a new AI feature that helps people write better emails",
        "bullets": "",
        "content_format": "Ad copy",
        "tone": "confident and playful, for a Gen-Z audience",
        "brand_voice": "",
        "max_revisions": 2,
        "revision_count": 0,
        "history": [],
    }
    final_state = app.invoke(initial_state)

    print("\n=== TRACE ===")
    for step in final_state["history"]:
        print(f"[{step['step']}] {step['detail'][:200]}")

    print("\n=== FINAL DRAFT ===")
    print(final_state["draft"])

    print("\n=== VARIANTS ===")
    for k, v in final_state["variants"].items():
        print(f"\n-- {k.upper()} --\n{v}")
