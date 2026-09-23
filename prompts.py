"""
Prompt templates for each node in the content-generation LangGraph agent.

Each function returns a plain string prompt (rather than a LangChain
PromptTemplate object) so the graph nodes can build them with simple f-strings.
This keeps the wiring in graph.py easy to read.
"""

from __future__ import annotations


def outline_prompt(brief: str, bullets: str, content_format: str, tone: str) -> str:
    bullets_block = f"\nKey points to include:\n{bullets}" if bullets.strip() else ""
    return f"""You are a senior content strategist. Produce a short, structured outline
for a piece of {content_format} content.

Brief: {brief}{bullets_block}
Target tone/audience: {tone}

Return the outline as:
1. Headline (one strong option)
2. 3-5 bullet points covering the key beats the copy should hit, in order.

Keep it tight — this is an outline, not the final copy. No preamble, just the outline."""


def draft_prompt(
    outline: str,
    content_format: str,
    tone: str,
    brand_voice: str,
) -> str:
    brand_voice_block = (
        f"\nBrand voice guidelines to follow closely:\n{brand_voice}\n"
        if brand_voice.strip()
        else ""
    )
    format_length_hint = {
        "Blog intro": "120-180 words",
        "Product description": "60-120 words",
        "Social media caption": "20-60 words",
        "Ad copy": "25-60 words, punchy",
    }.get(content_format, "100-150 words")

    return f"""You are an expert copywriter. Write the final {content_format} copy based
on this outline:

{outline}

Target tone/audience: {tone}{brand_voice_block}
Length guidance: {format_length_hint}

Write only the final copy — no headers, no meta-commentary, no explanation of
what you did. It should read as finished, publish-ready content."""


def critique_prompt(draft: str, content_format: str, tone: str, brand_voice: str) -> str:
    brand_voice_block = (
        f"\nIt is also being checked against these brand voice guidelines:\n{brand_voice}\n"
        if brand_voice.strip()
        else ""
    )
    return f"""You are a meticulous editor reviewing a piece of {content_format} copy.
Target tone/audience: {tone}{brand_voice_block}

Copy to review:
---
{draft}
---

Score it 1-10 on each of these, then give one overall score (1-10, the
minimum of the four, rounded to the nearest whole number is fine as a
shortcut) and 2-4 sentences of specific, actionable feedback:
- Clarity (is the message immediately understandable?)
- Tone/brand-voice fit (does it match the requested tone/voice?)
- Conciseness (no filler, no redundant phrases?)
- Call-to-action strength (does it prompt a clear next action, where relevant?)

Respond in EXACTLY this format, nothing else:
SCORE: <integer 1-10>
FEEDBACK: <your specific, actionable feedback in 2-4 sentences>"""


def revise_prompt(draft: str, feedback: str, content_format: str, tone: str) -> str:
    return f"""You are an expert copywriter revising a piece of {content_format} copy
based on editor feedback.

Original copy:
---
{draft}
---

Editor feedback to address:
{feedback}

Target tone/audience: {tone}

Rewrite the copy to directly address the feedback. Write only the final
revised copy — no headers, no meta-commentary, no explanation of changes."""


def variants_prompt(draft: str, content_format: str) -> str:
    return f"""You are a copywriter producing tone variants of an approved piece of
{content_format} copy, for A/B testing.

Approved copy:
---
{draft}
---

Produce exactly 3 variants that carry the same core message but in these
distinct tones:
1. Professional — polished, confident, business-appropriate.
2. Casual — warm, conversational, like a friendly human wrote it.
3. Punchy — short, high-energy, attention-grabbing (fine to be shorter than the original).

Respond in EXACTLY this format, nothing else:
PROFESSIONAL: <variant text>
CASUAL: <variant text>
PUNCHY: <variant text>"""
