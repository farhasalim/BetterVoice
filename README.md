BetterVoice

A small LangGraph agent that turns a rough content brief into polished, on-brand copy with a genuine self-critique-and-revise loop, with multiple tone variants.

Why?

AI content-generation tools are everywhere, but most treat quality control as an afterthought: generate once, hand it to a human to fix. I wanted to explore what it looks like to build that quality-control step into the generation pipeline itself — an agent that drafts, evaluates its own output against a concrete rubric, and revises before anything reaches the user.

I also gained my first insight into agentic system design (LangGraph, conditional branching, a revise loop)!

How it works:

generate_outline → generate_draft → self_critique
│
score < 7 and revisions left?
├── yes → revise_draft → self_critique (loops)
└── no → generate_variants → done

generate_outline — turns the brief (+ optional bullet points) into a structured outline (headline + key beats).

generate_draft — writes the full copy from the outline, in the chosen format and tone, honoring optional brand voice guidelines.

self_critique — an LLM call acting as an editor: scores the draft 1-10 on clarity, tone/brand-voice fit, conciseness, and call-to-action strength, and gives specific feedback.

revise_draft — rewrites the draft to address that feedback. This loops back into self_critique again, up to a configurable revision cap (default 2), so the agent can genuinely improve its own output rather than critiquing once and moving on regardless.
generate_variants — once the draft passes (or the revision cap is hit), produces three tone variants (Professional / Casual / Punchy) of the same core message, for quick A/B-style comparison.

The whole thing is built with LangGraph's StateGraph, using a real conditional edge for the revise loop rather than a fixed-length chain — the routing decision (\_route_after_critique in graph.py) is made by the graph itself based on the critique score.

Project structure
BetterVoice/
├── README.md
├── requirements.txt
├── .env.example
├── prompts.py # prompt templates for every node
├── graph.py # the LangGraph StateGraph — nodes, state, conditional routing
└── app.py # Streamlit UI
Setup

Both supported providers have a genuinely free tier — use whichever key you already have, or grab one:

Groq (default): free key at console.groq.com, no card required.
Gemini: free key at aistudio.google.com/apikey, no card required.

Steps:

1. pip install -r requirements.txt
2. Copy .env.example to .env and fill in the provider you're using:
   Example:
   LLM_PROVIDER=gemini
   GOOGLE_API_KEY=your-key-here

3. Run the Streamlit app:

   streamlit run app.py
