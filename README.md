# Brand Voice Content Agent

A small LangGraph agent that turns a rough content brief into polished,
on-brand copy — with a genuine self-critique-and-revise loop, not just a
single prompt call — plus ready-to-test tone variants.

## Why I built this

I was applying to a Product Manager role at Writesonic, whose core product is
AI content generation with a "Brand Voice" feature. Rather than send a
generic resume follow-up, I wanted something that showed I understand the
actual problem their product solves: getting an LLM to consistently produce
copy that fits a specific tone or brand voice, and to catch and fix its own
weak drafts rather than requiring a human to.

This also gave me a concrete, working example of agentic system design
(LangGraph, conditional branching, a real revise loop) rather than a static
one-shot prompt — the kind of experience that's come up as a gap across
several AI-product-adjacent roles I've applied to.

## How it works

```
generate_outline → generate_draft → self_critique
                                        │
                        score < 7 and revisions left?
                        ├── yes → revise_draft → self_critique  (loops)
                        └── no  → generate_variants → done
```

- **generate_outline** — turns the brief (+ optional bullet points) into a
  structured outline (headline + key beats).
- **generate_draft** — writes the full copy from the outline, in the chosen
  format and tone, honoring optional brand voice guidelines.
- **self_critique** — an LLM call acting as an editor: scores the draft 1-10
  on clarity, tone/brand-voice fit, conciseness, and call-to-action strength,
  and gives specific feedback.
- **revise_draft** — rewrites the draft to address that feedback. This loops
  back into `self_critique` again, up to a configurable revision cap
  (default 2), so the agent can genuinely improve its own output rather than
  critiquing once and moving on regardless.
- **generate_variants** — once the draft passes (or the revision cap is hit),
  produces three tone variants (Professional / Casual / Punchy) of the same
  core message, for quick A/B-style comparison.

The whole thing is built with [LangGraph](https://langchain-ai.github.io/langgraph/)'s
`StateGraph`, using a real conditional edge for the revise loop rather than a
fixed-length chain — the routing decision (`_route_after_critique` in
`graph.py`) is made by the graph itself based on the critique score.

## Project structure

```
writesonic-content-agent/
├── README.md
├── requirements.txt
├── .env.example
├── prompts.py   # prompt templates for every node
├── graph.py     # the LangGraph StateGraph — nodes, state, conditional routing
└── app.py       # Streamlit UI
```

## Setup

Both supported providers have a genuinely free tier — use whichever key you
already have, or grab one:

- **Groq** (default): free key at [console.groq.com](https://console.groq.com), no card required.
- **Gemini**: free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), no card required.

Steps:

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in the provider you're using:
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=your-key-here
   ```
   or
   ```
   LLM_PROVIDER=gemini
   GOOGLE_API_KEY=your-key-here
   ```
   (Alternatively, pick the provider and paste the key straight into the
   sidebar when the app is running — it's only kept for that session, no
   `.env` file needed.)
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
4. Or run the graph directly from the command line for a quick smoke test:
   ```
   python graph.py
   ```
   This runs a deliberately vague sample brief through the full pipeline and
   prints the trace, final draft, and variants — useful for confirming the
   critique/revise loop actually triggers a revision, not just always passing
   on the first draft.

## Swapping the LLM provider

`graph.py`'s `_get_llm()` reads the `LLM_PROVIDER` env var (`groq` or
`gemini`, default `groq`) and instantiates the matching chat model — it's
also selectable live in the Streamlit sidebar. Both are free, so pick
whichever key you have. Adding a third provider (Anthropic/OpenAI, if you
have paid API access) is a small addition to that same function —
everything else in the graph is provider-agnostic.

## Honest notes

- This was built and structurally tested (the graph compiles, all nodes
  route correctly, prompts render as expected) without a live API key in the
  build environment — the first live run with real model output should be
  done locally with your own Groq key before treating any specific example
  output as final.
- The critique step is itself an LLM call, not a hand-built rubric scorer —
  it's a reasonable, cheap way to add a self-correction loop, but it isn't a
  guaranteed-accurate judge of quality. Worth being upfront about that
  distinction if this comes up in conversation.
