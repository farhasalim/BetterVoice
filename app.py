"""
Streamlit UI for the Better Voice Agent.

Run with: streamlit run app.py
Requires a GROQ_API_KEY in a .env file (see .env.example) or entered in the
sidebar at runtime.
"""

import os
from datetime import date

import streamlit as st
from dotenv import load_dotenv

from graph import build_graph

load_dotenv()

# --- Usage caps ---
# Two layers, both intentionally simple (no database):
#   1. SESSION_LIMIT — stops one visitor from looping generations endlessly
#      in a single browser session (st.session_state, per-visitor).
#   2. DAILY_LIMIT — a shared cap across ALL visitors to this running app
#      instance, so a demo getting shared around doesn't blow through your
#      free-tier API quota for the day. st.cache_resource gives every
#      visitor's session a reference to the SAME dict, which is exactly
#      what a shared counter needs. It resets whenever the app restarts/
#      redeploys (normal on Streamlit Community Cloud) — that's fine for a
#      lightweight safeguard, not a hard guarantee.
SESSION_LIMIT = 5
DAILY_LIMIT = 25


@st.cache_resource
def _usage_counter():
    return {"date": None, "count": 0}


def _daily_limit_ok() -> bool:
    counter = _usage_counter()
    today = date.today().isoformat()
    if counter["date"] != today:
        counter["date"] = today
        counter["count"] = 0
    if counter["count"] >= DAILY_LIMIT:
        return False
    counter["count"] += 1
    return True

# On Streamlit Community Cloud, secrets are set in the app's dashboard
# ("Secrets" section, TOML format) and read via st.secrets rather than a
# .env file. Copy any of ours into os.environ so the rest of the app works
# unchanged whether it's running locally (.env) or deployed (Cloud secrets).
try:
    for _key in ("LLM_PROVIDER", "GROQ_API_KEY", "GOOGLE_API_KEY", "GROQ_MODEL", "GEMINI_MODEL"):
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass  # no secrets.toml / Cloud secrets configured — fine when running locally with just .env

st.set_page_config(page_title="Brand Voice Content Agent", page_icon="✍️", layout="wide")

st.title("✍️ Brand Voice Content Agent")
st.caption(
    "A small LangGraph agent that outlines, drafts, self-critiques, and revises "
    "content before producing tone variants — built as a hands-on exploration "
    "of the kind of AI content workflow Writesonic's product is built around."
)

# --- Sidebar: config ---
with st.sidebar:
    st.header("Settings")

    default_provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    provider_label = st.radio(
        "LLM provider",
        options=["Groq", "Gemini"],
        index=0 if default_provider != "gemini" else 1,
        help="Both have a genuinely free tier — pick whichever key you have.",
    )
    provider = provider_label.lower()
    os.environ["LLM_PROVIDER"] = provider

    # IMPORTANT: never pass an existing secret as `value=` to a text_input,
    # even one with type="password". Password-type inputs only mask what's
    # displayed on screen — the value is still written into the page's raw
    # HTML, visible to anyone via "View Page Source" or browser dev tools.
    # So: if a key is already configured (local .env, or Streamlit Cloud
    # secrets), just say so and leave the field blank; typing something
    # only overrides it for this browser session, never displayed back.
    key_env_var = "GOOGLE_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
    key_label = "Google (Gemini)" if provider == "gemini" else "Groq"
    key_url = "aistudio.google.com/apikey" if provider == "gemini" else "console.groq.com"

    configured_key = os.environ.get(key_env_var, "")
    if configured_key:
        st.success(f"{key_label} API key loaded from configured secrets.")
    api_key_input = st.text_input(
        f"{key_label} API key" + (" (override, optional)" if configured_key else ""),
        value="",
        type="password",
        help=f"Free key from {key_url}. Never pre-filled, never displayed — only kept in memory for this session.",
    )
    if api_key_input:
        os.environ[key_env_var] = api_key_input

    max_revisions = st.slider("Max self-revision rounds", min_value=0, max_value=3, value=2)

    st.divider()
    st.caption(
        "Workflow: outline → draft → self-critique → revise (loops until it "
        "scores 7+/10 or hits the revision cap) → tone variants."
    )

# --- Main form ---
col1, col2 = st.columns(2)

with col1:
    brief = st.text_area(
        "Content brief *",
        placeholder="e.g. Announce a new AI feature that summarizes long email threads into one-line action items.",
        height=100,
    )
    bullets = st.text_area(
        "Optional key points (one per line)",
        placeholder="- Saves ~20 minutes a day\n- Works inside Gmail and Outlook\n- Free for the first 3 months",
        height=100,
    )

with col2:
    content_format = st.selectbox(
        "Content format",
        ["Blog intro", "Product description", "Social media caption", "Ad copy"],
    )
    tone = st.text_input(
        "Target tone / audience",
        placeholder="e.g. confident and playful, for a Gen-Z audience",
    )
    brand_voice = st.text_area(
        "Optional brand voice guidelines",
        placeholder="e.g. Warm but not corny. Short sentences. Never uses exclamation points more than once per piece.",
        height=68,
    )

run = st.button("Generate", type="primary", use_container_width=True)

if run:
    key_present = (
        os.environ.get("GOOGLE_API_KEY") if provider == "gemini" else os.environ.get("GROQ_API_KEY")
    )
    if not key_present:
        st.error(
            "Add your Gemini API key in the sidebar first (it's free — aistudio.google.com/apikey)."
            if provider == "gemini"
            else "Add your Groq API key in the sidebar first (it's free — console.groq.com)."
        )
    elif not brief.strip():
        st.error("Content brief is required.")
    elif not tone.strip():
        st.error("Target tone / audience is required.")
    elif st.session_state.get("session_run_count", 0) >= SESSION_LIMIT:
        st.warning(
            f"You've reached the demo limit of {SESSION_LIMIT} runs for this session. "
            "Refresh the page to reset, or clone the repo and run it locally with your own free API key."
        )
    elif not _daily_limit_ok():
        st.warning(
            "This demo has hit its shared daily limit — keeping the free API tier "
            "available for everyone who visits. Please check back tomorrow, or clone "
            "the repo and run it locally with your own free API key."
        )
    else:
        st.session_state["session_run_count"] = st.session_state.get("session_run_count", 0) + 1
        graph_app = build_graph()
        initial_state = {
            "brief": brief,
            "bullets": bullets,
            "content_format": content_format,
            "tone": tone,
            "brand_voice": brand_voice,
            "max_revisions": max_revisions,
            "revision_count": 0,
            "history": [],
        }

        trace_placeholder = st.empty()
        trace_lines = []
        final_state = None

        with st.spinner("Running the agent..."):
            for event in graph_app.stream(initial_state, stream_mode="values"):
                final_state = event
                history = event.get("history", [])
                if history:
                    last = history[-1]
                    label = {
                        "generate_outline": "Outline generated",
                        "generate_draft": "Draft generated",
                        "self_critique": "Critique",
                        "revise_draft": "Revising draft",
                        "generate_variants": "Tone variants generated",
                    }.get(last["step"], last["step"])
                    trace_lines.append(f"**{label}**")
                    trace_placeholder.markdown("\n\n".join(trace_lines))

        if final_state:
            st.success("Done.")

            st.subheader("Final approved copy")
            st.markdown(f"> {final_state['draft']}")

            st.subheader("Tone variants")
            variants = final_state.get("variants", {})
            tabs = st.tabs(["Professional", "Casual", "Punchy"])
            for tab, key in zip(tabs, ["professional", "casual", "punchy"]):
                with tab:
                    st.write(variants.get(key, "(not generated)"))

            with st.expander("How this was generated (critique/revision trace)"):
                for step in final_state["history"]:
                    st.markdown(f"**{step['step']}**")
                    st.text(step["detail"][:1000])
                    st.markdown("---")