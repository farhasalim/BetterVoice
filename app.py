"""
Streamlit UI for the Brand Voice Content Agent.

Run with: streamlit run app.py
Requires a GROQ_API_KEY in a .env file (see .env.example) or entered in the
sidebar at runtime.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from graph import build_graph

load_dotenv()

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

    if provider == "gemini":
        env_key = os.environ.get("GOOGLE_API_KEY", "")
        api_key_input = st.text_input(
            "Google (Gemini) API key",
            value=env_key,
            type="password",
            help="Free key from aistudio.google.com/apikey. Only stored for this session.",
        )
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input
    else:
        env_key = os.environ.get("GROQ_API_KEY", "")
        api_key_input = st.text_input(
            "Groq API key",
            value=env_key,
            type="password",
            help="Free key from console.groq.com. Only stored for this session.",
        )
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input

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
    else:
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
