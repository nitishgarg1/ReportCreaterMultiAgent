import streamlit as st

from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎", layout="wide")


# ── Helpers ───────────────────────────────────────────────
def to_text(content) -> str:
    """Gemini can return a string or a list of content blocks; normalise to str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def run_pipeline(topic: str, status_container) -> dict:
    """Same steps as run_research_pipeline, but reports progress to the UI."""
    state = {}

    # Step 1: search
    with status_container.status("Step 1 · Search agent is searching...", expanded=False) as s:
        search_agent = build_search_agent()
        result = search_agent.invoke(
            {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
        )
        state["search_results"] = to_text(result["messages"][-1].content)
        s.update(label="Step 1 · Search complete", state="complete")

    # Step 2: read / scrape
    with status_container.status("Step 2 · Reader agent is scraping top source...", expanded=False) as s:
        reader_agent = build_reader_agent()
        result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:800]}",
                    )
                ]
            }
        )
        state["scraped_content"] = to_text(result["messages"][-1].content)
        s.update(label="Step 2 · Scraping complete", state="complete")

    # Step 3: write
    with status_container.status("Step 3 · Writer is drafting the report...", expanded=False) as s:
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = to_text(
            writer_chain.invoke({"topic": topic, "research": research_combined})
        )
        s.update(label="Step 3 · Report drafted", state="complete")

    # Step 4: critic
    with status_container.status("Step 4 · Critic is reviewing the report...", expanded=False) as s:
        state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
        s.update(label="Step 4 · Review complete", state="complete")

    return state


# ── Session state ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"topic": str, "state": dict}


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🔎 Research Assistant")
    st.caption("Search → Read → Write → Critique")

    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("Past reports")
    if not st.session_state.history:
        st.write("Nothing yet.")
    for i, item in enumerate(reversed(st.session_state.history)):
        st.write(f"{len(st.session_state.history) - i}. {item['topic']}")


# ── Main area ─────────────────────────────────────────────
st.title("AI Research Assistant")
st.write("Enter a topic and the agents will research it and write a report.")

with st.form("topic_form", clear_on_submit=False):
    topic = st.text_input("Research topic", placeholder="e.g. Latest advances in solid-state batteries")
    submitted = st.form_submit_button("Start research", type="primary")


def render_result(state: dict):
    tab_report, tab_feedback, tab_search, tab_scrape = st.tabs(
        ["📄 Report", "🧐 Critic feedback", "🔍 Search results", "🌐 Scraped content"]
    )
    with tab_report:
        st.markdown(state["report"])
        st.download_button(
            "Download report (.md)",
            data=state["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )
    with tab_feedback:
        st.markdown(state["feedback"])
    with tab_search:
        st.text(state["search_results"])
    with tab_scrape:
        st.text(state["scraped_content"])


if submitted:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        progress = st.container()
        try:
            result_state = run_pipeline(topic.strip(), progress)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
        else:
            st.session_state.history.append({"topic": topic.strip(), "state": result_state})
            st.success("Done!")
            render_result(result_state)

elif st.session_state.history:
    last = st.session_state.history[-1]
    st.info(f"Showing your latest report: **{last['topic']}**")
    render_result(last["state"])