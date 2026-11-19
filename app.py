"""
HealthLens - Streamlit web app
Run:  streamlit run app.py

Tabs
1. Upload & Explain   - PDF / scanned PDF / image / text report -> structured table + explanation
2. Ask Questions      - conversational Q&A about the uploaded report
3. Compare Reports    - reports from different dates -> table, charts, plain-language description
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import explainer  # noqa: E402
from src.config import PROCESSED_DATA_DIR, SAMPLE_REPORTS_DIR  # noqa: E402
from src.pipeline import analyze_trends, compare_reports, process_medical_report  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="HealthLens", page_icon="🩺", layout="wide")

STATUS_ICON = {"NORMAL": "✅ Normal", "LOW": "🔽 Low", "HIGH": "🔼 High", "UNKNOWN": "❔ Unknown"}
SAMPLE_TREND_FILES = [PROCESSED_DATA_DIR / f"sample_patient_report_2024_0{m}_validated.json" for m in (1, 4, 7)]


@st.cache_resource(show_spinner="Loading medical knowledge base...")
def get_pipeline() -> RAGPipeline:
    pipe = RAGPipeline()
    pipe.build_index()
    return pipe


def process_upload(uploaded) -> dict:
    """Save an uploaded file to disk and run ingestion -> OCR -> extraction -> validation."""
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name
    return process_medical_report(path, save_to_json=False)


def tests_dataframe(report: dict) -> pd.DataFrame:
    df = pd.DataFrame(report["tests"])
    df["status"] = df["status"].map(lambda s: STATUS_ICON.get(s, s))
    cols = ["test_name", "value", "unit", "reference_range", "status"]
    return df[cols].rename(columns={"test_name": "Test", "value": "Value", "unit": "Unit",
                                    "reference_range": "Lab reference range", "status": "Status"})


def date_key(report: dict):
    raw = (report.get("metadata") or {}).get("report_date") or ""
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d %b %Y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return datetime.max


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
pipe = get_pipeline()
status = pipe.get_status()

with st.sidebar:
    st.title("🩺 HealthLens")
    st.caption("Multimodal RAG for simplifying and comparing medical reports")
    level = st.selectbox("Explanation level", explainer.LEVELS, format_func=str.title)
    language = st.selectbox("Language", list(explainer.LANGUAGES), format_func=lambda c: explainer.LANGUAGES[c])
    st.divider()
    st.caption(f"Embeddings: `{status['embedding_backend']}` · Indexed chunks: {status['indexed_chunks']}")
    if status["gemini_api_key_configured"]:
        st.success(f"LLM: {status['generation_model']}")
    else:
        st.warning("No GEMINI_API_KEY: using the offline explainer (full English; translated structure for "
                   "Malayalam/Hindi). Add a key to `.env` for full LLM explanations and translation.")
    st.info("HealthLens explains lab reports for education. It does not diagnose or prescribe. "
            "Always consult a qualified doctor.")

tab_explain, tab_chat, tab_compare = st.tabs(["📄 Upload & Explain", "💬 Ask Questions", "📈 Compare Reports"])

# ---------------------------------------------------------------------------
# Tab 1 - upload & explain
# ---------------------------------------------------------------------------
with tab_explain:
    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader("Upload a lab report (PDF, scanned PDF, image or text)",
                                    type=["pdf", "png", "jpg", "jpeg", "txt"])
    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.button("Use sample report")

    if uploaded is not None:
        with st.spinner("Reading report (OCR if needed)..."):
            st.session_state["report"] = process_upload(uploaded)
            st.session_state["chat"] = []
    elif use_sample:
        st.session_state["report"] = process_medical_report(
            SAMPLE_REPORTS_DIR / "sample_medical_report.pdf", save_to_json=False)
        st.session_state["chat"] = []

    report = st.session_state.get("report")
    if report is None:
        st.info("Upload a report or click **Use sample report** to begin.")
    elif not report["tests"]:
        st.error("No laboratory tests could be extracted from this file. Try a clearer scan or a different format.")
    else:
        meta, proc = report["metadata"], report["processing"]
        st.caption(f"Source: {proc['source_type']} · OCR used: {proc['ocr_used']} · Report date: {meta['report_date']}")
        s = meta["status_summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tests found", meta["total_tests"])
        c2.metric("Within range", s["NORMAL"])
        c3.metric("Below range", s["LOW"])
        c4.metric("Above range", s["HIGH"])
        st.dataframe(tests_dataframe(report), width="stretch", hide_index=True)

        with st.spinner("Generating explanation..."):
            result = pipe.explain_report(report, "Explain my report in simple language", level, language)
        st.markdown(result["answer"].answer)
        with st.expander("How was this generated?"):
            st.write(f"Model: **{result['answer'].model_used}**")
            if result["answer"].sources:
                st.write("Retrieved knowledge sources: " + ", ".join(result["answer"].sources))
            st.write("Values, units, lab ranges and status flags come from your report (extracted "
                     "deterministically); medical background comes from the retrieved knowledge base.")

# ---------------------------------------------------------------------------
# Tab 2 - conversational Q&A
# ---------------------------------------------------------------------------
with tab_chat:
    report = st.session_state.get("report")
    if report is None or not report["tests"]:
        st.info("Load a report in the first tab, then ask questions such as "
                "“What does my hemoglobin mean?” or “Explain my cholesterol”.")
    else:
        history = st.session_state.setdefault("chat", [])
        for role, text in history:
            with st.chat_message(role):
                st.markdown(text)
        question = st.chat_input("Ask about your report...")
        if question:
            history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = pipe.explain_report(report, question, level, language)["answer"].answer
                st.markdown(answer)
            history.append(("assistant", answer))

# ---------------------------------------------------------------------------
# Tab 3 - compare reports
# ---------------------------------------------------------------------------
with tab_compare:
    st.write("Upload two or more reports from different dates to see how values changed.")
    files = st.file_uploader("Reports from different dates", type=["pdf", "png", "jpg", "jpeg", "txt"],
                             accept_multiple_files=True, key="compare_files")
    use_trend_sample = st.checkbox("Use the 3 sample reports (Jan / Apr / Jul 2024)")

    reports = []
    if use_trend_sample:
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in SAMPLE_TREND_FILES if p.exists()]
    elif files and len(files) >= 2:
        with st.spinner("Reading reports..."):
            reports = [process_upload(f) for f in files]
    elif files:
        st.info("Add at least one more report to compare.")

    if len(reports) >= 2:
        reports = sorted(reports, key=date_key)
        first, last = reports[0], reports[-1]
        comparison = compare_reports(first, last, save_to_json=False)
        rows = comparison["comparisons"]

        st.subheader("Changes over time")
        table = pd.DataFrame(rows)
        if not table.empty:
            shown = ["test_name", "previous_value", "current_value", "unit", "absolute_change",
                     "percentage_change", "change_direction", "status_previous", "status_current"]
            st.dataframe(table[[c for c in shown if c in table.columns]], width="stretch", hide_index=True)

        st.subheader("In plain language")
        st.markdown(pipe.generator.describe_trends(rows, level, language))

        st.subheader("Trend charts")
        charts_dir = Path(tempfile.mkdtemp())
        trends = analyze_trends(reports=reports, generate_charts=True, save_to_json=False, charts_dir=charts_dir)
        charts = trends.get("charts_generated", [])
        if charts:
            cols = st.columns(2)
            for i, chart in enumerate(charts):
                cols[i % 2].image(chart, width="stretch")
        else:
            st.caption("No comparable numeric trends to plot.")
