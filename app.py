"""RecruitRAG web UI.  Run:  streamlit run app.py"""
import os
import tempfile

import streamlit as st

from src.factory import create_pipeline

st.set_page_config(page_title="RecruitRAG", page_icon="🧑‍💼", layout="wide")


@st.cache_resource(show_spinner="Loading models and vector store…")
def get_pipeline():
    return create_pipeline()


pipe = get_pipeline()

with st.sidebar:
    st.title("🧑‍💼 RecruitRAG")
    st.caption("Resume Q&A and explainable candidate matching — fully local RAG.")
    anonymize = st.toggle("🙈 Blind screening", value=False,
                          help="Masks names, emails, phones and links before anything is shown or sent to the LLM.")
    st.caption(f"LLM: `{os.getenv('RECRUIT_RAG_MODEL', 'gemma3:1b')}` via Ollama")
    st.divider()
    uploads = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Index uploaded resumes", disabled=not uploads):
        tmp = tempfile.mkdtemp()
        paths = []
        for f in uploads:
            path = os.path.join(tmp, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            paths.append(path)
        with st.spinner("Parsing and indexing…"):
            n = pipe.add_documents(paths)
        st.success(f"Indexed {n} resume(s)")
    if st.button("Index bundled sample resumes"):
        folder = "sample_data/resumes"
        pipe.add_documents([os.path.join(folder, f) for f in sorted(os.listdir(folder))])
        st.success("Sample resumes indexed")

profiles = pipe.datastore.get_profiles()
st.sidebar.metric("Candidates indexed", len(profiles))

tab_chat, tab_match, tab_profiles = st.tabs(["💬 Ask about candidates", "🎯 Match to a job", "📇 Profiles"])

# ------------------------------------------------------------------ chat
with tab_chat:
    st.caption('Try: "Who has Kubernetes and Terraform experience?" · "Summarise Priya\'s background"')
    st.session_state.setdefault("history", [])
    for role, text, hits in st.session_state["history"]:
        with st.chat_message(role):
            st.markdown(text)
            if hits:
                with st.expander("Sources"):
                    for h in hits:
                        st.markdown(f"**{h.candidate_id}** · _{h.section}_ · score {h.score:.2f}")
                        st.code(h.content, language=None)
    if question := st.chat_input("Ask a question about your candidate pool"):
        st.session_state["history"].append(("user", question, None))
        if not profiles:
            st.session_state["history"].append(("assistant", "Index some resumes first (sidebar).", None))
        else:
            with st.spinner("Searching resumes…"):
                try:
                    ans = pipe.ask(question, anonymize=anonymize)
                    st.session_state["history"].append(("assistant", ans.text, ans.hits))
                except Exception as exc:  # e.g. Ollama not running
                    st.session_state["history"].append(("assistant", f"⚠️ LLM error: {exc}", None))
        st.rerun()

# ----------------------------------------------------------------- match
with tab_match:
    default_jd = ""
    sample_jd = "sample_data/jobs/ml_engineer_llm.txt"
    if os.path.exists(sample_jd):
        with open(sample_jd, encoding="utf-8") as f:
            default_jd = f.read()
    jd = st.text_area("Job description", value=default_jd, height=240)
    top_n = st.slider("Candidates to show", 1, 20, 5)
    explain = True   # deterministic explanations are instant and always consistent

    if st.button("Rank candidates", type="primary", disabled=not (jd.strip() and profiles)):
        with st.spinner("Scoring candidates…"):
            job, matches = pipe.match_job(jd, top_n=top_n, anonymize=anonymize, explain=explain)
        st.markdown(f"**{job.title}** — required: {', '.join(job.required_skills) or '–'} · "
                    f"preferred: {', '.join(job.preferred_skills) or '–'} · min years: {job.min_years or '–'}")
        st.dataframe(
            [{"Candidate": m.name, "Score": m.overall, "Verdict": m.recommendation,
              "Skills %": round(m.skill_score * 100), "Semantic %": round(m.semantic_score * 100),
              "Experience %": round(m.experience_score * 100), "Years": m.years_experience} for m in matches],
            column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")},
            hide_index=True,
        )
        for m in matches:
            with st.expander(f"{m.name} — {m.overall}/100 ({m.recommendation})"):
                st.markdown(f"✅ **Matched:** {', '.join(m.matched_required + m.matched_preferred) or '–'}")
                st.markdown(f"❌ **Missing required:** {', '.join(m.missing_required) or '–'}")
                if m.explanation:
                    st.info(m.explanation)
                for e in m.evidence:
                    st.code(e, language=None)
        st.caption("Scores are decision support, not decisions: a human should review every shortlist.")

# -------------------------------------------------------------- profiles
with tab_profiles:
    if not profiles:
        st.info("No resumes indexed yet.")
    else:
        from src.util.privacy import Anonymizer
        anon = Anonymizer(profiles.values())
        rows = []
        for p in profiles.values():
            rows.append({
                "Name": anon.label(p.candidate_id) if anonymize else p.name,
                "Email": "[hidden]" if anonymize else p.email,
                "Years": p.years_experience,
                "Education": "; ".join(p.education),
                "Skills": ", ".join(p.skills),
            })
        st.dataframe(rows, hide_index=True)
