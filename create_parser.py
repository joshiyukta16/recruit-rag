import argparse


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recruit-rag", description="RecruitRAG - resume Q&A and candidate matching")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("reset", help="Wipe the vector DB and stored profiles")

    add = sub.add_parser("add", help="Index resumes (.pdf .docx .txt .md)")
    add.add_argument("-p", "--path", default="sample_data/resumes", help="File or folder of resumes")

    sub.add_parser("profiles", help="List parsed candidate profiles")

    q = sub.add_parser("query", help="Ask a question about the candidates")
    q.add_argument("prompt")
    q.add_argument("--anonymize", action="store_true", help="Blind mode: mask names and contact info")
    q.add_argument("--top-k", type=int, default=8)

    m = sub.add_parser("match", help="Rank candidates against a job description")
    m.add_argument("-j", "--job", required=True, help="Path to a job description .txt")
    m.add_argument("--top", type=int, default=5)
    m.add_argument("--explain", action="store_true", help="Deterministic fit summary per candidate (no LLM)")
    m.add_argument("--llm-explain", action="store_true", help="LLM-written summary (small models may contradict the facts)")
    m.add_argument("--anonymize", action="store_true")
    m.add_argument("--json", action="store_true", help="Machine-readable output (for ATS/API integration)")

    cal = sub.add_parser("calibrate", help="Show raw semantic similarity per candidate for a JD")
    cal.add_argument("-j", "--job", required=True)

    ev = sub.add_parser("evaluate", help="Retrieval metrics (Hit@k, Recall@k, MRR) on a question set")
    ev.add_argument("-f", "--eval_file", default="sample_data/eval/sample_questions.json")
    ev.add_argument("-k", type=int, default=3)
    ev.add_argument("--llm-judge", action="store_true", help="Also grade generated answers with the LLM")
    return p
