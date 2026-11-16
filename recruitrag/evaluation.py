"""Quality measurement: free retrieval metrics, and an LLM that grades generated answers."""
from typing import Dict, List

from .contracts import Searcher
from .llm import ask_llm
from .models import JudgeVerdict
from .xml_tags import tag_text

GRADER_INSTRUCTIONS = """You are grading an answer written by a resume-search assistant.
You will be given a <question>, the assistant's <answer>, and a <reference> that is known to be correct.

Decide whether the answer reaches the same conclusion as the reference for what the question asks.
Different wording or extra detail is acceptable. Wrong, contradictory or missing key facts is not.

Reply in exactly this format and nothing else:
<why>one or two sentences</why>
<verdict>correct</verdict>
Use <verdict>incorrect</verdict> when the answer fails."""


class AnswerJudge:
    """LLM-based grader. Small local models are noisy graders, so treat scores as a rough signal."""

    def grade(self, question: str, response: str, reference: str) -> JudgeVerdict:
        prompt = (f"<question>\n{question}\n</question>\n"
                  f"<answer>\n{response}\n</answer>\n"
                  f"<reference>\n{reference}\n</reference>")
        raw = ask_llm(GRADER_INSTRUCTIONS, prompt, temperature=0.0)
        verdict = tag_text(raw, "verdict")
        why = tag_text(raw, "why")
        if verdict is None:
            return JudgeVerdict(question, response, reference, False, f"Unparseable grader reply: {raw!r}")
        return JudgeVerdict(question, response, reference, verdict.strip().lower() == "correct", why)


def retrieval_metrics(retriever: Searcher, questions: List[Dict], k: int = 5) -> Dict:
    """Hit@k, Recall@k and MRR over questions that list `expected_candidates` (candidate ids)."""
    rows = []
    for item in questions:
        expected = set(item.get("expected_candidates", []))
        if not expected:
            continue
        hits = retriever.search(item["question"], top_k=max(k * 3, 15), max_per_candidate=1)
        ranked = [h.candidate_id for h in hits]          # one hit per candidate, best first
        top = ranked[:k]
        first = next((i + 1 for i, c in enumerate(ranked) if c in expected), None)
        rows.append({
            "question": item["question"],
            "expected": sorted(expected),
            "retrieved": top,
            "hit": bool(expected & set(top)),
            "recall": len(expected & set(top)) / len(expected),
            "rr": 1 / first if first else 0.0,
        })
    n = len(rows) or 1
    return {
        "k": k, "n_questions": len(rows),
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "recall": sum(r["recall"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "rows": rows,
    }
