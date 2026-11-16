"""Two evaluators: (1) retrieval metrics - free, deterministic; (2) LLM-as-judge for answers."""
from typing import Dict, List

from ..interface.base_evaluator import BaseEvaluator
from ..interface.base_retriever import BaseRetriever
from ..interface.models import EvaluationResult
from ..util.extract_xml import extract_xml_tag
from ..util.invoke_ai import invoke_ai

SYSTEM_PROMPT = """
You are a system that evaluates the correctness of a response to a question.
The question will be provided in <question>...</question> tags.
The response will be provided in <response>...</response> tags.
The expected answer will be provided in <expected_answer>...</expected_answer> tags.

The response doesn't have to exactly match all the words/context of the expected answer. It just needs to be right
about the answer to the actual question itself.

Evaluate whether the response is correct or not, and return your reasoning in <reasoning>...</reasoning> tags.
Then return the result in <result>...</result> tags - either 'true' or 'false'.
"""


class Evaluator(BaseEvaluator):
    """LLM-as-judge (kept from the original pipeline)."""

    def evaluate(self, query: str, response: str, expected_answer: str) -> EvaluationResult:
        user_prompt = (f"<question>\n{query}\n</question>\n<response>\n{response}\n</response>\n"
                       f"<expected_answer>\n{expected_answer}\n</expected_answer>")
        content = invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_prompt)
        reasoning, result = extract_xml_tag(content, "reasoning"), extract_xml_tag(content, "result")
        if result is None:
            return EvaluationResult(query, response, expected_answer, False, f"No result found: ({content})")
        return EvaluationResult(query, response, expected_answer, result.lower() == "true", reasoning)


def retrieval_metrics(retriever: BaseRetriever, questions: List[Dict], k: int = 5) -> Dict:
    """Hit@k, Recall@k and MRR over questions that list `expected_candidates` (candidate ids)."""
    rows = []
    for item in questions:
        expected = set(item.get("expected_candidates", []))
        if not expected:
            continue
        hits = retriever.search(item["question"], top_k=max(k * 3, 15), max_per_candidate=1)
        ranked = [h.candidate_id for h in hits]          # unique candidates, best first
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
