from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .impl.evaluator import retrieval_metrics
from .impl.matcher import CandidateMatcher
from .interface import (
    Answer, BaseDatastore, BaseEvaluator, BaseIndexer, BaseResponseGenerator,
    BaseRetriever, CandidateMatch, EvaluationResult, JobRequirements,
)
from .util.privacy import Anonymizer
from .util.skills import extract_skills


@dataclass
class RAGPipeline:
    """Orchestrates indexing, Q&A, job matching and evaluation."""
    datastore: BaseDatastore
    indexer: BaseIndexer
    retriever: BaseRetriever
    response_generator: BaseResponseGenerator
    evaluator: Optional[BaseEvaluator] = None
    matcher: Optional[CandidateMatcher] = None

    def __post_init__(self):
        if self.matcher is None:
            self.matcher = CandidateMatcher(self.datastore)

    def reset(self) -> None:
        self.datastore.reset()

    def add_documents(self, paths: List[str]) -> int:
        """Index resumes. Re-adding the same file replaces its old chunks (no stale data)."""
        items, profiles = self.indexer.index(paths)
        for p in profiles:
            self.datastore.delete_candidate(p.candidate_id)
        self.datastore.add_items(items)
        self.datastore.add_profiles(profiles)
        print(f"✅ Indexed {len(profiles)} resumes → {len(items)} chunks.")
        return len(profiles)

    def _skill_facts(self, query: str, anonymizer: Optional[Anonymizer]) -> str:
        """Exact skill lookup over parsed profiles. Reliable even when the LLM is small."""
        wanted = extract_skills(query)
        if not wanted:
            return ""
        profiles = list(self.datastore.get_profiles().values())
        label = (lambda p: anonymizer.label(p.candidate_id)) if anonymizer else (lambda p: p.name)
        have = lambda p: {s.lower() for s in p.skills}
        full = [p for p in profiles if all(s.lower() in have(p) for s in wanted)]
        if full:
            who = "; ".join(f"{label(p)} ({p.years_experience} yrs)" for p in full)
            return f"Candidates whose parsed skills include ALL of [{', '.join(wanted)}]: {who}."
        part = sorted(((sum(s.lower() in have(p) for s in wanted), p) for p in profiles),
                      key=lambda t: t[0], reverse=True)
        part = [(n, p) for n, p in part if n > 0][:4]
        if not part:
            return f"No candidate has any of [{', '.join(wanted)}] in their parsed skills."
        who = "; ".join(f"{label(p)} ({n}/{len(wanted)})" for n, p in part)
        return f"No candidate has ALL of [{', '.join(wanted)}]. Closest: {who}."

    def ask(self, query: str, anonymize: bool = False, top_k: int = 8) -> Answer:
        hits = self.retriever.search(query, top_k=top_k)
        anonymizer = Anonymizer(self.datastore.get_profiles().values()) if anonymize else None
        if anonymizer:
            hits = anonymizer.mask_hits(hits)
        facts = self._skill_facts(query, anonymizer)
        return Answer(text=self.response_generator.generate_response(query, hits, facts), hits=hits)

    def process_query(self, query: str, anonymize: bool = False) -> str:
        return self.ask(query, anonymize).text

    def match_job(self, job_text: str, top_n: int = 10, anonymize: bool = False,
                  explain: bool = False, llm_explain: bool = False) -> Tuple[JobRequirements, List[CandidateMatch]]:
        job, matches = self.matcher.rank(job_text, top_n)
        if explain or llm_explain:  # built from structured facts only (no raw resume text)
            for m in matches:
                m.explanation = (self.response_generator.llm_explain_match(job, m) if llm_explain
                                 else self.response_generator.explain_match(job, m))
        if anonymize:
            matches = Anonymizer(self.datastore.get_profiles().values()).mask_matches(matches)
        return job, matches

    def evaluate_retrieval(self, questions: List[Dict], k: int = 5) -> Dict:
        return retrieval_metrics(self.retriever, questions, k)

    def evaluate_answers(self, questions: List[Dict]) -> List[EvaluationResult]:
        results = []
        for item in questions:
            if "answer" not in item:
                continue
            response = self.process_query(item["question"])
            results.append(self.evaluator.evaluate(item["question"], response, item["answer"]))
        return results
