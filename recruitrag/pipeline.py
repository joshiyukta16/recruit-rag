"""RecruitPipeline: ties indexing, question answering, job matching and evaluation together."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .contracts import Grader, Indexer, Responder, Searcher, Store
from .evaluation import retrieval_metrics
from .matching import CandidateMatcher
from .models import Answer, CandidateMatch, JobRequirements, JudgeVerdict
from .privacy import Anonymizer
from .skills import extract_skills


@dataclass
class RecruitPipeline:
    store: Store
    indexer: Indexer
    retriever: Searcher
    responder: Responder
    grader: Optional[Grader] = None
    matcher: Optional[CandidateMatcher] = None

    def __post_init__(self):
        if self.matcher is None:
            self.matcher = CandidateMatcher(self.store)

    def reset(self) -> None:
        self.store.reset()

    def ingest(self, paths: List[str]) -> int:
        """Index resumes. Re-ingesting a file replaces its previous chunks, so nothing goes stale."""
        chunks, profiles = self.indexer.index(paths)
        for profile in profiles:
            self.store.delete_candidate(profile.candidate_id)
        self.store.add_chunks(chunks)
        self.store.add_profiles(profiles)
        print(f"✅ Indexed {len(profiles)} resumes → {len(chunks)} chunks.")
        return len(profiles)

    def _skill_facts(self, query: str, anonymizer: Optional[Anonymizer]) -> str:
        """Exact skill lookup over parsed profiles; stays correct even when the LLM is small."""
        wanted = extract_skills(query)
        if not wanted:
            return ""
        profiles = list(self.store.get_profiles().values())
        label = (lambda p: anonymizer.label(p.candidate_id)) if anonymizer else (lambda p: p.name)
        skills_of = lambda p: {s.lower() for s in p.skills}
        full = [p for p in profiles if all(s.lower() in skills_of(p) for s in wanted)]
        if full:
            who = "; ".join(f"{label(p)} ({p.years_experience} yrs)" for p in full)
            return f"Candidates whose parsed skills include ALL of [{', '.join(wanted)}]: {who}."
        ranked = sorted(((sum(s.lower() in skills_of(p) for s in wanted), p) for p in profiles),
                        key=lambda pair: pair[0], reverse=True)
        ranked = [(n, p) for n, p in ranked if n > 0][:4]
        if not ranked:
            return f"No candidate has any of [{', '.join(wanted)}] in their parsed skills."
        who = "; ".join(f"{label(p)} ({n}/{len(wanted)})" for n, p in ranked)
        return f"No candidate has ALL of [{', '.join(wanted)}]. Closest: {who}."

    def ask(self, query: str, anonymize: bool = False, top_k: int = 8) -> Answer:
        hits = self.retriever.search(query, top_k=top_k)
        anonymizer = Anonymizer(self.store.get_profiles().values()) if anonymize else None
        if anonymizer:
            hits = anonymizer.mask_hits(hits)
        facts = self._skill_facts(query, anonymizer)
        return Answer(text=self.responder.respond(query, hits, facts), hits=hits)

    def match_job(self, job_text: str, top_n: int = 10, anonymize: bool = False,
                  explain: bool = False, llm_explain: bool = False
                  ) -> Tuple[JobRequirements, List[CandidateMatch]]:
        job, matches = self.matcher.rank(job_text, top_n)
        if explain or llm_explain:   # built from computed facts only, never from raw resume text
            for m in matches:
                m.explanation = (self.responder.llm_explain_match(job, m) if llm_explain
                                 else self.responder.explain_match(job, m))
        if anonymize:
            matches = Anonymizer(self.store.get_profiles().values()).mask_matches(matches)
        return job, matches

    def evaluate_retrieval(self, questions: List[Dict], k: int = 5) -> Dict:
        return retrieval_metrics(self.retriever, questions, k)

    def judge_answers(self, questions: List[Dict]) -> List[JudgeVerdict]:
        verdicts = []
        for item in questions:
            if "answer" in item:
                reply = self.ask(item["question"]).text
                verdicts.append(self.grader.grade(item["question"], reply, item["answer"]))
        return verdicts
