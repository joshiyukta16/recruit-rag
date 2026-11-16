"""Job-description -> candidate ranking with explainable sub-scores."""
from typing import Dict, List, Optional

from .contracts import Store
from .models import CandidateMatch, JobRequirements
from .job_parser import parse_job

DEFAULT_WEIGHTS = {"skills": 0.50, "semantic": 0.30, "experience": 0.20}
# Raw MiniLM cosine between a JD's requirements and a resume's best chunks was observed in
# roughly 0.31-0.57 on the sample set. Recalibrate with `python main.py calibrate -j <jd>`
# whenever the embedding model or the resume pool changes.
SEM_LOW, SEM_HIGH = 0.30, 0.60


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


class CandidateMatcher:
    def __init__(self, store: Store, weights: Optional[Dict[str, float]] = None,
                 sem_low: float = SEM_LOW, sem_high: float = SEM_HIGH):
        self.store = store
        self.weights = weights or DEFAULT_WEIGHTS
        self.sem_low, self.sem_high = sem_low, sem_high

    def raw_semantic(self, job: JobRequirements) -> Dict[str, tuple]:
        """candidate_id -> (mean cosine of top-3 chunks, those chunks). Uses requirements only,
        so company blurb text in the JD cannot attract unrelated resumes."""
        query = job.requirements_text or job.raw_text
        hits = self.store.search(query, top_k=max(60, len(self.store.get_profiles()) * 5))
        by_cand: Dict[str, list] = {}
        for h in hits:
            by_cand.setdefault(h.candidate_id, []).append(h)
        out = {}
        for cid, hs in by_cand.items():
            top = sorted(hs, key=lambda h: h.score, reverse=True)[:3]
            out[cid] = (sum(h.score for h in top) / len(top), top)
        return out

    def raw_scores(self, job_text: str) -> List[tuple]:
        """For calibration: [(name, raw_similarity)] best first."""
        profiles = self.store.get_profiles()
        raw = self.raw_semantic(parse_job(job_text))
        rows = [(profiles[c].name if c in profiles else c, v[0]) for c, v in raw.items()]
        return sorted(rows, key=lambda r: r[1], reverse=True)

    def rank(self, job_text: str, top_n: int = 10) -> tuple:
        job: JobRequirements = parse_job(job_text)
        profiles = self.store.get_profiles()
        raw = self.raw_semantic(job)
        w = dict(self.weights)
        if not job.required_skills and not job.preferred_skills:
            w["skills"] = 0.0  # nothing to match on -> redistribute
        total_w = sum(w.values()) or 1.0

        matches: List[CandidateMatch] = []
        for cid, p in profiles.items():
            have = {s.lower() for s in p.skills}
            m_req = [s for s in job.required_skills if s.lower() in have]
            x_req = [s for s in job.required_skills if s.lower() not in have]
            m_pref = [s for s in job.preferred_skills if s.lower() in have]
            denom = len(job.required_skills) + 0.5 * len(job.preferred_skills)
            skill = (len(m_req) + 0.5 * len(m_pref)) / denom if denom else 0.0
            exp = _clip(p.years_experience / job.min_years) if job.min_years else 1.0
            mean, top_hits = raw.get(cid, (0.0, []))
            sem = _clip((mean - self.sem_low) / (self.sem_high - self.sem_low)) if top_hits else 0.0
            overall = 100 * (w["skills"] * skill + w["semantic"] * sem + w["experience"] * exp) / total_w
            matches.append(CandidateMatch(
                candidate_id=cid, name=p.name, overall=round(overall, 1),
                skill_score=round(skill, 3), semantic_score=round(sem, 3), experience_score=round(exp, 3),
                years_experience=p.years_experience, matched_required=m_req, missing_required=x_req,
                matched_preferred=m_pref, evidence=[h.content for h in top_hits[:2]],
                recommendation="Strong" if overall >= 75 else "Moderate" if overall >= 50 else "Weak",
            ))
        matches.sort(key=lambda m: m.overall, reverse=True)
        return job, matches[:top_n]
