"""Plain dataclasses shared by every component (no heavy dependencies)."""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class DataItem:
    """One indexed chunk of a resume."""
    content: str = ""
    source: str = ""        # unique chunk id: "<candidate_id>:<n>"
    candidate_id: str = ""
    section: str = ""


@dataclass
class SearchHit:
    """A retrieved chunk plus its relevance score (0..1, higher = better)."""
    content: str
    source: str
    candidate_id: str
    section: str = ""
    score: float = 0.0


@dataclass
class ResumeProfile:
    """Structured facts extracted from one resume."""
    candidate_id: str
    filename: str
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: List[str] = field(default_factory=list)
    years_experience: float = 0.0
    education: List[str] = field(default_factory=list)
    sections_found: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class JobRequirements:
    title: str = ""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_years: float = 0.0
    raw_text: str = ""
    requirements_text: str = ""   # title + requirement sections only (no company blurb)


@dataclass
class CandidateMatch:
    candidate_id: str
    name: str
    overall: float                      # 0..100
    skill_score: float                  # 0..1
    semantic_score: float               # 0..1
    experience_score: float             # 0..1
    years_experience: float = 0.0
    matched_required: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    matched_preferred: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""            # Strong / Moderate / Weak
    explanation: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Answer:
    text: str
    hits: List[SearchHit] = field(default_factory=list)


@dataclass
class EvaluationResult:
    question: str
    response: str
    expected_answer: str
    is_correct: bool
    reasoning: Optional[str] = None
