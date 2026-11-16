"""Structural contracts for the swappable parts of the system.

These are typing.Protocol classes: an implementation does not inherit from anything, it only has to
provide matching methods. That is what lets the tests drop in an in-memory store and a fake LLM."""
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple, runtime_checkable

from .models import CandidateMatch, Chunk, JobRequirements, JudgeVerdict, ResumeProfile, SearchHit


@runtime_checkable
class Store(Protocol):
    """Vector index over resume chunks plus the structured profile of each candidate."""
    def reset(self) -> None: ...
    def add_chunks(self, chunks: Sequence[Chunk]) -> None: ...
    def delete_candidate(self, candidate_id: str) -> None: ...
    def add_profiles(self, profiles: Sequence[ResumeProfile]) -> None: ...
    def get_profiles(self) -> Dict[str, ResumeProfile]: ...
    def embed(self, text: str) -> List[float]: ...
    def search(self, query: str, top_k: int = 5,
               candidate_ids: Optional[Set[str]] = None) -> List[SearchHit]: ...


@runtime_checkable
class Indexer(Protocol):
    """Turns resume files into chunks and profiles."""
    def index(self, paths: Sequence[str]) -> Tuple[List[Chunk], List[ResumeProfile]]: ...


@runtime_checkable
class Searcher(Protocol):
    """Finds the most relevant chunks for a question."""
    def search(self, query: str, top_k: int = 8, max_per_candidate: int = 2,
               candidate_ids: Optional[Set[str]] = None) -> List[SearchHit]: ...


@runtime_checkable
class Responder(Protocol):
    """Writes answers and match explanations."""
    def respond(self, query: str, hits: Sequence[SearchHit], facts: str = "") -> str: ...
    def explain_match(self, job: JobRequirements, match: CandidateMatch) -> str: ...
    def llm_explain_match(self, job: JobRequirements, match: CandidateMatch) -> str: ...


@runtime_checkable
class Grader(Protocol):
    """Marks a generated answer against a known-correct reference."""
    def grade(self, question: str, response: str, reference: str) -> JudgeVerdict: ...
