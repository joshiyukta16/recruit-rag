"""Builds a ready-to-use pipeline. RECRUIT_RAG_STORE=memory selects the in-memory store (tests, demos)."""
import os
from typing import Optional

from .answering import LLMResponder
from .evaluation import AnswerJudge
from .indexing import ResumeIndexer
from .matching import CandidateMatcher
from .pipeline import RecruitPipeline
from .retrieval import HybridRetriever
from .store_lance import LanceStore
from .store_memory import MemoryStore


def create_pipeline(store: Optional[str] = None) -> RecruitPipeline:
    kind = store or os.getenv("RECRUIT_RAG_STORE", "lancedb")
    backend = MemoryStore() if kind == "memory" else LanceStore()
    return RecruitPipeline(
        store=backend,
        indexer=ResumeIndexer(),
        retriever=HybridRetriever(backend),
        responder=LLMResponder(),
        grader=AnswerJudge(),
        matcher=CandidateMatcher(backend),
    )
