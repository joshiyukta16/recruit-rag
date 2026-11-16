"""Builds a pipeline. RECRUIT_RAG_STORE=memory switches to the in-memory store (tests/demos)."""
import os

from .impl import CandidateMatcher, Datastore, Evaluator, Indexer, MemoryDatastore, ResponseGenerator, Retriever
from .rag_pipeline import RAGPipeline


def create_pipeline(store: str = None) -> RAGPipeline:
    store = store or os.getenv("RECRUIT_RAG_STORE", "lancedb")
    datastore = MemoryDatastore() if store == "memory" else Datastore()
    return RAGPipeline(
        datastore=datastore,
        indexer=Indexer(),
        retriever=Retriever(datastore),
        response_generator=ResponseGenerator(),
        evaluator=Evaluator(),
        matcher=CandidateMatcher(datastore),
    )
