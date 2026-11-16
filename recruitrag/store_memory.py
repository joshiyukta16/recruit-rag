"""In-memory store (numpy cosine search). Used by tests and for quick demos."""
from typing import Dict, List, Optional, Set

import numpy as np

from .models import Chunk, ResumeProfile, SearchHit
from .embeddings import Embedder, get_embedder


class MemoryStore:
    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or get_embedder()
        self.reset()

    def reset(self) -> None:
        self.chunks: Dict[str, Chunk] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.profiles: Dict[str, ResumeProfile] = {}

    def embed(self, text: str) -> List[float]:
        return self.embedder.encode(text)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        for chunk, vec in zip(chunks, self.embedder.encode_many([c.content for c in chunks])):
            self.chunks[chunk.source] = chunk
            self.vectors[chunk.source] = np.asarray(vec, dtype=np.float32)

    def delete_candidate(self, candidate_id: str) -> None:
        for src in [s for s, c in self.chunks.items() if c.candidate_id == candidate_id]:
            del self.chunks[src], self.vectors[src]

    def add_profiles(self, profiles: List[ResumeProfile]) -> None:
        self.profiles.update({p.candidate_id: p for p in profiles})

    def get_profiles(self) -> Dict[str, ResumeProfile]:
        return dict(self.profiles)

    def search(self, query: str, top_k: int = 5, candidate_ids: Optional[Set[str]] = None) -> List[SearchHit]:
        q = np.asarray(self.embed(query), dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        scored = []
        for src, item in self.chunks.items():
            if candidate_ids and item.candidate_id not in candidate_ids:
                continue
            v = self.vectors[src]
            scored.append((float(q @ v / (qn * (np.linalg.norm(v) or 1.0))), item))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [SearchHit(i.content, i.source, i.candidate_id, i.section, s) for s, i in scored[:top_k]]
