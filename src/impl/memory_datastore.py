"""In-memory datastore (numpy cosine search). Used by tests and for quick demos."""
from typing import Dict, List, Optional, Set

import numpy as np

from ..interface.base_datastore import BaseDatastore
from ..interface.models import DataItem, ResumeProfile, SearchHit
from .embedder import Embedder, get_embedder


class MemoryDatastore(BaseDatastore):
    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or get_embedder()
        self.reset()

    def reset(self) -> None:
        self.items: Dict[str, DataItem] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.profiles: Dict[str, ResumeProfile] = {}

    def get_vector(self, content: str) -> List[float]:
        return self.embedder.encode(content)

    def add_items(self, items: List[DataItem]) -> None:
        for item, vec in zip(items, self.embedder.encode_many([i.content for i in items])):
            self.items[item.source] = item
            self.vectors[item.source] = np.asarray(vec, dtype=np.float32)

    def delete_candidate(self, candidate_id: str) -> None:
        for src in [s for s, i in self.items.items() if i.candidate_id == candidate_id]:
            del self.items[src], self.vectors[src]

    def add_profiles(self, profiles: List[ResumeProfile]) -> None:
        self.profiles.update({p.candidate_id: p for p in profiles})

    def get_profiles(self) -> Dict[str, ResumeProfile]:
        return dict(self.profiles)

    def search(self, query: str, top_k: int = 5, candidate_ids: Optional[Set[str]] = None) -> List[SearchHit]:
        q = np.asarray(self.get_vector(query), dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        scored = []
        for src, item in self.items.items():
            if candidate_ids and item.candidate_id not in candidate_ids:
                continue
            v = self.vectors[src]
            scored.append((float(q @ v / (qn * (np.linalg.norm(v) or 1.0))), item))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [SearchHit(i.content, i.source, i.candidate_id, i.section, s) for s, i in scored[:top_k]]
