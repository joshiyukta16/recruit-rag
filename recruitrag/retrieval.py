"""Hybrid retriever: vector similarity + keyword overlap, name-aware, diversified per candidate."""
import re
from typing import List, Optional, Set

from .contracts import Store
from .models import SearchHit

_STOP = set("""a an and are any about all at be by can candidate candidates do does experience find for from give has have
how i in is it list me of on or resume resumes show tell that the their there to was were what when where which who
whom whose with years year work worked knows know good strong""".split())


def _terms(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9+#.]+", text.lower()) if t not in _STOP and len(t) > 1}


class HybridRetriever:
    def __init__(self, store: Store, vector_weight: float = 0.7):
        self.store = store
        self.vector_weight = vector_weight

    def _mentioned_candidates(self, query: str) -> Set[str]:
        """If the query names a candidate ("What are Priya's skills?"), restrict to them."""
        words = set(re.findall(r"[a-z]+", query.lower()))
        found = set()
        for cid, p in self.store.get_profiles().items():
            if any(len(part) >= 3 and part.lower() in words for part in p.name.split()):
                found.add(cid)
        return found

    def search(self, query: str, top_k: int = 8, max_per_candidate: int = 2,
               candidate_ids: Optional[Set[str]] = None) -> List[SearchHit]:
        named = self._mentioned_candidates(query)
        if named:
            candidate_ids = (candidate_ids & named) if candidate_ids else named
            max_per_candidate = max(max_per_candidate, top_k)  # drill into one person freely

        pool = self.store.search(query, top_k=max(top_k * 6, 40), candidate_ids=candidate_ids)
        q_terms = _terms(query)
        rescored = []
        for h in pool:
            lexical = len(q_terms & _terms(h.content)) / len(q_terms) if q_terms else 0.0
            h.score = self.vector_weight * max(h.score, 0.0) + (1 - self.vector_weight) * lexical
            rescored.append(h)
        rescored.sort(key=lambda h: h.score, reverse=True)

        counts, out = {}, []
        for h in rescored:
            if counts.get(h.candidate_id, 0) >= max_per_candidate:
                continue
            counts[h.candidate_id] = counts.get(h.candidate_id, 0) + 1
            out.append(h)
            if len(out) == top_k:
                break
        return out
