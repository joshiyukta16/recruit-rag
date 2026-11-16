from abc import ABC, abstractmethod
from typing import List, Optional, Set

from .models import SearchHit


class BaseRetriever(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 8,
        max_per_candidate: int = 2,
        candidate_ids: Optional[Set[str]] = None,
    ) -> List[SearchHit]: ...
