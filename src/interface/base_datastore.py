from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set

from .models import DataItem, ResumeProfile, SearchHit


class BaseDatastore(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def add_items(self, items: List[DataItem]) -> None: ...

    @abstractmethod
    def delete_candidate(self, candidate_id: str) -> None: ...

    @abstractmethod
    def add_profiles(self, profiles: List[ResumeProfile]) -> None: ...

    @abstractmethod
    def get_profiles(self) -> Dict[str, ResumeProfile]: ...

    @abstractmethod
    def get_vector(self, content: str) -> List[float]: ...

    @abstractmethod
    def search(
        self, query: str, top_k: int = 5, candidate_ids: Optional[Set[str]] = None
    ) -> List[SearchHit]: ...
