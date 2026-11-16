from abc import ABC, abstractmethod
from typing import List

from .models import SearchHit


class BaseResponseGenerator(ABC):
    @abstractmethod
    def generate_response(self, query: str, hits: List[SearchHit], facts: str = "") -> str: ...
