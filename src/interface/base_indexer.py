from abc import ABC, abstractmethod
from typing import List, Tuple

from .models import DataItem, ResumeProfile


class BaseIndexer(ABC):
    @abstractmethod
    def index(self, document_paths: List[str]) -> Tuple[List[DataItem], List[ResumeProfile]]:
        """Parse resumes into searchable chunks AND structured profiles."""
