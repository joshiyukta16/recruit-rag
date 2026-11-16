from abc import ABC, abstractmethod

from .models import EvaluationResult


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, query: str, response: str, expected_answer: str) -> EvaluationResult: ...
