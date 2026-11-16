"""Embedding backends. Default = all-MiniLM-L6-v2 (same model as the original project).
`RECRUIT_RAG_EMBEDDER=hash` selects a dependency-free hashing embedder for tests/CI."""
import math
import os
import re
import zlib
from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    dim: int

    @abstractmethod
    def encode(self, text: str) -> List[float]: ...

    def encode_many(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]


class SentenceTransformerEmbedder(Embedder):
    dim = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def encode_many(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words (+bigram) hashing vectors. Not semantic - test use only."""
    dim = 384

    def encode(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-z0-9+#.]+", text.lower())
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for f in feats:
            h = zlib.crc32(f.encode())
            vec[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def get_embedder() -> Embedder:
    if os.getenv("RECRUIT_RAG_EMBEDDER", "minilm").lower() == "hash":
        return HashingEmbedder()
    return SentenceTransformerEmbedder()
