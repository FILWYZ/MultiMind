 
from __future__ import annotations

import hashlib
import math
import re
from typing import List

import numpy as np

__all__ = ["LocalHashEmbeddingFunction"]


class LocalHashEmbeddingFunction:
                           

    dimension = 128

    def __call__(self, input: List[str]) -> List[np.ndarray]:
        return [self._embed(str(text or "")) for text in input]

    @classmethod
    def _embed(cls, text: str) -> np.ndarray:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        vector = [0.0] * cls.dimension
        if not normalized:
            return np.zeros(cls.dimension, dtype=np.float32)

        tokens = [normalized]
        tokens.extend(normalized[i : i + 2] for i in range(max(0, len(normalized) - 1)))
        tokens.extend(normalized[i : i + 3] for i in range(max(0, len(normalized) - 2)))
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:4], "big") % cls.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + len(token) / 10.0)

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return np.asarray([round(value / norm, 8) for value in vector], dtype=np.float32)
