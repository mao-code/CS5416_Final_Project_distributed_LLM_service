from dataclasses import dataclass
import numpy

@dataclass
class EmbeddingsRequest:
    queries: list[str]

@dataclass
class EmbeddingsResponse:
    embeddings: numpy.ndarray
