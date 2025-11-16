from dataclasses import dataclass
import numpy

@dataclass
class EmbeddingsRequest:
    query: list[str]

@dataclass
class EmbeddingsResponse:
    embeddings: numpy.ndarray
