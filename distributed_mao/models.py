from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    query: str


class RetrievalItem(BaseModel):
    request_id: str
    query: str
    start_time: float


class RetrievalBatch(BaseModel):
    batch: List[RetrievalItem]


class GenerationItem(BaseModel):
    request_id: str
    query: str
    reranked_doc_ids: List[int]
    start_time: float
    retrieval_finished_at: Optional[float] = None
    stage_embeddings: Optional[float] = None
    stage_faiss_search: Optional[float] = None
    stage_fetch_documents: Optional[float] = None
    stage_rerank: Optional[float] = None


class GenerationBatch(BaseModel):
    batch: List[GenerationItem]


class PipelineResult(BaseModel):
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str
    start_time: Optional[float] = None
    processing_time: Optional[float] = None
    retrieval_finished_at: Optional[float] = None
    generation_finished_at: Optional[float] = None
    retrieval_duration: Optional[float] = None
    generation_duration: Optional[float] = None
    stage_embeddings: Optional[float] = None
    stage_faiss_search: Optional[float] = None
    stage_fetch_documents: Optional[float] = None
    stage_rerank: Optional[float] = None
    stage_generate: Optional[float] = None
    stage_sentiment: Optional[float] = None
    stage_safety_filter: Optional[float] = None


class ResultBatch(BaseModel):
    results: List[PipelineResult]
