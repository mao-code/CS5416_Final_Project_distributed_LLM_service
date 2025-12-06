import os
from dataclasses import dataclass
from typing import Tuple


def _parse_host_port(addr: str, default_port: int = 8000) -> Tuple[str, int]:
    """Split an address of the form host:port into components."""
    if "://" in addr:
        addr = addr.split("://", maxsplit=1)[1]
    if ":" not in addr:
        return addr, default_port
    host, port = addr.rsplit(":", maxsplit=1)
    return host, int(port)


def _get_bool(env_name: str, default: bool = False) -> bool:
    val = os.environ.get(env_name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    total_nodes: int = 3
    node_number: int = 0
    node_0_ip: str = "127.0.0.1:8000"
    node_1_ip: str = "127.0.0.1:8001"
    node_2_ip: str = "127.0.0.1:8002"
    faiss_index_path: str = "faiss_index.bin"
    documents_dir: str = "documents"

    retrieval_k: int = 10  #You must retrieve this many documents from the FAISS index
    max_tokens: int = 128 #You must use this max token limit
    truncate_length: int = 512 # You must use this truncate length
    llm_max_batch: int = 2
    num_shards: int = 4

    max_batch_size_0: int = 8
    batch_timeout_0: float = 0.02
    max_batch_size_1: int = 8
    batch_timeout_1: float = 0.05
    max_batch_size_2: int = 2
    batch_timeout_2: float = 0.05

    http_timeout: float = 120.0
    request_timeout: float = 600.0

    # Metrics/logging
    metrics_csv_path: str = "mao_request_timings.csv"
    metrics_enabled: bool = True

    prefer_gpu: bool = False
    only_cpu: bool = False

    @property
    def documents_db_path(self) -> str:
        return os.path.join(self.documents_dir, "documents.db")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            total_nodes=int(os.environ.get("TOTAL_NODES", "3")),
            node_number=int(os.environ.get("NODE_NUMBER", "0")),
            node_0_ip=os.environ.get("NODE_0_IP", "127.0.0.1:8000"),
            node_1_ip=os.environ.get("NODE_1_IP", "127.0.0.1:8001"),
            node_2_ip=os.environ.get("NODE_2_IP", "127.0.0.1:8002"),
            faiss_index_path=os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin"),
            documents_dir=os.environ.get("DOCUMENTS_DIR", "documents"),
            retrieval_k=int(os.environ.get("RETRIEVAL_K", "10")),
            max_tokens=int(os.environ.get("MAX_TOKENS", "128")),
            truncate_length=int(os.environ.get("TRUNCATE_LENGTH", "512")),
            llm_max_batch=int(os.environ.get("LLM_MAX_BATCH", "2")),
            num_shards=int(os.environ.get("NUM_SHARDS", "4")),
            max_batch_size_0=int(os.environ.get("MAX_BATCH_SIZE_0", "8")),
            batch_timeout_0=float(os.environ.get("BATCH_TIMEOUT_0", "0.02")),
            max_batch_size_1=int(os.environ.get("MAX_BATCH_SIZE_1", "8")),
            batch_timeout_1=float(os.environ.get("BATCH_TIMEOUT_1", "0.05")),
            max_batch_size_2=int(os.environ.get("MAX_BATCH_SIZE_2", "4")),
            batch_timeout_2=float(os.environ.get("BATCH_TIMEOUT_2", "0.05")),
            http_timeout=float(os.environ.get("HTTP_TIMEOUT", "120")),
            request_timeout=float(os.environ.get("REQUEST_TIMEOUT", "600")),
            metrics_csv_path=os.environ.get("METRICS_CSV_PATH", "mao_request_timings.csv"),
            metrics_enabled=_get_bool("METRICS_ENABLED", True),
            prefer_gpu=_get_bool("USE_GPU", False),
            only_cpu=_get_bool("ONLY_CPU", True),
        )
