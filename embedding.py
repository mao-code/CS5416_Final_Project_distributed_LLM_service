import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers import pipeline as hf_pipeline
import warnings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Queue
import threading
from models.embedding_model import (
    EmbeddingsRequest,
    EmbeddingsResponse
)

# Read environment variables
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8000')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8000')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')

class EmbeddingNode:
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"Initializing pipeline on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
    
    def process_request(self, request: list[str]) -> EmbeddingsResponse:
        """
        Backwards-compatible single-request entry point that delegates
        to the batch processor with a batch size of 1.
        """
        responses = self.process_batch([request])
        return responses[0]

    def process_batch(self, requests: EmbeddingsRequest) -> EmbeddingsResponse:
        """
        Main pipeline execution for a batch of requests.
        """
        if not requests:
            return []

        batch_size = len(requests)
        queries = [req.query for req in requests]
        
        # Step 1: Generate embeddings
        print("\n[Step 1/7] Generating embeddings for batch...")
        start = time.time()
        query_embeddings = self._generate_embeddings_batch(queries)
        print("Embeddings time:", time.time() - start)
        
        responses = []
        for idx, request in enumerate(requests):
            responses.append(EmbeddingsResponse(
                request_id=request.request_id,
                embedding=query_embeddings
            ))
        
        return responses
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Step 1: Generate embeddings for a batch of queries"""
        model = SentenceTransformer(self.embedding_model_name).to(self.device)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        del model
        gc.collect()
        return embeddings
