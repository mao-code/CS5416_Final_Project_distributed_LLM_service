import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
import requests
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
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

# Read environment variables
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP = os.environ.get('NODE_1_IP', 'localhost:8000')
NODE_2_IP = os.environ.get('NODE_2_IP', 'localhost:8000')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')

# Configuration
CONFIG = {
    'faiss_index_path': FAISS_INDEX_PATH,
    'documents_path': DOCUMENTS_DIR,
    'faiss_dim': 768, #You must use this dimension
    'max_tokens': 128, #You must use this max token limit
    'retrieval_k': 10, #You must retrieve this many documents from the FAISS index
    'truncate_length': 512 # You must use this truncate length
}

# Flask app
app = Flask(__name__)

# Request queue and results storage
request_queue = Queue()
results = {}
results_lock = threading.Lock()

@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float

@dataclass
class PipelineResponse:
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str
    processing_time: float

class Node0Work:

    """
    Node 0's work
    """
    
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"Initializing pipeline on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"Documents path: {CONFIG['documents_path']}")
        
        # Model names
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.safety_model_name = 'unitary/toxic-bert'
        self.model = SentenceTransformer(self.embedding_model_name).to(self.device)
    
    def process_request(self, request: PipelineRequest) -> PipelineResponse:
        """
        Backwards-compatible single-request entry point that delegates
        to the batch processor with a batch size of 1.
        """
        responses = self.process_batch([request])
        return responses[0]

    def process_batch(self, incoming_requests: List[PipelineRequest]):
        """
        Pipeline starts and ends here
        """
        if not incoming_requests:
            return []

        batch_size = len(incoming_requests)
        start_times = [time.time() for _ in incoming_requests]
        queries = [req.query for req in incoming_requests]

        print("\n" + "="*60)
        print(f"Processing batch of {batch_size} requests")
        print("="*60)
        for request in incoming_requests:
            print(f"- {request.request_id}: {request.query[:50]}...")
        
        # Step 1: Generate embeddings
        print("\n[Step 1/7] Generating embeddings for batch...")
        query_embeddings = self._generate_embeddings_batch(queries)

        payload = {
            'query_embeddings': query_embeddings.tolist(),
            'queries': queries,
        }
        reranked_docs_batch = None
        try:
            # send query_embeddings and queries to node 1
            response = requests.post(f"http://{NODE_1_IP}/context-retrieval", json=payload, timeout=300)
            reranked_docs_batch = response.json()
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error for sending query_embeddings to node 1: {str(e)}")

        json_requests = [asdict(request) for request in incoming_requests]
        payload = {
            'requests': json_requests,
            'queries': queries,
            'reranked_docs_batch': reranked_docs_batch,
            'start_times': start_times
        }
        
        try:
            request_response = requests.post(f"http://{NODE_2_IP}/context-retrieval", json=payload, timeout=300)
            responses = [PipelineResponse(**response) for response in request_response.json()]
            return responses
        except Exception as e:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error for sending  to node 2: {str(e)}")
            return []
        
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Step 2: Generate embeddings for a batch of queries"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings
    
class Node1Work:

    """
    Node 1's Work
    """
    
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"Initializing pipeline on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"Documents path: {CONFIG['documents_path']}")
        
        # Model names
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.safety_model_name = 'unitary/toxic-bert'
        self.index = faiss.read_index(CONFIG['faiss_index_path'])
        self.conn = sqlite3.connect(db_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name).to(self.device)
    

    def process_batch(self, query_embeddings: np.ndarray, queries: List[str]) -> List[List[Dict]]:
        """
        Node 1's pipeline execution for a batch of requests. 
        """
        # Step 2: FAISS ANN search
        print("\n[Step 2/7] Performing FAISS ANN search for batch...")
        doc_id_batches = self._faiss_search_batch(query_embeddings)

        # Step 3: Fetch documents from disk
        print("\n[Step 3/7] Fetching documents for batch...")
        documents_batch = self._fetch_documents_batch(doc_id_batches)

        # Step 4: Rerank documents
        print("\n[Step 4/7] Reranking documents for batch...")
        reranked_docs_batch = self._rerank_documents_batch(
            queries,
            documents_batch
        )

        return reranked_docs_batch
    
    def _faiss_search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        """Step 3: Perform FAISS ANN search for a batch of embeddings"""
        if not os.path.exists(CONFIG['faiss_index_path']):
            raise FileNotFoundError("FAISS index not found. Please create the index before running the pipeline.")
        
        print("Loading FAISS index")
        query_embeddings = query_embeddings.astype('float32')
        _, indices = self.index.search(query_embeddings, CONFIG['retrieval_k'])
        return [row.tolist() for row in indices]
    
    def _fetch_documents_batch(self, doc_id_batches: List[List[int]]) -> List[List[Dict]]:
        """Step 4: Fetch documents for each query in the batch using SQLite"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        cursor = self.conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    documents.append({
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    })
            documents_batch.append(documents)
        cursor.close()
        return documents_batch
    
    def _rerank_documents_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Step 5: Rerank retrieved documents for each query in the batch"""
        self.model.eval()
        reranked_batches = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc['content']] for doc in documents]
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=CONFIG['truncate_length']
                ).to(self.device)
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        return reranked_batches


class Node2Work:

    """
    Node 2's work
    """
    
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"Initializing pipeline on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"Documents path: {CONFIG['documents_path']}")
        
        # Model names
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.safety_model_name = 'unitary/toxic-bert'

        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            dtype=torch.float16,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.classifier = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=self.device
        )
        self.classifier2 = hf_pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=self.device
        )

    def process_batch(self, requests: List[PipelineRequest], queries: List[str], reranked_docs_batch: List[List[Dict]], start_times: list[float]) -> List[PipelineResponse]:
        """
        Node 2's pipeline execution for a batch of requests. 
        """
        # Step 5: Generate LLM responses
        print("\n[Step 5/7] Generating LLM responses for batch...")
        responses_text = self._generate_responses_batch(
            queries,
            reranked_docs_batch
        )

        # Step 6: Sentiment analysis
        print("\n[Step 6/7] Analyzing sentiment for batch...")
        sentiments = self._analyze_sentiment_batch(responses_text)

        # Step 7: Safety filter on responses
        print("\n[Step 7/7] Applying safety filter to batch...")
        toxicity_flags = self._filter_response_safety_batch(responses_text)
        
        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(f"\n✓ Request {request.request_id} processed in {processing_time:.2f} seconds")
            sensitivity_result = "true" if toxicity_flags[idx] else "false"
            responses.append(PipelineResponse(
                request_id=request.request_id,
                generated_response=responses_text[idx],
                sentiment=sentiments[idx],
                is_toxic=sensitivity_result,
                processing_time=processing_time
            ))
        
        return responses
    
    def _generate_responses_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[str]:
        """Step 6: Generate LLM responses for each query in the batch"""
        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]])
            messages = [
                {"role": "system",
                 "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=CONFIG['max_tokens'],
                temperature=0.01,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        return responses
    
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Step 7: Analyze sentiment for each generated response"""
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.classifier(truncated_texts)
        sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result['label'], 'neutral'))
        return sentiments
    
    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 8: Filter responses for safety for each entry in the batch"""
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.classifier2(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result['score'] > 0.5)
        return toxicity_flags
    

# Global pipeline instance
pipeline = None

def process_requests_worker():
    """Worker thread that processes requests from the queue"""
    global pipeline
    while True:
        try:
            request_data = request_queue.get()
            if request_data is None:  # Shutdown signal
                break
            
            # Create request object
            req = PipelineRequest(
                request_id=request_data['request_id'],
                query=request_data['query'],
                timestamp=time.time()
            )
            
            # Process request
            response = pipeline.process_request(req)
            
            # Store result
            with results_lock:
                results[request_data['request_id']] = {
                    'request_id': response.request_id,
                    'generated_response': response.generated_response,
                    'sentiment': response.sentiment,
                    'is_toxic': response.is_toxic
                }
            
            request_queue.task_done()
        except Exception as e:
            print(f"Error processing request: {e}")
            request_queue.task_done()


# Node 1 begins it's stages here because node 0 was like hi plz do this
@app.route('/context-retrieval', methods=['POST'])
def handle_handover():
    """Handle node 0 -> node 1 handover"""
    try:
        if NODE_NUMBER == 1:
            data = request.json
            query_embeddings = data.get('query_embeddings')
            queries = data.get('queries')
            
            if not query_embeddings or not queries:
                return jsonify({'error': 'Missing information'}), 400
            
            query_embeddings = np.array(query_embeddings, dtype=np.float32)
            
            reranked_docs_batch = pipeline.process_batch(query_embeddings, queries) # we will return this result to node 0
            return jsonify(reranked_docs_batch), 200
        elif NODE_NUMBER == 2:
            data = request.json
            requests_json = data.get('requests')
            queries = data.get('queries')
            reranked_docs_batch = data.get('reranked_docs_batch')
            start_times = data.get('start_times')
                        
            if not requests_json or not queries or not reranked_docs_batch or not start_times:
                return jsonify({'error': 'Missing information'}), 400
            
            request_list = [PipelineRequest(**request) for request in requests_json]
            responses = pipeline.process_batch(request_list, queries, reranked_docs_batch, start_times)
            responses_json = [asdict(response) for response in responses] # response will be returned to node 0
            return jsonify(responses_json), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@app.route('/query', methods=['POST'])
def handle_query():
    """Handle incoming query requests"""
    try:
        data = request.json
        request_id = data.get('request_id')
        query = data.get('query')
        
        if not request_id or not query:
            return jsonify({'error': 'Missing request_id or query'}), 400
        
        # Check if result already exists (request already processed)
        with results_lock:
            if request_id in results:
                return jsonify(results[request_id]), 200
        
        print(f"queueing request {request_id}")
        # Add to queue
        request_queue.put({
            'request_id': request_id,
            'query': query
        })

        # Wait for processing (with timeout). Very inefficient - would suggest using a more efficient waiting and timeout mechanism.
        timeout = 600  # 10 minutes
        start_wait = time.time()
        while True:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
                    return jsonify(result), 200
            
            if time.time() - start_wait > timeout:
                return jsonify({'error': 'Request timeout'}), 504
            
            time.sleep(0.1)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200


def main():
    """
    Main execution function
    """
    global pipeline
    
    print("="*60)
    print("MONOLITHIC CUSTOMER SUPPORT PIPELINE")
    print("="*60)
    print(f"\nRunning on Node {NODE_NUMBER} of {TOTAL_NODES} nodes")
    print(f"Node IPs: 0={NODE_0_IP}, 1={NODE_1_IP}, 2={NODE_2_IP}")
    print("\nNOTE: This implementation is deliberately inefficient.")
    print("Your task is to optimize this for a 3-node cluster.\n")

    worker_thread = None
    
    # Initialize pipeline
    print("Initializing pipeline...")
    if NODE_NUMBER == 0:
        pipeline = Node0Work()

        # Start worker thread
        worker_thread = threading.Thread(target=process_requests_worker, daemon=True)
        worker_thread.start()
        print("Worker thread started!")

    elif NODE_NUMBER == 1:
        pipeline = Node1Work()
    else:
        pipeline = Node2Work()
    print("Pipeline initialized!")
    
    
    # Start Flask server
    print(f"\nStarting Flask server")
    node_ips = [NODE_0_IP, NODE_1_IP, NODE_2_IP]
    curr_node_ip = node_ips[NODE_NUMBER]
    hostname = curr_node_ip.split(':')[0]
    port = int(curr_node_ip.split(':')[1]) if ':' in curr_node_ip else 8000
    app.run(host=hostname, port=port, threaded=True)


if __name__ == "__main__":
    main()