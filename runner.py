import requests
import json

embedding_request = json.dumps(["hello", "worlds!"])
query_embeddings = requests.post(f"http://127.0.0.1:5000/embedding", json=embedding_request, timeout=300)
