import requests
import json
from flask import jsonify

embedding_request = json.dumps(["hello", "worlds!"])
print(embedding_request)
query_embeddings = requests.post(f"http://127.0.0.1:5000/embedding", json=jsonify(embedding_request), timeout=300)
