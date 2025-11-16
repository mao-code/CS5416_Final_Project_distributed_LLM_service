import json
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/embedding', methods=['POST'])
def embed_query():
    """Handle incoming query requests"""
    print("HEY, you made it")
    print(request.json)

    return jsonify({}), 200

