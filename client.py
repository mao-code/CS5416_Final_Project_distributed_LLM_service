#!/usr/bin/env python3
"""
Client script for testing the ML inference pipeline.
Runs multiple experiments with varying request counts and intervals.
"""

import logging
import os
import time
import requests
import threading
from datetime import datetime
from typing import Dict

# Read NODE_0_IP from environment variable
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
SERVER_URL = f"http://{NODE_0_IP}/query"

# Test queries
TEST_QUERIES = [
    "How do I return a defective product?",
    "What is your refund policy?",
    "My order hasn't arrived yet, tracking number is ABC123",
    "How do I update my billing information?",
    "Is there a warranty on electronic items?",
    "Can I change my shipping address after placing an order?",
    "What payment methods do you accept?",
    "How long does shipping typically take?"
]

# Experiment settings
REQUEST_COUNTS = [10, 20, 50]
REQUEST_INTERVALS = [10, 5, 1]

# Create a custom logger for memory usage
logger = logging.getLogger('memory_logger')
logger.setLevel(logging.INFO)
# Writes to memory_use.log
fh = logging.FileHandler('memory_use.log')
fh.setLevel(logging.INFO)
# Message formatting
formatter = logging.Formatter('%(asctime)s — %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def send_request_async(
    request_id: str,
    query: str,
    send_time: float,
    results: Dict[str, dict],
    results_lock: threading.Lock,
):
    """Send a single request to the server asynchronously"""
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending request {request_id}")
        print(f"Query: {query}")
        
        payload = {
            'request_id': request_id,
            'query': query
        }
        
        start_time = time.time()
        response = requests.post(SERVER_URL, json=payload, timeout=600)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Response received for {request_id} in {elapsed_time:.2f}s")
            print(f"  Generated Response: {result.get('generated_response', '')[:100]}...")
            print(f"  Sentiment: {result.get('sentiment')}")
            print(f"  Is Toxic: {result.get('is_toxic')}")
            
            with results_lock:
                results[request_id] = {
                    'result': result,
                    'elapsed_time': elapsed_time,
                    'send_time': send_time,
                    'success': True
                }
        else:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error for {request_id}: HTTP {response.status_code}")
            print(f"  Response: {response.text}")
            
            with results_lock:
                results[request_id] = {
                    'error': f"HTTP {response.status_code}",
                    'elapsed_time': elapsed_time,
                    'send_time': send_time,
                    'success': False
                }
            
    except requests.exceptions.Timeout:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} timed out after 600s")
        with results_lock:
            results[request_id] = {
                'error': 'Timeout',
                'send_time': send_time,
                'success': False
            }
    except requests.exceptions.ConnectionError:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Failed to connect to server for {request_id}")
        with results_lock:
            results[request_id] = {
                'error': 'Connection error',
                'send_time': send_time,
                'success': False
            }
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Error for {request_id}: {str(e)}")
        with results_lock:
            results[request_id] = {
                'error': str(e),
                'send_time': send_time,
                'success': False
            }


def run_experiment(total: int, interval: float, experiment_idx: int) -> Dict[str, float]:
    """Run a single experiment with a given request count and interval."""
    results: Dict[str, dict] = {}
    requests_sent = []
    results_lock = threading.Lock()
    requests_lock = threading.Lock()

    print(f"\nStarting experiment {experiment_idx}: {total} requests, interval {interval}s")
    
    start_time = time.time()
    threads = []
    
    # Send "total" requests at "interval"-second intervals
    for i in range(total):
        target_send_time = start_time + (i * interval)
        
        current_time = time.time()
        if current_time < target_send_time:
            wait_time = target_send_time - current_time
            if i > 0:
                print(f"\nWaiting {wait_time:.2f}s before next request...")
            time.sleep(wait_time)
        
        request_id = f"req_{int(time.time())}_{i}"
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        
        with requests_lock:
            requests_sent.append({
                'request_id': request_id,
                'query': query,
                'send_time': time.time()
            })
        
        thread = threading.Thread(
            target=send_request_async,
            args=(request_id, query, time.time(), results, results_lock)
        )
        thread.start()
        threads.append(thread)
    
    print(f"\n\nWaiting for all responses (up to 5 minutes)...")
    for thread in threads:
        thread.join(timeout=320)  # 5 min 20 sec to allow for some buffer
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"SUMMARY - Experiment {experiment_idx}")
    print("="*70)
    print(f"Total requests sent: {total}")
    
    with results_lock:
        successful = sum(1 for r in results.values() if r.get('success', False))
        print(f"Successful responses: {successful}")
        print(f"Failed requests: {total - successful}")
    
    print(f"Total elapsed time: {total_time:.2f}s")
    
    with results_lock:
        if results:
            print("\nResults:")
            with requests_lock:
                for i, req_info in enumerate(requests_sent, 1):
                    req_id = req_info['request_id']
                    if req_id in results:
                        res_info = results[req_id]
                        print(f"\n{i}. Request ID: {req_id}")
                        print(f"   Query: {req_info['query'][:60]}...")
                        
                        if res_info.get('success'):
                            result = res_info['result']
                            print(f"   Success (took {res_info['elapsed_time']:.2f}s)")
                            print(f"   Sentiment: {result.get('sentiment')}")
                            print(f"   Is Toxic: {result.get('is_toxic')}")
                            print(f"   Response: {result.get('generated_response', '')[:80]}...")
                        else:
                            print(f"   Failed: {res_info.get('error', 'Unknown error')}")
                    else:
                        print(f"\n{i}. Request ID: {req_id}")
                        print(f"   ⏳ Still pending or not received")
    
    print("\n" + "="*70)

    return {
        'total': total,
        'interval': interval,
        'successful': successful,
        'failed': total - successful,
        'elapsed_time': total_time,
    }


def main():
    """
    Main function: runs every combination of request counts and intervals.
    Requests are sent at fixed intervals regardless of response time.
    """
    print("="*70)
    print("ML INFERENCE PIPELINE CLIENT")
    print("="*70)
    print(f"Server URL: {SERVER_URL}")
    print(f"Request counts: {REQUEST_COUNTS}")
    print(f"Intervals (s): {REQUEST_INTERVALS}")
    print("="*70)
    
    # Check if server is healthy once before experiments
    try:
        health_response = requests.get(f"http://{NODE_0_IP}/health", timeout=5)
        if health_response.status_code == 200:
            print(f"Server is healthy: {health_response.json()}")
        else:
            print(f"Server health check returned status {health_response.status_code}")
    except Exception:
        print("Could not reach server health endpoint")
    
    experiment_counter = 1
    all_summaries = []

    for total in REQUEST_COUNTS:
        for interval in REQUEST_INTERVALS:
            logger.info(f"Beginning experiment with n = {total} requests, sent at an interval of t = {interval} seconds")
            summary = run_experiment(total, interval, experiment_counter)
            all_summaries.append((experiment_counter, summary))
            experiment_counter += 1

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    for idx, summary in all_summaries:
        print(
            f"Experiment {idx}: {summary['total']} requests at {summary['interval']}s "
            f"=> success {summary['successful']}/{summary['total']} "
            f"(elapsed {summary['elapsed_time']:.2f}s)"
        )


if __name__ == "__main__":
    main()
