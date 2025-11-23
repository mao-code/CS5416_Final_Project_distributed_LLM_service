#!/usr/bin/env python3
"""
Client script for testing the ML inference pipeline.
Sends one request every 10 seconds for 1 minute (6 requests total).
Requests are sent at fixed intervals regardless of response time.
"""

import os
import time
import requests
import json
import csv  # Timing logic (new)
import threading
from datetime import datetime
from typing import Dict, Optional

# Read NODE_0_IP from environment variable
NODE_0_IP = os.environ.get('NODE_0_IP', 'localhost:8000')
SERVER_URL = f"http://{NODE_0_IP}/query"
METRICS_CSV_PATH = os.environ.get('METRICS_CSV_PATH', 'mao_request_timings.csv')  # Timing logic (new)

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

# Shared data structures
results = {}
results_lock = threading.Lock()
requests_sent = []
requests_lock = threading.Lock()


def send_request_async(request_id: str, query: str, send_time: float):
    """Send a single request to the server asynchronously"""
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending request {request_id}")
        print(f"Query: {query}")
        
        payload = {
            'request_id': request_id,
            'query': query
        }
        
        start_time = time.time()
        response = requests.post(SERVER_URL, json=payload, timeout=300)
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
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Request {request_id} timed out after 300s")
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


# Timing logic (new): helper to parse floats safely from CSV
def _safe_float(val: Optional[str]) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# Timing logic (new): summarize per-node latency/throughput from metrics CSV
def _print_node_metrics_summary():
    if not os.path.exists(METRICS_CSV_PATH):
        print(f"\nMetrics file not found at {METRICS_CSV_PATH}, skipping node summary.")
        return

    node_stats: Dict[int, Dict[str, float]] = {}
    try:
        with open(METRICS_CSV_PATH, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                node_raw = row.get('node_number')
                try:
                    node_id = int(node_raw)
                except (TypeError, ValueError):
                    continue
                latency = _safe_float(row.get('node_latency'))
                throughput = _safe_float(row.get('node_throughput_rps'))
                stats = node_stats.setdefault(
                    node_id,
                    {'lat_sum': 0.0, 'lat_count': 0, 'thr_sum': 0.0, 'thr_count': 0}
                )
                if latency is not None:
                    stats['lat_sum'] += latency
                    stats['lat_count'] += 1
                if throughput is not None:
                    stats['thr_sum'] += throughput
                    stats['thr_count'] += 1
    except Exception as exc:
        print(f"\nUnable to read metrics file {METRICS_CSV_PATH}: {exc}")
        return

    if not node_stats:
        print(f"\nNo node metrics available in {METRICS_CSV_PATH}.")
        return

    print("\nAverage node latency/throughput (from metrics CSV):")
    for node_id in sorted(node_stats.keys()):
        stats = node_stats[node_id]
        avg_latency = (stats['lat_sum'] / stats['lat_count']) if stats['lat_count'] else None
        avg_throughput = (stats['thr_sum'] / stats['thr_count']) if stats['thr_count'] else None
        latency_display = f"{avg_latency:.3f}s" if avg_latency is not None else "n/a"
        throughput_display = f"{avg_throughput:.3f} req/s" if avg_throughput is not None else "n/a"
        print(f" - Node {node_id}: latency={latency_display}, throughput={throughput_display}")


def main():
    """
    Main function: sends requests every 10 seconds for 1 minute
    Requests are sent at fixed intervals regardless of response time
    """
    print("="*70)
    print("ML INFERENCE PIPELINE CLIENT")
    print("="*70)
    print(f"Server URL: {SERVER_URL}")
    print(f"Sending 6 requests")
    print("="*70)
    
    # Check if server is healthy
    try:
        health_response = requests.get(f"http://{NODE_0_IP}/health", timeout=5)
        if health_response.status_code == 200:
            print(f"Server is healthy: {health_response.json()}")
        else:
            print(f"Server health check returned status {health_response.status_code}")
    except:
        print(f"Could not reach server health endpoint")
    
    start_time = time.time()
    threads = []
    
    # Send 6 requests at 10-second intervals
    for i in range(6):
        # Calculate when this request should be sent
        target_send_time = start_time + (i * 10)
        
        # Wait until the target send time
        current_time = time.time()
        if current_time < target_send_time:
            wait_time = target_send_time - current_time
            if i > 0:
                print(f"\nWaiting {wait_time:.2f}s before next request...")
            time.sleep(wait_time)
        
        # Send request in a separate thread
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
            args=(request_id, query, time.time())
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete (with a reasonable timeout)
    # print(f"\n\nWaiting for all responses (up to 5 minutes)...")
    print(f"\n\nWaiting for all responses (up to 10 minutes)...")
    for thread in threads:
        # thread.join(timeout=320)  # 5 min 20 sec to allow for some buffer
        thread.join(timeout=620)  # 10 min 20 sec to allow for some buffer
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total requests sent: 6")
    
    with results_lock:
        successful = sum(1 for r in results.values() if r.get('success', False))
        print(f"Successful responses: {successful}")
        print(f"Failed requests: {6 - successful}")
    
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
    
    _print_node_metrics_summary()  # Timing logic (new)
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
