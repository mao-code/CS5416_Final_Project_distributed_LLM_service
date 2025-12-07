#!/usr/bin/env bash

echo "Sending test queries to Node0 with IP:$NODE_0_IP..."
NODE_0_IP=$NODE_0_IP python client.py

# Example Usage
# NODE_0_IP=132.236.91.188:8010 ./client.sh