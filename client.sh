#!/usr/bin/env bash

echo "Sending test queries to Node0 with IP:$NODE_0_IP..."
NODE_0_IP=$NODE_0_IP python -m distributed.client