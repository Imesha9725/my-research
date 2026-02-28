#!/bin/bash
# Run both backends: Emotion API (5002) + Node (5001)
# Usage: ./scripts/run-backend.sh

cd "$(dirname "$0")/.."
export PROJECT_ROOT="$PWD"

echo "Starting Emotion API on port 5002..."
python -m uvicorn ml.app:app --reload --port 5002 &
EMOTION_PID=$!
sleep 3

echo "Starting Node backend on port 5001..."
cd server && node index.js &
NODE_PID=$!

trap "kill $EMOTION_PID $NODE_PID 2>/dev/null" EXIT
wait
