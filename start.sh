#!/bin/bash
set -e
echo "=== SafeSight AI Backend ==="
echo "Starting Flask server..."
exec python3 /home/runner/workspace/python-backend/app.py
