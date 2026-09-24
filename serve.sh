#!/bin/bash
# Simple script to serve the app locally
# Usage: ./serve.sh

echo "Starting local server at http://localhost:8080"
echo "Open http://localhost:8080/index.html in your browser"
echo "Press Ctrl+C to stop"

python3 -m http.server 8080
