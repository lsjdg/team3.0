#!/bin/bash

# Script to watch both log files in real-time

echo "=== Watching Main Server and AI Server logs ==="
echo "Press Ctrl+C to stop"
echo ""

tail -f main_server.log ai_server.log 2>/dev/null
