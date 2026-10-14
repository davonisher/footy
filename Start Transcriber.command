#!/bin/bash
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

echo "Batch Transcriber starten…"
echo "Model: ~/Library/Application Support/com.pais.handy/models/ggml-large-v3-turbo.bin"
echo "Transcripts → ~/Documents/Transcripts"
echo ""

# Open browser shortly after server boots
(sleep 1.2 && open "http://127.0.0.1:5055") &

exec arch -arm64 ./.venv/bin/python app.py
