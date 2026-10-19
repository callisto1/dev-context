#!/bin/bash
set -e

# Start Ollama server in the background
ollama serve &
pid=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done
echo "Ollama is ready."

# Pull required models
for model in ${OLLAMA_MODELS:-"qwen2.5-coder:14b qwen2.5:14b"}; do
  echo "Pulling model: $model"
  ollama pull "$model"
done

echo "All models ready."

# Wait for the Ollama server process
wait $pid
