#!/bin/bash
set -e

ENV=${1:-development}

echo "Starting RAG Project in $ENV mode..."

if [ "$ENV" = "development" ]; then
    docker compose -f docker/docker-compose.dev.yaml up -d
    echo "Waiting for services..."
    sleep 5
    python scripts/init_db.py
    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
else
    docker compose -f docker/docker-compose.yaml up -d
fi
