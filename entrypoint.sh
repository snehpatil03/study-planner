#!/bin/bash
set -euo pipefail

# Default host and port
HOST="0.0.0.0"
PORT="8080"

# Allow overriding host and port via environment
if [[ -n "${PORT_OVERRIDE:-}" ]]; then
    PORT="$PORT_OVERRIDE"
fi

exec uvicorn ai_personalized_learning_assistant.api.app:app --host "$HOST" --port "$PORT"
