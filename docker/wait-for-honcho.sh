#!/bin/sh
# Wait for Honcho API to be ready before starting Hermes gateway
HONCHO_URL="${HONCHO_BASE_URL:-http://hermes-honcho-api:8000}"
MAX_WAIT=120
INTERVAL=3
elapsed=0

echo "Waiting for Honcho at $HONCHO_URL ..."
while [ $elapsed -lt $MAX_WAIT ]; do
    if python3 -c "import urllib.request,sys; urllib.request.urlopen(sys.argv[1], timeout=5)" "$HONCHO_URL/openapi.json" 2>/dev/null; then
        echo "Honcho is ready (${elapsed}s)"
        break
    fi
    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
done

if [ $elapsed -ge $MAX_WAIT ]; then
    echo "WARNING: Honcho not ready after ${MAX_WAIT}s, starting anyway"
fi

# Delegate to the original entrypoint
exec /opt/hermes/docker/entrypoint.sh "$@"
