#!/usr/bin/env bash
# Levanta el servicio y valida que responda correctamente, sin depender de una API key real.
set -euo pipefail

PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"

uv run uvicorn app.main:app --port "$PORT" > /tmp/estimador-cag-ci.log 2>&1 &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Esperando a que el servicio arranque (PID $SERVER_PID)..."
for i in $(seq 1 20); do
    if curl -s -o /dev/null "$BASE_URL/health"; then
        break
    fi
    sleep 0.5
done

echo "-- Verificando GET /health --"
HEALTH_STATUS=$(curl -s -o /tmp/health.json -w "%{http_code}" "$BASE_URL/health")
cat /tmp/health.json
echo ""
if [ "$HEALTH_STATUS" != "200" ]; then
    echo "ERROR: /health devolvio status $HEALTH_STATUS"
    cat /tmp/estimador-cag-ci.log
    exit 1
fi

echo "-- Verificando que /api/v1/estimate este registrado en el OpenAPI schema --"
OPENAPI_STATUS=$(curl -s -o /tmp/openapi.json -w "%{http_code}" "$BASE_URL/openapi.json")
if [ "$OPENAPI_STATUS" != "200" ]; then
    echo "ERROR: /openapi.json devolvio status $OPENAPI_STATUS"
    exit 1
fi

if ! grep -q '"/api/v1/estimate"' /tmp/openapi.json; then
    echo "ERROR: el endpoint /api/v1/estimate no esta registrado"
    exit 1
fi

echo "-- Verificando que /docs (Swagger) responda --"
DOCS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/docs")
if [ "$DOCS_STATUS" != "200" ]; then
    echo "ERROR: /docs devolvio status $DOCS_STATUS"
    exit 1
fi

echo "Servicio validado correctamente."
