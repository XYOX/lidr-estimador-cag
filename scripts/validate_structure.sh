#!/usr/bin/env bash
# Valida que la estructura de carpetas del Proyecto 1 (estimador-cag) este completa.
set -euo pipefail

REQUIRED_FILES=(
    "app/__init__.py"
    "app/main.py"
    "app/config.py"
    "app/routers/__init__.py"
    "app/routers/estimations.py"
    "app/services/__init__.py"
    "app/services/llm_service.py"
    "app/context/__init__.py"
    "app/context/examples.py"
    "pyproject.toml"
    ".env.example"
    ".gitignore"
    "README.md"
)

missing=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "FALTA: $file"
        missing=1
    fi
done

if [ -f ".env" ]; then
    if git ls-files --error-unmatch .env >/dev/null 2>&1; then
        echo "ERROR: .env esta versionado en git (no deberia estarlo)"
        missing=1
    fi
fi

if [ "$missing" -eq 1 ]; then
    echo "Estructura de carpetas invalida."
    exit 1
fi

echo "Estructura de carpetas OK."
