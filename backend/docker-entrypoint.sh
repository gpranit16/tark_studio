#!/bin/sh
set -e

echo "[TARK Studio] Running database migrations..."
alembic upgrade head
echo "[TARK Studio] Database migrations applied successfully."

echo "[TARK Studio] Starting FastAPI backend with Uvicorn..."
exec uvicorn app.main:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8000}" --workers "${UVICORN_WORKERS:-2}"
