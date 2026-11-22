#!/bin/sh
set -e

echo "[TARK Studio] Running database migrations..."
alembic upgrade head
echo "[TARK Studio] Database migrations applied successfully."

PORT="${PORT:-${APP_PORT:-8000}}"
echo "[TARK Studio] Starting FastAPI backend with Uvicorn on port ${PORT}..."
exec uvicorn app.main:app --host "${APP_HOST:-0.0.0.0}" --port "${PORT}" --workers "${UVICORN_WORKERS:-1}"
