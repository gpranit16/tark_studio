@echo off
setlocal enabledelayedexpansion
title TARK Studio - 1-Click Sovereign Launcher

echo ========================================================
echo   TARK STUDIO - AUTOMATIC 1-CLICK LAUNCHER
echo ========================================================
echo.

REM 1. Check and start Ollama if not running
echo [1/4] Checking Ollama AI Engine...
curl.exe -s http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo       Ollama is starting in background...
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" app
    timeout /t 3 /nobreak >nul
) else (
    echo       Ollama is ready and active!
)

REM 2. Check and start Docker / PostgreSQL (pgvector)
echo [2/4] Checking PostgreSQL + pgvector (Docker)...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo       Docker Desktop is starting... Please wait a few seconds.
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    :WAIT_DOCKER
    timeout /t 3 /nobreak >nul
    docker info >nul 2>&1
    if !errorlevel! neq 0 goto WAIT_DOCKER
    echo       Docker is now ready!
)

REM Ensure tark_pgvector container is running
docker start tark_pgvector >nul 2>&1
if %errorlevel% neq 0 (
    echo       Starting database via docker compose...
    cd /d %~dp0 && docker compose up -d >nul 2>&1
)
echo       Database (PostgreSQL + pgvector) is online!

REM 3. Launch Backend
echo [3/4] Launching Backend on http://127.0.0.1:8000...
start "TARK Studio Backend (Port 8000)" cmd /k "cd /d %~dp0backend && python run.py"

REM 4. Launch Frontend
echo [4/4] Launching Frontend on http://localhost:5173...
start "TARK Studio Frontend (Port 5173)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================================
echo   ALL SYSTEMS GO! TARK Studio is running:
echo   - Local UI      : http://localhost:5173
echo   - Local Backend : http://127.0.0.1:8000
echo   - RTX 3050 GPU  : Ollama Active
echo ========================================================
echo   (Services are running in separate windows. Close this launcher.)
pause
