@echo off
setlocal enabledelayedexpansion
title TARK Studio - Production Local Engine (RTX 3050)

echo ========================================================
echo   TARK STUDIO — AIR-GAPPED ENGINE FOR PRODUCTION SITE
echo   Target: https://tark-studio.pages.dev
echo ========================================================
echo.

REM 1. Ensure Ollama is running
echo [1/3] Checking Ollama AI Engine (RTX 3050)...
curl.exe -s http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo       Ollama is starting in background...
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" app
    timeout /t 3 /nobreak >nul
) else (
    echo       Ollama is ready and active!
)

REM 2. Ensure Docker PostgreSQL container is running
echo [2/3] Checking Local Database (pgvector)...
docker start tark_pgvector >nul 2>&1
if %errorlevel% neq 0 (
    docker info >nul 2>&1
    if !errorlevel! neq 0 (
        echo       Starting Docker Desktop...
        start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        :WAIT_DOCKER
        timeout /t 3 /nobreak >nul
        docker info >nul 2>&1
        if !errorlevel! neq 0 goto WAIT_DOCKER
    )
    cd /d %~dp0 && docker compose up -d >nul 2>&1
)
echo       Database is online on port 5433!

REM 3. Launch Local Backend
echo [3/3] Starting Local FastAPI Backend on http://127.0.0.1:8000...
echo.
echo ========================================================
echo   LOCAL SOVEREIGN ENGINE IS NOW READY!
echo.
echo   Next Step:
echo   1. Open: https://tark-studio.pages.dev
echo   2. Click the top toggle: [ Local RTX 3050 ]
echo   3. Status will turn green: 0B Egress (127.0.0.1:8000)
echo.
echo   (Keep this window open while chatting. Close window to stop.)
echo ========================================================
echo.

cd /d %~dp0backend
python run.py
