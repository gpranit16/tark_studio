@echo off
title TARK Studio - Clean Stop

echo ========================================================
echo   Stopping TARK Studio Local Services
echo ========================================================

echo Stopping Python Backend (Port 8000)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

echo Stopping Vite Frontend (Port 5173 / 5174)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5174 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

echo.
echo ========================================================
echo   All TARK Studio local services stopped cleanly!
echo ========================================================
timeout /t 2 /nobreak >nul
