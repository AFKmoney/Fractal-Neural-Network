@echo off
title NFN AGI — Neural Fractal Network
echo.
echo  +══════════════════════════════════════+
echo  ^|   Neural Fractal Network             ^|
echo  ^|   AGI -- All-in-One App              ^|
echo  ^|                                      ^|
echo  ^|   Chat · Train · Explore · Adapt     ^|
echo  +══════════════════════════════════════+
echo.

REM ── Check Python ─────────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found on your system.
    echo.
    echo  Install Python 3.10 or later from:
    echo  https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM ── Free port 8000 if already in use ─────────────────────────────────────────
echo  Checking port 8000...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 "') do (
    echo  Stopping previous instance (PID %%a)...
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM ── Launch ───────────────────────────────────────────────────────────────────
python run.py %*
if errorlevel 1 (
    echo.
    echo  [ERROR] The NFN AGI server encountered an error.
    echo.
    echo  Troubleshooting:
    echo    1. Make sure Python 3.10+ is installed
    echo    2. Install dependencies:  pip install -r requirements.txt
    echo    3. Check the logs above for details
    echo    4. If port 8000 is busy, use:  python run.py --port 8080
    echo.
    pause
)
