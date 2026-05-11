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
    echo  [ERROR] Python not found.
    echo  Install from https://www.python.org/downloads/
    echo  Check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

python --version
echo.

REM ── Install / check dependencies ─────────────────────────────────────────────
python -c "import numpy, torch, fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Installing required packages (first run)...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo  [ERROR] pip install failed.
        echo  Try: pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    echo.
)

REM ── Free port 8000 ───────────────────────────────────────────────────────────
echo  Freeing port 8000...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

REM ── Launch (try port 8000, fall back to 8080) ─────────────────────────────────
echo  Starting server...
echo.
python run.py %*
set EXIT_CODE=%errorlevel%

if %EXIT_CODE% equ 0 goto :done

REM If it failed, check if it was a port conflict and retry on 8080
echo.
echo  [INFO] Port 8000 may still be busy. Trying port 8080...
echo.
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8080 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 2 /nobreak >nul
python run.py --port 8080 %*
if errorlevel 1 (
    echo.
    echo  [ERROR] Could not start the server.
    echo.
    echo  Try running manually:
    echo    python run.py --port 9000
    echo.
    pause
)

:done
