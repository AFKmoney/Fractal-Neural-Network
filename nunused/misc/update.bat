@echo off
title NFN AGI — Update
echo.
echo  Downloading latest updates...
echo.

git pull
if errorlevel 1 (
    echo.
    echo  [WARNING] git pull failed. Make sure git is installed.
    echo  Download git from: https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo.
echo  Update complete. Starting app...
echo.
call start.bat
