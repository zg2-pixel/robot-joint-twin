@echo off
setlocal
cd /d "%~dp0"
title Robot Joint Twin Dashboard
if not exist ".venv\Scripts\python.exe" (
    echo The project Python environment is missing.
    echo Create .venv and install requirements.txt before starting.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" launch_dashboard.py
if errorlevel 1 pause
