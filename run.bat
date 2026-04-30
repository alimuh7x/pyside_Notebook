@echo off
set SCRIPT_DIR=%~dp0
set VENV=%SCRIPT_DIR%.venv

cd /d "%SCRIPT_DIR%"

if not exist "%VENV%\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: Could not create venv. Is Python installed?
        pause
        exit /b 1
    )
)

"%VENV%\Scripts\python.exe" -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    "%VENV%\Scripts\python.exe" -m pip install -r "%SCRIPT_DIR%requirements.txt"
    if errorlevel 1 (
        echo ERROR: pip install failed.
        pause
        exit /b 1
    )
)

echo Launching...
"%VENV%\Scripts\python.exe" "%SCRIPT_DIR%main.py"
pause
