@echo off
setlocal
set SCRIPT_DIR=%~dp0
set VENV=%SCRIPT_DIR%.venv

echo.
echo ==========================================
echo  calculationNotebook PySide6 Launcher
echo ==========================================
echo.

cd /d "%SCRIPT_DIR%"

:: ── Step 1: Find Python 3.12 or 3.13 ──────────────────────────────
set PY_CMD=
for %%C in (python3.13 python3.12 python3 python) do (
    if "!PY_CMD!"=="" (
        where %%C >nul 2>&1 && (
            for /f "tokens=2 delims= " %%V in ('%%C --version 2^>^&1') do (
                echo %%V | findstr /r "^3\.\(12\|13\)" >nul && set PY_CMD=%%C
            )
        )
    )
)

if "%PY_CMD%"=="" (
    :: Fallback: try py launcher
    py -3.13 --version >nul 2>&1 && set PY_CMD=py -3.13
    if "%PY_CMD%"=="" py -3.12 --version >nul 2>&1 && set PY_CMD=py -3.12
)

if "%PY_CMD%"=="" (
    echo ERROR: Python 3.12 or 3.13 not found.
    echo Install from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: ── Step 2: Create venv if missing ────────────────────────────────
if not exist "%VENV%\Scripts\python.exe" (
    echo Creating virtual environment...
    %PY_CMD% -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: ── Step 3: Install/verify dependencies ───────────────────────────
"%VENV%\Scripts\python.exe" -c "import PySide6, numpy, plotly" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip --quiet
    "%VENV%\Scripts\python.exe" -m pip install -r "%SCRIPT_DIR%requirements.txt"
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
)

:: ── Step 4: Launch ─────────────────────────────────────────────────
if exist "%VENV%\Scripts\pythonw.exe" (
    start "" "%VENV%\Scripts\pythonw.exe" "%SCRIPT_DIR%main.py"
) else (
    start "" "%VENV%\Scripts\python.exe" "%SCRIPT_DIR%main.py"
)
exit
