@echo off
rem Start Jarvis. Double-click for voice mode, or run from a terminal:
rem     jarvis            voice mode (hold the talk key and speak)
rem     jarvis --text     type instead of speaking
setlocal
cd /d "%~dp0"
set "SYS=%SystemRoot%\System32"

rem Use this folder's virtual environment when there is one.
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

rem Start Ollama if it is installed but not running yet.
"%SYS%\tasklist.exe" /FI "IMAGENAME eq ollama.exe" 2>nul | "%SYS%\find.exe" /I "ollama.exe" >nul
if errorlevel 1 (
    "%SYS%\where.exe" ollama >nul 2>nul
    if not errorlevel 1 (
        echo Starting Ollama...
        start "" /B ollama serve >nul 2>nul
        "%SYS%\timeout.exe" /t 3 /nobreak >nul
    )
)

if /I "%~1"=="--text" (
    "%PY%" -m jarvis
) else (
    "%PY%" -m jarvis --voice %*
)

rem Keep the window open when something went wrong, so the message can be read.
if errorlevel 1 pause
endlocal
