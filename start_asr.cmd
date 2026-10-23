@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B -u realtime_asr.py %*
set ASR_EXIT=%ERRORLEVEL%
echo.
pause
exit /b %ASR_EXIT%
