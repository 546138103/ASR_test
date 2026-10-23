@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
".venv\Scripts\python.exe" -B -u realtime_asr_test.py %*
set ASR_EXIT=%ERRORLEVEL%
pause
exit /b %ASR_EXIT%
