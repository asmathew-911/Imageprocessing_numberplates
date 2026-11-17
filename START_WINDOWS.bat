@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
 echo Open http://127.0.0.1:8000 in your browser.
 ".venv\Scripts\python.exe" app.py
) else (
 echo First time setup is needed. Starting setup...
 call "%~dp0SETUP_AND_RUN_WINDOWS.bat"
)
pause
