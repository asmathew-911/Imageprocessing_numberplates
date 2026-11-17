@echo off
setlocal
cd /d "%~dp0"
echo PlateVision v5 - First time setup
if not exist "app.py" (
 echo ERROR: app.py is missing. Re-extract the downloaded ZIP to this folder.
 pause
 exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
 py -3.12 -m venv .venv
 if errorlevel 1 python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
 echo ERROR: Could not create virtual environment. Check your Python installation.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
 echo ERROR: Package installation failed.
 pause
 exit /b 1
)
echo.
echo Open http://127.0.0.1:8000 in Chrome/Edge.
".venv\Scripts\python.exe" app.py
pause
