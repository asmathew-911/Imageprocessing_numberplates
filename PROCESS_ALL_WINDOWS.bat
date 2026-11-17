@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo First run SETUP_AND_RUN_WINDOWS.bat to install dependencies.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" batch.py --dataset data/google_images --output results
".venv\Scripts\python.exe" reference_mode.py --dataset data/google_images --output results
pause
