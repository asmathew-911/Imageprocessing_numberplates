@echo off
cd /d "%~dp0"
echo PlateVision v4 - Full dataset XML audit + separate classical ANPR
if not exist "app.py" (
 echo ERROR: app.py not found in this folder.
 pause
 exit /b 1
)
if not exist "dataset_reference_catalog.json" (
 echo ERROR: catalog is missing. Extract the full PlateVision v4 ZIP.
 pause
 exit /b 1
)
python app.py
if errorlevel 1 (
 echo Ensure dependencies are installed with: python -m pip install -r requirements.txt
 pause
)
