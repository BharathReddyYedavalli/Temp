@echo off
echo ================================
echo Starting Glaucoma Detection Backend Server...
echo ================================
cd /d "c:\Users\bhara\OneDrive\Desktop\GlaucoAI\backend"
echo Installing/updating dependencies...
C:/Users/bhara/AppData/Local/Microsoft/WindowsApps/python3.10.exe -m pip install -r requirements.txt
echo Starting server on http://localhost:8000
C:/Users/bhara/AppData/Local/Microsoft/WindowsApps/python3.10.exe app.py
pause
