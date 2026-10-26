@echo off
rem Double-click to open the interactive demo (local only: contains broadcast audio, do not share publicly).
cd /d "%~dp0reports\demo"
start "" cmd /c "timeout /t 2 >nul & start http://127.0.0.1:8792/index.html"
"%~dp0.venv\Scripts\python.exe" -m http.server 8792 --bind 127.0.0.1
