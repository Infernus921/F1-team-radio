@echo off
rem Double-click to start the Stage A speaker-attribution labelling page.
rem Keep this window open while you label; close it when you're done. Labels save after every click.
cd /d "%~dp0"
rem open the browser after a short delay so the server is up first
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8765"
.venv\Scripts\python.exe scripts\label_server.py --annotator A1
pause
