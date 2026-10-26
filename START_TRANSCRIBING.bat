@echo off
rem Double-click to start the Stage B hand-transcription page (100 clips, for WER).
rem Keep this window open while you type; it saves after every clip.
cd /d "%~dp0"
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8766"
.venv\Scripts\python.exe scripts\transcribe_server.py --annotator A1
pause
