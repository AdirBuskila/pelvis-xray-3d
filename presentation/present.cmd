@echo off
rem Serves the deck locally (the speaker view needs http, not file://) and opens it.
cd /d "%~dp0"
start "" http://127.0.0.1:8765/index.html
python -m http.server 8765 --bind 127.0.0.1
