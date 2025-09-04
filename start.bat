@echo off
cd /d "%~dp0"
start "Agent A (8081)" cmd /k python server_main.py

echo Started Agent A on 8081 in a new window.
exit /b 0
