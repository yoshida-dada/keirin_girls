@echo off
rem KEIRIN girls EV-AI: launch the local resident scheduler.
rem Called from the Startup folder; runs live_scheduler.py in a minimized console.
rem NOTE: comments kept ASCII-only so cmd.exe (cp932) does not misparse them.
rem Prefers the project-local venv (migration\02_setup_new_pc.ps1 creates .venv); falls back to the
rem base Python 3.10 install this machine has used historically, so this stays a no-op change here.
cd /d "%~dp0.."
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "KEIRIN_PY=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "KEIRIN_PY=C:\Users\yoshi\AppData\Local\Programs\Python\Python310\python.exe"
)
start "KeirinGirlsLive" /min "%KEIRIN_PY%" "scripts\live_scheduler.py"
