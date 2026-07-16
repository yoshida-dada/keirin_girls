@echo off
rem KEIRIN girls EV-AI: launch the local resident scheduler.
rem Called from the Startup folder; runs live_scheduler.py in a minimized console.
rem NOTE: comments kept ASCII-only so cmd.exe (cp932) does not misparse them.
cd /d "%~dp0.."
start "KeirinGirlsLive" /min "C:\Users\yoshi\AppData\Local\Programs\Python\Python310\python.exe" "scripts\live_scheduler.py"
