@echo off
rem KEIRINガールズ予測AI ローカル常駐スケジューラ 起動用。
rem タスクスケジューラ(ONLOGON)から呼ばれ、最小化コンソールで常駐する。
cd /d "%~dp0.."
start "KeirinGirlsLive" /min "C:\Users\yoshi\AppData\Local\Programs\Python\Python310\python.exe" "scripts\live_scheduler.py"
