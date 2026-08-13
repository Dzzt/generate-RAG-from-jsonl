@echo off
setlocal
cd /d "%~dp0"
py diagnose_title.py %*
endlocal
