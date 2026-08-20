@echo off
setlocal
cd /d "%~dp0"
py -3.13 build_production.py %*
endlocal
