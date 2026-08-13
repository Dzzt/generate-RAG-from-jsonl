@echo off
setlocal
cd /d "%~dp0"
py build_production.py %*
endlocal
