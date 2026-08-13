@echo off
setlocal
cd /d "%~dp0"
py 03_benchmark_indexes.py %*
endlocal
