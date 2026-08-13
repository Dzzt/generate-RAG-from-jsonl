@echo off
setlocal
cd /d "%~dp0"
py 01_sample_chunks.py %*
endlocal
