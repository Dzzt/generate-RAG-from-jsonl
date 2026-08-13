@echo off
setlocal
cd /d "%~dp0"
py 02_embed_sample.py %*
endlocal
