@echo off
rem Abre el simulador. Los flags pasan directo: sim --robot go2 --vmax 1.0
cd /d "%~dp0entorno"
py -3 -m sim %*
if errorlevel 1 pause
