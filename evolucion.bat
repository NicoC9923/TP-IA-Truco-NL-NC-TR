@echo off
cd /d "%~dp0proyecto"
py -3 servidor_truco.py --sin-robot --abrir-evolucion %*
if errorlevel 1 pause
