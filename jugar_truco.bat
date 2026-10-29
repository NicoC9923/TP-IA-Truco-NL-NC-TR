@echo off
title Truco & Voz con Ottoman (Unitree G1)

rem Lanza el simulador 3D del Unitree G1 en otra ventana si no esta corriendo
start "Simulador Unitree G1" "%~dp0sim.bat" --robot g1
timeout /t 2 /nobreak >nul

rem Levanta el servidor web de Truco + Voz y abre el navegador en http://127.0.0.1:8080
cd /d "%~dp0proyecto"
py -3 servidor_truco.py --abrir %*
if errorlevel 1 pause
