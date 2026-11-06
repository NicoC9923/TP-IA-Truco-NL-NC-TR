@echo off
title Instalar dependencias del Truco con Ottoman

rem Instala una sola vez las dependencias de Python y despues revisa el entorno.
rem Los lanzadores (jugar_truco.bat, sim.bat) nunca instalan nada.
py -3 -m pip install --user -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo.
  echo  No se pudieron instalar las dependencias. Revisa el mensaje de arriba.
  pause
  exit /b 1
)

cd /d "%~dp0entorno"
py -3 -m sim --solo-revisar
pause
