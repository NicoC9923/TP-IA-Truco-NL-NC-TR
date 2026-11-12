@echo off
chcp 65001 >nul
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

py -3 -c "import torch" >nul 2>&1
if errorlevel 1 (
  echo Instalando PyTorch ^(CPU, ~200 MB^) para la IA de aprendizaje por refuerzo...
  py -3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 (
    echo No se pudo instalar PyTorch: la IA de aprendizaje por refuerzo va a aparecer desactivada. Probá de nuevo o instalalo a mano con: py -3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
  )
)

cd /d "%~dp0entorno"
py -3 -m sim --solo-revisar
pause
