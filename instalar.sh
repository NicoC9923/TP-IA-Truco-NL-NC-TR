#!/usr/bin/env sh
# Instala una sola vez las dependencias de Python y despues revisa el entorno.
# Los lanzadores (jugar_truco.sh, sim.sh) nunca instalan nada.
cd "$(dirname "$0")" || exit 1
python3 -m pip install --user -r requirements.txt || exit 1
cd entorno && exec python3 -m sim --solo-revisar
