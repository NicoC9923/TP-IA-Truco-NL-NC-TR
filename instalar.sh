#!/usr/bin/env sh
# Instala una sola vez las dependencias de Python y despues revisa el entorno.
# Los lanzadores (jugar_truco.sh, sim.sh) nunca instalan nada.
cd "$(dirname "$0")" || exit 1
python3 -m pip install --user -r requirements.txt || exit 1
if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "Instalando PyTorch (CPU, ~200 MB) para la IA de aprendizaje por refuerzo..."
    if [ "$(uname -s)" = Darwin ]; then
        if ! python3 -m pip install --user torch; then
            echo "No se pudo instalar PyTorch: la IA de aprendizaje por refuerzo va a aparecer desactivada. Probá de nuevo o instalalo a mano con: python3 -m pip install --user torch"
        fi
    else
        if ! python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu; then
            echo "No se pudo instalar PyTorch: la IA de aprendizaje por refuerzo va a aparecer desactivada. Probá de nuevo o instalalo a mano con: python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu"
        fi
    fi
fi
cd entorno && exec python3 -m sim --solo-revisar
