#!/usr/bin/env bash
cd "$(dirname "$0")/proyecto" || exit 1
python3 servidor_truco.py --abrir "$@"
