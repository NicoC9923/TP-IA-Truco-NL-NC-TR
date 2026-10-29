#!/usr/bin/env sh
# Abre el simulador. Los flags pasan directo: ./sim.sh --robot go2 --vmax 1.0
cd "$(dirname "$0")/entorno" || exit 1
exec python3 -m sim "$@"
