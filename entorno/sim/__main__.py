"""Arranque del simulador de robots Unitree.

    python -m sim --robot g1
    python -m sim --robot go2 --vmax 1.0 --tmax 30

Verifica el entorno, levanta MuJoCo con el modelo oficial, abre el transporte
(socket local por defecto, DDS con --dds) y la ventana. Sin flags de limites,
no hay limites.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import signal
import sys
import threading
import time

from .mundo import Mundo
from .robots import ROBOTS, obtener
from .safety import Limites
from .verificar import informar, verificar

ARCHIVO_ACTIVO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".simulador_activo.json")
ARCHIVO_POSE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".pose_actual.json")


def _publicar_activo(robot, domain, interfaz, puerto=None):
    """Deja escrito como encontrar al simulador. Lo lee `robot.py`."""
    from .local import PUERTO

    # `transporte` es lo que lee `robot.py` para saber por donde hablar. Sin
    # este dato el cliente adivinaba, y con el simulador levantado en --dds
    # se quedaba golpeando un socket que nadie abrio.
    datos = {"robot": robot.clave,
             "transporte": "dds" if puerto is False else "local",
             "domain": domain, "interfaz": interfaz,
             "pose_archivo": ARCHIVO_POSE,
             "puerto": PUERTO if puerto in (None, False) else puerto}
    with open(ARCHIVO_ACTIVO, "w", encoding="utf-8") as f:
        json.dump(datos, f)


# ---------------------------------------------------------------------------
#  Un simulador por vez
# ---------------------------------------------------------------------------
#
# Dos simuladores abiertos a la vez pelean por el mismo puerto (o por el mismo
# dominio DDS), y tu programa terminaria hablandole al otro robot sin ningun
# error que avise. La sena va al temp del sistema para detectarlo aunque los
# dos se hayan abierto desde carpetas distintas.

ARCHIVO_SENA = os.path.join(tempfile.gettempdir(), "unitree_sim_activo.json")


def _sena_viva():
    """El otro simulador anotado en la sena, si sigue vivo. Si no, None."""
    try:
        with open(ARCHIVO_SENA, encoding="utf-8") as f:
            datos = json.load(f)
        pid = int(datos["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if pid == os.getpid():
        return None
    try:
        # Senal 0: no hace nada, solo pregunta si el proceso existe.
        os.kill(pid, 0)
    except OSError:
        return None      # murio mal y dejo la sena colgada: no bloquea
    return datos


def _dejar_sena(robot) -> None:
    try:
        with open(ARCHIVO_SENA, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "robot": robot.clave}, f)
    except OSError:
        pass


def _borrar_sena() -> None:
    try:
        datos = json.load(open(ARCHIVO_SENA, encoding="utf-8"))
        if int(datos.get("pid", -1)) == os.getpid():
            os.remove(ARCHIVO_SENA)
    except (OSError, ValueError, TypeError):
        pass


def _limpiar():
    _borrar_sena()
    for f in (ARCHIVO_ACTIVO, ARCHIVO_POSE):
        try:
            os.remove(f)
        except OSError:
            pass


def _publicar_pose(mundo, detener):
    """Tu programa puede leer la pose de aca.

    Va por archivo y no por DDS porque el simulador oficial publica la pose del
    Go2 en rt/sportmodestate, pero la del G1 no: el G1 usa mensajes unitree_hg,
    que no traen ese topico. Un archivo funciona igual para los dos.
    """
    while not detener.is_set():
        try:
            with open(ARCHIVO_POSE, "w", encoding="utf-8") as f:
                json.dump(mundo.leer(), f)
        except OSError:
            pass
        time.sleep(0.1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="sim", description="Simulador de robots Unitree (G1 / Go2)")
    ap.add_argument("--robot", default="g1", choices=sorted(ROBOTS))
    ap.add_argument("--vmax", type=float, default=None,
                    help="tope de velocidad lineal en m/s (por defecto, sin tope)")
    ap.add_argument("--wmax", type=float, default=None,
                    help="tope de velocidad angular en rad/s (por defecto, sin tope)")
    ap.add_argument("--tmax", type=float, default=None,
                    help="tope de duracion por orden en s (por defecto, sin tope)")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--interfaz", default="lo")
    ap.add_argument("--sin-ventana", action="store_true")
    ap.add_argument("--silencioso", action="store_true")
    ap.add_argument("--dds", action="store_true",
                    help="usa DDS y el SDK real en vez del socket local "
                         "(solo Linux, Python 3.10)")
    ap.add_argument("--puerto", type=int, default=None,
                    help="puerto del socket local (por defecto 8765)")
    ap.add_argument("--igualmente", action="store_true",
                    help="abre aunque parezca haber otro simulador andando")
    ap.add_argument("--solo-revisar", action="store_true",
                    help="revisa el entorno y sale")
    args = ap.parse_args(argv)

    # 1. Verificar el entorno ANTES de prometer nada.
    r = verificar(dds=args.dds)
    if not informar(r):
        return 1
    if args.solo_revisar:
        return 0

    otro = None if args.igualmente else _sena_viva()
    if otro:
        print()
        print("  YA HAY UN SIMULADOR ABIERTO")
        print(f"    Robot   : {otro.get('robot', '?')}")
        print(f"    Proceso : {otro.get('pid', '?')}")
        print("  Cerralo o agrega --igualmente.")
        print()
        return 1

    robot = obtener(args.robot)
    limites = Limites(velocidad_max=args.vmax,
                      velocidad_angular_max=args.wmax,
                      duracion_max=args.tmax)
    mundo = Mundo(limites)

    try:
        if args.dds:
            from .arrancar import SimuladorOficial
            sim = SimuladorOficial(mundo, robot, r.repo, args.domain,
                                   args.interfaz, verboso=not args.silencioso)
        else:
            from .simulador import SimuladorLocal
            sim = SimuladorLocal(mundo, robot, r.repo,
                                 verboso=not args.silencioso,
                                 puerto=args.puerto)
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}\n", file=sys.stderr)
        return 1

    _publicar_activo(robot, args.domain, args.interfaz,
                     puerto=(False if args.dds else args.puerto))
    _dejar_sena(robot)
    detener = threading.Event()
    threading.Thread(target=_publicar_pose, args=(mundo, detener),
                     daemon=True).start()

    print("=" * 62)
    print("  SIMULADOR UNITREE")
    print("=" * 62)
    print(f"  Robot     : {robot.nombre} ({robot.tipo})")
    print(f"  Escena    : {robot.escena}")
    print(f"  Limites   : {limites}")
    if args.dds:
        print(f"  Transporte: DDS, dominio {args.domain}, "
              f"interfaz {args.interfaz}")
    else:
        from .local import HOST, PUERTO
        print(f"  Transporte: local, {HOST}:{args.puerto or PUERTO}")
    print("=" * 62)
    print()
    print("  Deja esta ventana abierta y ejecuta tu programa aparte.")
    print("  Para cerrar: Ctrl+C")
    print()

    # Ctrl+C tiene que cerrar SIEMPRE. Con la ventana abierta el hilo principal
    # queda bloqueado adentro del visor de MuJoCo y Python no llega a procesar
    # el KeyboardInterrupt: el simulador se quedaria vivo ocupando el puerto.
    def _cerrar(signum, frame):
        detener.set()
        _limpiar()
        print("\n[SIM] Simulador cerrado.")
        sys.stdout.flush()
        os._exit(0)

    signal.signal(signal.SIGINT, _cerrar)
    signal.signal(signal.SIGTERM, _cerrar)

    from .visor import mujoco_disponible

    hay_ventana, motivo = mujoco_disponible()
    try:
        if args.sin_ventana or not hay_ventana:
            if not args.sin_ventana:
                print(f"  Sin ventana 3D ({motivo.splitlines()[0]}): "
                      f"sigo en modo consola.")
                print()
            sim.correr_sin_ventana()
        else:
            sim.correr_con_ventana()
    except KeyboardInterrupt:
        pass
    finally:
        detener.set()
        _limpiar()
        print("\n[SIM] Simulador cerrado.")
        sys.stdout.flush()

    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
