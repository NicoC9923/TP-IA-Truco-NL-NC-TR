"""Verifica que todo este instalado ANTES de abrir el simulador.

No instala nada: si falta algo, dice exactamente que comando correr.

Nunca dejamos que el simulador arranque a medias: es peor un simulador que abre
y falla raro a los dos minutos que uno que no abre y explica por que.
"""

from __future__ import annotations

import importlib
import os
import sys

# Solo la copia que viene en el repo, para que nunca se use otra que haya
# quedado en la maquina.
UBICACIONES_REPO = [os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "unitree_mujoco")]

VERDE, ROJO, GRIS, FIN = "\033[92m", "\033[91m", "\033[90m", "\033[0m"

if os.name == "nt" or not sys.stdout.isatty():
    VERDE = ROJO = GRIS = FIN = ""


class Resultado:
    def __init__(self):
        self.problemas: list[str] = []
        self.repo: str | None = None

    @property
    def todo_bien(self) -> bool:
        return not self.problemas


def _ok(texto: str, detalle: str = "") -> None:
    print(f"  {VERDE}[OK]{FIN}    {texto}" + (f" {GRIS}{detalle}{FIN}" if detalle else ""))


def _falta(texto: str, detalle: str = "") -> None:
    print(f"  {ROJO}[FALTA]{FIN} {texto}" + (f" {GRIS}{detalle}{FIN}" if detalle else ""))


def buscar_repo_oficial() -> str | None:
    """Devuelve la carpeta del repo oficial si tiene lo que necesitamos."""
    for base in UBICACIONES_REPO:
        modelos = os.path.join(base, "unitree_robots")
        bridge = os.path.join(base, "simulate_python", "unitree_sdk2py_bridge.py")
        if os.path.isdir(modelos) and os.path.isfile(bridge):
            return base
    return None


def _tiene(modulo: str) -> bool:
    try:
        importlib.import_module(modulo)
        return True
    except Exception:
        return False


def _pip(paquete: str) -> str:
    # Dentro de un venv, pip rechaza --user.
    en_venv = sys.prefix != sys.base_prefix
    return f"{sys.executable} -m pip install {'' if en_venv else '--user '}{paquete}"


# Las tres DLL que MuJoCo necesita en Windows y que NO vienen con Python.
# Son parte del Visual C++ Redistributable de Microsoft.
_DLLS_WINDOWS = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
_URL_VCREDIST = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def _faltan_dlls_windows() -> list[str]:
    """Cuales de las DLL de Microsoft faltan. Vacio si estan todas o no es Windows."""
    if os.name != "nt":
        return []
    import ctypes.util

    faltantes = []
    for dll in _DLLS_WINDOWS:
        if ctypes.util.find_library(dll) is None and not os.path.exists(
                os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                             "System32", dll)):
            faltantes.append(dll)
    return faltantes


def diagnosticar_mujoco() -> tuple[bool, str]:
    """Dice si MuJoCo se puede importar y, si no, POR QUE.

    Existe porque `_tiene()` se traga la excepcion, y eso hacia un diagnostico
    directamente falso: en Windows sin el Visual C++ Redistributable, `pip
    install mujoco` termina **bien** pero `import mujoco` falla al cargar la
    DLL. Decir "falta MuJoCo" ahi manda a reinstalar algo que ya esta.
    """
    try:
        importlib.import_module("mujoco")
        return True, ""
    except ImportError as exc:
        texto = str(exc)
        faltan = _faltan_dlls_windows()
        # "DLL load failed" es el sintoma clasico del redistribuible ausente.
        if faltan or "DLL load failed" in texto:
            detalle = ", ".join(faltan) if faltan else "las DLL de Microsoft"
            return False, (
                f"MuJoCo ESTA instalado, pero no puede cargarse: falta el\n"
                f"      Microsoft Visual C++ Redistributable ({detalle}).\n"
                f"      No viene con Python ni con MuJoCo; se instala aparte:\n"
                f"        {_URL_VCREDIST}\n"
                f"      Despues de instalarlo, cerra y volve a abrir la consola.")
        if "mujoco" in texto.lower() and "no module" in texto.lower():
            return False, f"Falta MuJoCo. Instalalo con:\n      {_pip('mujoco')}"
        return False, f"No se pudo importar MuJoCo: {texto}"
    except Exception as exc:                                   # noqa: BLE001
        return False, f"No se pudo importar MuJoCo: {exc}"


def verificar(dds: bool = False) -> Resultado:
    """Revisa el entorno. No instala nada.

    `dds=True` pide ademas CycloneDDS y el SDK de Unitree. Va apagado por
    defecto: son las dos piezas que NO se pueden instalar en macOS ni en
    Windows. Ver la cabecera de `local.py`.
    """
    r = Resultado()
    print()
    print("  Revisando que este todo listo...")
    print()

    # 1. Python
    v = sys.version_info
    if v >= (3, 8):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _falta(f"Python {v.major}.{v.minor}", "hace falta 3.8 o mas nuevo")
        r.problemas.append(
            "Tu Python es muy viejo. Instala uno nuevo desde https://python.org")

    # 2. MuJoCo (la ventana 3D)
    ok_mujoco, porque = diagnosticar_mujoco()
    if ok_mujoco:
        import mujoco
        _ok("MuJoCo", f"v{mujoco.__version__}")
    else:
        _falta("MuJoCo", "instalado pero no carga" if "Visual C++" in porque else "")
        r.problemas.append(porque)

    # 3 y 4. CycloneDDS y el SDK de Unitree: SOLO en modo --dds.
    #
    # CycloneDDS 0.10.2 solo publica wheels hasta cp310 (con 3.11+ pip intenta
    # compilar y falla), y `unitree_sdk2py` resuelve `timerfd_create` por
    # ctypes al importar, que es una syscall de Linux.
    if dds:
        if _tiene("cyclonedds"):
            _ok("CycloneDDS")
        elif v < (3, 11):
            _falta("CycloneDDS")
            r.problemas.append(
                f"Falta CycloneDDS. Instalalo con:\n"
                f"      {_pip('cyclonedds==0.10.2')}")
        else:
            _falta("CycloneDDS")
            r.problemas.append(
                "Falta CycloneDDS y no se puede instalar en este Python.\n"
                f"      Tenes {v.major}.{v.minor} y la 0.10.2 solo tiene wheels\n"
                "      hasta 3.10. El modo --dds necesita Python 3.10 y Linux.\n"
                "      Sin --dds, el simulador anda igual y no hace falta nada "
                "de esto.")

        if _tiene("unitree_sdk2py"):
            _ok("SDK de Unitree (unitree_sdk2py)")
        else:
            _falta("SDK de Unitree")
            r.problemas.append(
                "Falta el SDK de Unitree. Se instala con:\n"
                "      git clone https://github.com/unitreerobotics/"
                "unitree_sdk2_python\n"
                f"      {sys.executable} -m pip install --user -e "
                f"unitree_sdk2_python")

        # El bridge oficial importa pygame incondicionalmente, aunque no haya
        # joystick. Sin esto el import explota y el error no dice por que.
        if _tiene("pygame"):
            _ok("pygame", "lo pide el bridge oficial de Unitree")
        else:
            _falta("pygame", "lo pide el bridge oficial de Unitree")
            r.problemas.append(f"Falta pygame. Instalalo con:\n      {_pip('pygame')}")

    # 5. Los modelos oficiales. Vienen en el repo.
    repo = buscar_repo_oficial()
    if repo:
        _ok("Modelos oficiales de Unitree", repo)
        r.repo = repo
    else:
        _falta("Modelos oficiales de Unitree")
        r.problemas.append(
            f"No encuentro los modelos en {UBICACIONES_REPO[0]}.\n"
            f"      Vienen en el repo: restauralos con\n"
            f"      git checkout -- entorno/sim/unitree_mujoco")

    return r


def informar(r: Resultado) -> bool:
    print()
    if r.todo_bien:
        print(f"  {VERDE}Todo listo.{FIN}")
        print()
        return True

    print("  " + "=" * 58)
    print(f"  {ROJO}FALTAN COSAS PARA PODER ABRIR EL SIMULADOR{FIN}")
    print("  " + "=" * 58)
    print()
    for i, p in enumerate(r.problemas, 1):
        print(f"  {i}. {p}")
        print()
    print("  " + "=" * 58)
    print()
    return False


if __name__ == "__main__":
    raise SystemExit(0 if informar(verificar()) else 1)
