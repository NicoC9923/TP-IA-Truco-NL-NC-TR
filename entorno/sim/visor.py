"""Chequeo de si se puede abrir la ventana 3D de MuJoCo."""

from __future__ import annotations

import importlib.util
import sys


def mujoco_disponible() -> tuple[bool, str]:
    try:
        import mujoco  # noqa: F401
        import mujoco.viewer  # noqa: F401
        return True, ""
    except Exception as exc:
        return False, str(exc)


def comando_instalar_mujoco() -> str:
    """The one-off install command for MuJoCo (same version as requirements.txt)."""
    en_venv = sys.prefix != sys.base_prefix
    return f"{sys.executable} -m pip install {'' if en_venv else '--user '}mujoco==3.12.0"


def aviso_sin_mujoco(buscar=importlib.util.find_spec) -> str | None:
    """A Spanish warning when MuJoCo is not installed, else None. Never installs.

    Only looks for the package (no import), so the game server can call it at
    startup without paying MuJoCo's import time.
    """
    if buscar("mujoco") is not None:
        return None
    return ("Vista 3D desactivada: falta MuJoCo, asi que no se ven el robot ni las "
            "cartas en la mesa. El Truco en la web funciona igual.\n"
            "          Instalalo una vez con instalar.bat (o instalar.sh), o con:\n"
            f"          {comando_instalar_mujoco()}")
