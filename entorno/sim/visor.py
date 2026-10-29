"""Chequeo de si se puede abrir la ventana 3D de MuJoCo."""

from __future__ import annotations


def mujoco_disponible() -> tuple[bool, str]:
    try:
        import mujoco  # noqa: F401
        import mujoco.viewer  # noqa: F401
        return True, ""
    except Exception as exc:
        return False, str(exc)
