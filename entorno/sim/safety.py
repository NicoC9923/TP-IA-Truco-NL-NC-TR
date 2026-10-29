"""Limites de movimiento. Por defecto NO hay ninguno: los fijas vos.

    python -m sim --robot go2                     # sin limites
    python -m sim --robot go2 --vmax 1.5 --tmax 30

Un limite en None significa "sin tope". Los validadores igual rechazan lo que
nunca tiene sentido (NaN, infinito, tiempo negativo), haya limites o no.

RECHAZA, no recorta: si pedis 2.0 m/s con un tope de 1.0 y se recortara en
silencio, la distancia recorrida (v x t) saldria a la mitad sin aviso.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Tolerancia para comparar contra el tope: 0.2 no es representable en binario.
EPSILON = 1e-9


class ErrorDeSeguridad(ValueError):
    """Un valor pedido es invalido o excede un limite. Se rechaza, no se recorta."""


@dataclass(frozen=True)
class Limites:
    velocidad_max: float | None = None          # m/s
    velocidad_angular_max: float | None = None  # rad/s
    duracion_max: float | None = None           # s, por comando individual

    def como_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def desde_dict(cls, datos: dict | None) -> "Limites":
        datos = datos or {}
        return cls(**{k: datos.get(k) for k in cls.__dataclass_fields__})

    def __str__(self) -> str:
        def _f(valor, unidad):
            return "sin tope" if valor is None else f"{valor} {unidad}"
        return (f"{_f(self.velocidad_max, 'm/s')} | "
                f"{_f(self.velocidad_angular_max, 'rad/s')} | "
                f"{_f(self.duracion_max, 's por orden')}")


SIN_LIMITES = Limites()


def _numero(valor: object, nombre: str) -> float:
    try:
        numero = float(valor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ErrorDeSeguridad(
            f"{nombre} tiene que ser un numero, no {type(valor).__name__}."
        ) from None
    if numero != numero or numero in (float("inf"), float("-inf")):
        raise ErrorDeSeguridad(f"{nombre} no puede ser NaN ni infinito.")
    return numero


def validar_velocidad(velocidad: object, lim: Limites = SIN_LIMITES) -> float:
    v = _numero(velocidad, "La velocidad")
    if lim.velocidad_max is not None and abs(v) > lim.velocidad_max + EPSILON:
        raise ErrorDeSeguridad(
            f"Velocidad {v:.3f} m/s: el tope es {lim.velocidad_max:.2f} m/s.")
    return v


def validar_velocidad_angular(velocidad: object, lim: Limites = SIN_LIMITES) -> float:
    v = _numero(velocidad, "La velocidad de giro")
    if (lim.velocidad_angular_max is not None
            and abs(v) > lim.velocidad_angular_max + EPSILON):
        raise ErrorDeSeguridad(
            f"Velocidad de giro {v:.3f} rad/s: el tope es "
            f"{lim.velocidad_angular_max:.2f} rad/s.")
    return v


def validar_duracion(duracion: object, lim: Limites = SIN_LIMITES) -> float:
    t = _numero(duracion, "El tiempo")
    if t < 0.0:
        raise ErrorDeSeguridad(f"El tiempo no puede ser negativo (pediste {t:.2f} s).")
    if lim.duracion_max is not None and t > lim.duracion_max + EPSILON:
        raise ErrorDeSeguridad(
            f"Tiempo {t:.2f} s: el tope por comando es {lim.duracion_max:.1f} s.")
    return t
