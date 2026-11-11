"""Human-like pacing for the robots: how long each event takes (seconds).

Pure module, no game or simulator dependency, so the human-vs-robot match and
the spectator mode use the same numbers.

    ritmo = Ritmo.humano(velocidad=1.0)        # real pacing
    ritmo = Ritmo.instantaneo()                # tests: zero delays, synchronous
    espera = ritmo.retraso_pensar(["truco"])   # 2.5-4.5 s, a big decision
    espera = ritmo.retraso(CARTA)              # arm reaches and lays the card
    espera = ritmo.retraso(SEGUIR)             # 0.5-1 s, next step of a chain

`velocidad` is a speed multiplier: x2 halves every delay, x0.5 doubles it.
The card timings mirror the 3D table's own `Ritmo` (`entorno/sim/cartas.py`,
`alcanzar` + `llevar`); a test keeps both in sync.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Iterable, Optional

# Events accepted by `Ritmo.retraso`.
PENSAR = "pensar"                # an ordinary decision
PENSAR_GRANDE = "pensar_grande"  # a big call or accepting one
SEGUIR = "seguir"                # the next step of a chain already thought out
CARTA = "carta"                  # the arm reaching, then laying the card
BAZA = "baza"                    # short pause after a baza is decided
MANO = "mano"                    # short pause between hands
EVENTOS = (PENSAR, PENSAR_GRANDE, SEGUIR, CARTA, BAZA, MANO)

VELOCIDADES = (0.5, 1.0, 2.0)

# Calls that make a real player stop and think. The strings are the action
# names of `voz_jugador` (without importing it).
ACCIONES_GRANDES = frozenset({
    "truco", "retruco", "vale_cuatro", "falta_envido", "real_envido",
})


def es_decision_grande(accion: Optional[str], pendiente: Optional[str] = None) -> bool:
    """True when `accion` deserves a longer think.

    `pendiente` is the call being answered, if any: accepting (or refusing)
    a big call is a big decision even though "quiero" alone is not.
    """
    if accion in ACCIONES_GRANDES:
        return True
    return pendiente in ACCIONES_GRANDES


@dataclass(frozen=True)
class Ritmo:
    pensar: tuple = (1.5, 3.0)
    pensar_grande: tuple = (2.5, 4.5)
    seguir: tuple = (0.5, 1.0)   # answer -> tanto -> card: no second full think
    alcanzar: float = 0.45       # arm reaching for the card (3D Ritmo.alcanzar)
    colocar: float = 0.8         # laying it on the table (3D Ritmo.llevar)
    pausa_baza: float = 0.9
    pausa_mano: float = 1.6
    velocidad: float = 1.0
    es_instantaneo: bool = False  # zero delays AND synchronous robot turns

    # ---- constructors ----
    @classmethod
    def humano(cls, velocidad: float = 1.0) -> "Ritmo":
        return cls().con_velocidad(velocidad)

    @classmethod
    def instantaneo(cls) -> "Ritmo":
        """Zero delays and synchronous robot turns (tests, headless runs)."""
        return cls(es_instantaneo=True)

    def con_velocidad(self, velocidad: float) -> "Ritmo":
        velocidad = float(velocidad)
        if velocidad not in VELOCIDADES:
            raise ValueError(f"Velocidad invalida: {velocidad}. Usa {VELOCIDADES}.")
        return replace(self, velocidad=velocidad)

    # ---- delays ----
    def _escala(self, segundos: float) -> float:
        return 0.0 if self.es_instantaneo else max(0.0, segundos) / self.velocidad

    def retraso(self, evento: str, rng: Optional[random.Random] = None) -> float:
        """Seconds to wait for `evento`. Thinking is randomized inside its range."""
        rng = rng or random
        if evento == PENSAR:
            return self._escala(rng.uniform(*self.pensar))
        if evento == PENSAR_GRANDE:
            return self._escala(rng.uniform(*self.pensar_grande))
        if evento == SEGUIR:
            return self._escala(rng.uniform(*self.seguir))
        if evento == CARTA:
            return self._escala(self.alcanzar + self.colocar)
        if evento == BAZA:
            return self._escala(self.pausa_baza)
        if evento == MANO:
            return self._escala(self.pausa_mano)
        raise ValueError(f"Evento desconocido: {evento!r}")

    def retraso_pensar(self, acciones: Iterable[str] = (), pendiente: Optional[str] = None,
                       rng: Optional[random.Random] = None) -> float:
        """Think time for a decision that produced `acciones` (voz_jugador names).

        Longer when any of them is a big call or answers one (`pendiente`).
        """
        grande = es_decision_grande(None, pendiente) or any(es_decision_grande(a) for a in acciones)
        return self.retraso(PENSAR_GRANDE if grande else PENSAR, rng)

    @property
    def rango_pensar(self) -> tuple:
        """(min, max) seconds of an ordinary think at the current speed."""
        return (self._escala(self.pensar[0]), self._escala(self.pensar[1]))


def desde_entorno(valor: Optional[str]) -> Ritmo:
    """Builds a Ritmo from a CLI/env value: "0" or "sin" = instant, "0.5"/"1"/"2" = speed."""
    if valor is None or str(valor).strip() == "":
        return Ritmo.humano()
    texto = str(valor).strip().lower()
    if texto in ("0", "sin", "off", "instantaneo"):
        return Ritmo.instantaneo()
    return Ritmo.humano(float(texto.replace("x", "")))
