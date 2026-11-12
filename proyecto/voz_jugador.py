"""What a real Truco player says out loud (ADR 0066), per action.

Pure module, no game or simulator dependency (shared with spectator mode).
Only calls, answers, the tanto when the rules require it, and short phrases
when playing a card. Never reasoning, never cards, never a tanto outside the
announcement. "Mentira" is a UI word; robots do not say it, and never "farol".

    frase(TRUCO)                         -> "¡Truco!"
    frase(CARTA, rng=rng)                -> "Ahí va." or None (silence)
    frase_tanto(31, 28, robot_es_mano=True)   -> "Treinta y uno."
    perfil_voz(1)                        -> voice preferences, pitch, rate
    gesto(TRUCO)                         -> "desafiar" (a `Robot` method name)
"""

from __future__ import annotations

import random
from typing import Optional

# ---- actions ----
TRUCO = "truco"
RETRUCO = "retruco"
VALE_CUATRO = "vale_cuatro"
ENVIDO = "envido"
ENVIDO_ENVIDO = "envido_envido"
REAL_ENVIDO = "real_envido"
FALTA_ENVIDO = "falta_envido"
QUIERO = "quiero"
NO_QUIERO = "no_quiero"
MAZO = "mazo"
CARTA = "carta"
TANTO = "tanto"
SON_BUENAS = "son_buenas"

ACCIONES = (TRUCO, RETRUCO, VALE_CUATRO, ENVIDO, ENVIDO_ENVIDO, REAL_ENVIDO, FALTA_ENVIDO,
            QUIERO, NO_QUIERO, MAZO, CARTA, SON_BUENAS)

# Engine call names (partida_truco / symmetric engine) -> action.
_DESDE_CANTO = {
    "TRUCO": TRUCO, "RETRUCO": RETRUCO, "VALE_CUATRO": VALE_CUATRO,
    "ENVIDO": ENVIDO, "ENVIDO_ENVIDO": ENVIDO_ENVIDO, "REAL_ENVIDO": REAL_ENVIDO,
    "FALTA_ENVIDO": FALTA_ENVIDO, "QUIERO": QUIERO, "NO_QUIERO": NO_QUIERO,
    "MAZO": MAZO, "FOLD": MAZO,
}


def accion_de_canto(canto: str) -> Optional[str]:
    """Maps an engine call name ("REAL_ENVIDO", "fold", "quiero"...) to an action."""
    return _DESDE_CANTO.get(str(canto).strip().upper())


_FRASES = {
    TRUCO: ("¡Truco!",),
    RETRUCO: ("¡Quiero retruco!",),
    VALE_CUATRO: ("¡Quiero vale cuatro!",),
    ENVIDO: ("¡Envido!",),
    ENVIDO_ENVIDO: ("¡Envido, envido!",),
    REAL_ENVIDO: ("¡Real envido!",),
    FALTA_ENVIDO: ("¡Falta envido!",),
    QUIERO: ("¡Quiero!",),
    NO_QUIERO: ("No quiero.",),
    MAZO: ("Me voy al mazo.",),
    SON_BUENAS: ("Son buenas.",),
}

# Phrases when laying a card. They never depend on the card. None = silence.
_FRASES_CARTA = ("Ahí va.", "Tomá.", "Esta.", "Vamos.", None, None)

_UNIDADES = ("cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
             "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
             "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
             "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve")


def numero_en_palabras(n: int) -> str:
    """Spanish number words for a tanto (0-33): 27 -> "veintisiete", 33 -> "treinta y tres"."""
    if type(n) is not int or n < 0 or n > 33:
        raise ValueError(f"Tanto fuera de rango: {n!r}")
    if n < 30:
        return _UNIDADES[n]
    return "treinta" if n == 30 else f"treinta y {_UNIDADES[n - 30]}"


def frase_tanto(propio: int, rival: Optional[int] = None, robot_es_mano: bool = True) -> str:
    """What the robot says when the tantos are compared after an accepted envido.

    The mano speaks first and always says its number. The pie says "son buenas"
    when it cannot beat the mano (a tie goes to the mano), otherwise its number.
    `rival` is only used for the pie's decision; a human mano's number is on screen.
    """
    if robot_es_mano or rival is None:
        return numero_en_palabras(propio).capitalize() + "."
    if propio > rival:
        return numero_en_palabras(propio).capitalize() + "."
    return _FRASES[SON_BUENAS][0]


def frase(accion: str, rng: Optional[random.Random] = None) -> Optional[str]:
    """The phrase for `accion`, or None when a real player would stay quiet."""
    rng = rng or random
    if accion == CARTA:
        return rng.choice(_FRASES_CARTA)
    opciones = _FRASES.get(accion)
    if not opciones:
        raise ValueError(f"Accion sin frase: {accion!r}")
    return rng.choice(opciones)


# ---- gestures: names of `Robot` methods ----
_GESTOS = {
    TRUCO: "desafiar", RETRUCO: "desafiar", VALE_CUATRO: "desafiar",
    ENVIDO: "desafiar", ENVIDO_ENVIDO: "desafiar", REAL_ENVIDO: "desafiar",
    FALTA_ENVIDO: "desafiar", MAZO: "gesto_mazo", NO_QUIERO: "gesto_mazo",
}


def gesto(accion: str) -> Optional[str]:
    """The `Robot` gesture method to trigger with `accion`, if any."""
    return _GESTOS.get(accion)


# ---- voice profiles per seat ----
# `preferencias`: whole words of speechSynthesis voice names, best first
# (matched as words, so "Male" never matches "Female"). The page picks the
# first es-AR/es voice that matches, then any es-AR, any es. Seat 2 avoids the
# voice seat 1 got when there is another Spanish voice. The names cover the
# usual Windows (Microsoft ...), Edge "Online (Natural)" and Chrome voices.
_PERFILES = {
    1: {"nombre": "Ottoman", "idioma": "es-AR", "tono": 0.75, "velocidad": 1.0,
        "preferencias": ["Tomás", "Diego", "Álvaro", "Jorge", "Pablo", "Raúl", "Gerardo",
                         "Juan", "Male"]},
    2: {"nombre": "Segundo", "idioma": "es-AR", "tono": 1.25, "velocidad": 1.08,
        "preferencias": ["Elena", "Paulina", "Mónica", "Sabina", "Helena", "Laura", "Dalia",
                         "Elvira", "Female"]},
}


def perfil_voz(asiento: int) -> dict:
    """Voice profile for a seat (1 = Ottoman, 2 = the second robot). A copy."""
    if asiento not in _PERFILES:
        raise ValueError(f"Asiento sin perfil de voz: {asiento!r}")
    perfil = dict(_PERFILES[asiento])
    perfil["preferencias"] = list(perfil["preferencias"])
    return perfil


def todas_las_frases() -> list:
    """Every fixed phrase a robot can say (for tests and docs)."""
    salida = [f for opciones in _FRASES.values() for f in opciones]
    salida += [f for f in _FRASES_CARTA if f]
    salida += [numero_en_palabras(n).capitalize() + "." for n in range(34)]
    return salida
