"""Reglas de Truco Argentino: jerarquía de cartas, cálculo de envido y utilidades.

Soporta Truco Argentino Tradicional (sin Flor, a 15 o 30 puntos) y reconocimiento
de nombres de cartas en lenguaje natural.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

PALOS = ("Espada", "Basto", "Oro", "Copa")
VALORES = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)

ICONOS_PALO = {
    "Espada": "⚔️",
    "Basto": "🌿",
    "Oro": "🪙",
    "Copa": "🏆",
}


def valor_carta_truco(valor: int, palo: str) -> int:
    """Devuelve el poder de la carta en el Truco tradicional (0 a 13)."""
    palo = str(palo).capitalize()
    v = int(valor)
    if v == 1 and palo == "Espada":
        return 13
    if v == 1 and palo == "Basto":
        return 12
    if v == 7 and palo == "Espada":
        return 11
    if v == 7 and palo == "Oro":
        return 10
    if v == 3:
        return 9
    if v == 2:
        return 8
    if v == 1 and palo in ("Oro", "Copa"):
        return 7
    if v == 12:
        return 6
    if v == 11:
        return 5
    if v == 10:
        return 4
    if v == 7 and palo in ("Basto", "Copa"):
        return 3
    if v == 6:
        return 2
    if v == 5:
        return 1
    return 0


def nombre_carta(valor: int, palo: str) -> str:
    """Nombre descriptivo criollo de la carta."""
    palo = str(palo).capitalize()
    v = int(valor)
    if v == 1 and palo == "Espada":
        return "Ancho de Espada"
    if v == 1 and palo == "Basto":
        return "Ancho de Basto"
    if v == 7 and palo == "Espada":
        return "7 de Espada"
    if v == 7 and palo == "Oro":
        return "7 de Oro"
    if v == 1 and palo in ("Oro", "Copa"):
        return f"Ancho Falso (1 de {palo})"
    return f"{v} de {palo}"


def calcular_envido(cartas: List[Dict]) -> int:
    """Calcula los puntos de envido para una mano (0 a 33)."""
    def val_e(v: int) -> int:
        return int(v) if int(v) < 10 else 0

    if not cartas:
        return 0
    if len(cartas) == 1:
        return val_e(cartas[0]["valor"])

    por_palo: Dict[str, List[int]] = {}
    for c in cartas:
        p = str(c["palo"]).capitalize()
        v = val_e(c["valor"])
        por_palo.setdefault(p, []).append(v)

    max_puntos = 0
    for vals in por_palo.values():
        if len(vals) >= 2:
            vals_ord = sorted(vals, reverse=True)
            pts = 20 + vals_ord[0] + vals_ord[1]
            if pts > max_puntos:
                max_puntos = pts

    if max_puntos > 0:
        return max_puntos

    return max(val_e(c["valor"]) for c in cartas)


def poder_mano_truco(cartas: List[Dict]) -> float:
    """Puntaje ponderado de fuerza para Truco (0 a ~24.7)."""
    poderes = sorted(
        [valor_carta_truco(c["valor"], c["palo"]) for c in cartas],
        reverse=True,
    )
    if len(poderes) == 3:
        return round(poderes[0] * 1.0 + poderes[1] * 0.6 + poderes[2] * 0.3, 2)
    if len(poderes) == 2:
        return round(poderes[0] * 1.0 + poderes[1] * 0.6, 2)
    if len(poderes) == 1:
        return round(poderes[0] * 1.0, 2)
    return 0.0


def puntos_falta_envido(puntos_humano: int, puntos_robot: int, meta_puntos: int = 15) -> int:
    """Calcula cuántos puntos otorga ganar la Falta Envido según el estado del partido."""
    lider = max(puntos_humano, puntos_robot)
    if meta_puntos == 30 and lider < 15:
        # En malas de partido a 30, la falta envido gana el partido (o lo que le falta al líder para 30)
        return max(1, 30 - lider)
    return max(1, meta_puntos - lider)


def buscar_carta_en_texto(texto: str, cartas_disponibles: List[Dict]) -> Optional[Dict]:
    """Intenta identificar una carta de `cartas_disponibles` nombrada por voz."""
    if not texto or not cartas_disponibles:
        return None
    t = texto.lower().strip()

    # Mapeo de números hablados
    mapa_num = {
        "ancho": 1, "as": 1, "uno": 1, "1": 1,
        "dos": 2, "2": 2,
        "tres": 3, "3": 3,
        "cuatro": 4, "4": 4,
        "cinco": 5, "5": 5,
        "seis": 6, "6": 6,
        "siete": 7, "bravo": 7, "7": 7,
        "diez": 10, "sota": 10, "10": 10,
        "once": 11, "caballo": 11, "11": 11,
        "doce": 12, "rey": 12, "12": 12,
    }
    mapa_palo = {
        "espada": "Espada", "espadas": "Espada", "macho": "Espada",
        "basto": "Basto", "bastos": "Basto", "hembra": "Basto",
        "oro": "Oro", "oros": "Oro",
        "copa": "Copa", "copas": "Copa",
    }

    valor_buscado = None
    for palabra, num in mapa_num.items():
        if re.search(r"\b" + re.escape(palabra) + r"\b", t):
            valor_buscado = num
            break

    palo_buscado = None
    for palabra, palo in mapa_palo.items():
        if re.search(r"\b" + re.escape(palabra) + r"\b", t):
            palo_buscado = palo
            break

    if "macho" in t and valor_buscado is None:
        valor_buscado, palo_buscado = 1, "Espada"
    elif "hembra" in t and valor_buscado is None:
        valor_buscado, palo_buscado = 1, "Basto"

    for c in cartas_disponibles:
        v_ok = valor_buscado is None or int(c["valor"]) == valor_buscado
        p_ok = palo_buscado is None or str(c["palo"]).capitalize() == palo_buscado
        if v_ok and p_ok and (valor_buscado is not None or palo_buscado is not None):
            return c

    return None
