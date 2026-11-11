"""Draws the project's own 40-card Spanish deck (plus a card back) with Pillow.

Every card is drawn from code: no template image, no font files, no scans. The
same function always produces the same pixels, so the PNGs in `cartas/` can be
regenerated on any machine:

    py -3 generar_baraja.py            # writes cartas/{valor}_{Palo}.png + dorso.png

Layout follows the traditional Spanish deck: corner index numbers, pip layouts
for 1-7, simple original figures for 10 (sota), 11 (caballo) and 12 (rey), and
the classic broken frame ("pinta") that tells the suit at a glance: oros has no
break, copas one, espadas two and bastos three.

The web shows these files at an aspect ratio of 262/400 and the MuJoCo scene
uses them as textures, so the output size keeps that ratio exactly.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageDraw

PALOS = ("Espada", "Basto", "Oro", "Copa")
VALORES = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)

# Base design units (the web's aspect ratio). Everything below is expressed in
# these units and scaled on the way to pixels.
BASE_W, BASE_H = 262, 400
ESCALA_SALIDA = 2                   # final PNG: 524 x 800
SUPERMUESTREO = 2                   # drawn at 1048 x 1600, then downsampled
TAMANO_CARTA = (BASE_W * ESCALA_SALIDA, BASE_H * ESCALA_SALIDA)
NOMBRE_DORSO = "dorso.png"

Punto = tuple[float, float]

# --- palette -------------------------------------------------------------
PAPEL = (250, 246, 234)
TINTA = (40, 28, 22)
ORO = (242, 184, 36)
ORO_OSCURO = (150, 92, 14)
ROJO = (196, 44, 38)
ROJO_OSCURO = (120, 22, 20)
AZUL = (70, 128, 206)
AZUL_OSCURO = (24, 54, 120)
VERDE = (52, 132, 70)
VERDE_OSCURO = (24, 74, 36)
MADERA = (196, 118, 48)
MADERA_OSCURA = (96, 50, 18)
PIEL = (243, 204, 164)
GRIS_ACERO = (206, 220, 236)

COLOR_PALO = {
    "Oro": ORO_OSCURO,
    "Copa": ROJO,
    "Espada": AZUL_OSCURO,
    "Basto": VERDE_OSCURO,
}
# Breaks in the frame's top and bottom edges: the traditional suit cue.
CORTES_MARCO = {"Oro": 0, "Copa": 1, "Espada": 2, "Basto": 3}


def nombre_archivo(valor: int, palo: str) -> str:
    """File name the web and the 3D scene look up for a card."""
    return f"{int(valor)}_{str(palo).capitalize()}.png"


# --- drawing surface with a local transform ------------------------------

class Lienzo:
    """A Pillow drawing in base units, with an optional local transform.

    `con(cx, cy, rot, esc)` returns a child that draws in a frame centred at
    (cx, cy), rotated `rot` degrees and scaled `esc`, so every suit symbol is
    drawn once around the origin and placed anywhere at any angle.
    """

    def __init__(self, draw: ImageDraw.ImageDraw, k: float,
                 transf: Callable[[float, float], Punto] | None = None,
                 escala: float = 1.0):
        self.draw = draw
        self.k = k
        self._t = transf or (lambda x, y: (x, y))
        self.escala = escala

    def con(self, cx: float, cy: float, rot: float = 0.0, esc: float = 1.0) -> "Lienzo":
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        padre = self._t

        def t(x: float, y: float) -> Punto:
            return padre(cx + esc * (x * c - y * s), cy + esc * (x * s + y * c))

        return Lienzo(self.draw, self.k, t, self.escala * esc)

    def _px(self, pts: Iterable[Punto]) -> list[Punto]:
        out = []
        for x, y in pts:
            tx, ty = self._t(x, y)
            out.append((tx * self.k, ty * self.k))
        return out

    def _ancho(self, ancho: float) -> int:
        return max(1, int(round(ancho * self.escala * self.k)))

    def poligono(self, pts: Sequence[Punto], relleno=None, borde=None, ancho: float = 2.0) -> None:
        px = self._px(pts)
        if relleno is not None:
            self.draw.polygon(px, fill=relleno)
        if borde is not None:
            self.draw.line(px + [px[0]], fill=borde, width=self._ancho(ancho), joint="curve")
            self._tapas(px, borde, ancho)

    def linea(self, pts: Sequence[Punto], color, ancho: float = 2.0) -> None:
        px = self._px(pts)
        self.draw.line(px, fill=color, width=self._ancho(ancho), joint="curve")
        self._tapas(px, color, ancho)

    def _tapas(self, px: list[Punto], color, ancho: float) -> None:
        r = self._ancho(ancho) / 2.0
        if r < 1.5:
            return
        for x, y in (px[0], px[-1]):
            self.draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    def elipse(self, cx: float, cy: float, rx: float, ry: float, relleno=None,
               borde=None, ancho: float = 2.0) -> None:
        self.poligono(arco(cx, cy, rx, ry, 0, 360, 56)[:-1], relleno, borde, ancho)


def arco(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float,
         n: int = 24) -> list[Punto]:
    """Points along an elliptical arc; angles in degrees, 0 = right, 90 = down."""
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]


def espejo(pts: Sequence[Punto]) -> list[Punto]:
    """Closes a left-half outline with its mirror image across x = 0."""
    return list(pts) + [(-x, y) for x, y in reversed(pts)]


# --- index digits (stroke font on a 10 x 16 grid) ------------------------

DIGITOS: dict[str, list[list[Punto]]] = {
    "0": [arco(5, 8, 4, 7, 0, 360, 40)],
    "1": [[(2.2, 3.8), (5.6, 1.0), (5.6, 15.0)], [(2.4, 15.0), (8.8, 15.0)]],
    "2": [arco(5, 5, 4, 4, 180, 400, 24) + [(1.0, 15.0), (9.2, 15.0)]],
    "3": [arco(5, 4.4, 3.8, 3.4, 200, 450, 24), arco(5, 11.4, 4.1, 3.6, 270, 520, 26)],
    "4": [[(7.0, 15.0), (7.0, 1.0), (1.0, 10.6), (9.6, 10.6)]],
    "5": [[(8.6, 1.0), (2.6, 1.0), (2.1, 7.3)] + arco(5, 10.6, 4.0, 4.4, 225, 510, 28)],
    "6": [arco(8.6, 11.2, 7.6, 10.2, 263, 180, 16), arco(5, 11.2, 4.0, 3.8, 180, 540, 40)],
    "7": [[(1.0, 1.0), (9.2, 1.0), (4.2, 15.0)]],
}


def dibujar_numero(lz: Lienzo, texto: str, x: float, y: float, alto: float, color) -> None:
    """Writes `texto` with the stroke font; (x, y) is the top-left corner."""
    esc = alto / 16.0
    avance = 0.0
    for ch in texto:
        trazo = lz.con(x + avance, y, 0, esc)
        for linea in DIGITOS[ch]:
            trazo.linea(linea, color, ancho=2.7)
        avance += 10.6 * esc


def ancho_numero(texto: str, alto: float) -> float:
    esc = alto / 16.0
    return (10.6 * len(texto) - 0.6) * esc


# --- suit symbols (drawn around the origin, about 100 units tall) --------

def simbolo_oro(lz: Lienzo, detalle: bool = False) -> None:
    lz.elipse(0, 0, 46, 46, ORO, ORO_OSCURO, 4)
    lz.elipse(0, 0, 36, 36, None, ORO_OSCURO, 2.5)
    for i in range(16 if detalle else 12):
        a = math.radians(i * (360 / (16 if detalle else 12)))
        lz.elipse(41 * math.cos(a), 41 * math.sin(a), 2.2, 2.2, ORO_OSCURO)
    for i in range(8):
        petalo = lz.con(0, 0, i * 45)
        petalo.poligono([(0, -4), (8, -14), (0, -30), (-8, -14)], ROJO, ROJO_OSCURO, 1.5)
    lz.elipse(0, 0, 9, 9, ORO, ORO_OSCURO, 2.5)
    lz.elipse(0, 0, 3.5, 3.5, ROJO_OSCURO)
    if detalle:
        lz.elipse(0, 0, 22, 22, None, ORO_OSCURO, 1.5)


def simbolo_copa(lz: Lienzo, detalle: bool = False) -> None:
    # Foot, stem and knot.
    lz.poligono(espejo([(-26, 50), (-20, 40), (-9, 34)]), ROJO, ROJO_OSCURO, 3)
    lz.poligono(espejo([(-6, 34), (-6, 10)]), ORO, ORO_OSCURO, 3)
    lz.elipse(0, 16, 12, 6, ORO, ORO_OSCURO, 3)
    # Bowl.
    cuenco = arco(0, -26, 34, 38, 180, 0, 22)
    lz.poligono(cuenco[:1] + cuenco + [(34, -26)], ROJO, ROJO_OSCURO, 3)
    lz.poligono([(-34, -26), (34, -26), (36, -36), (-36, -36)], ORO, ORO_OSCURO, 3)
    lz.linea([(-20, -10), (20, -10)], ORO, 3)
    if detalle:
        for x in (-18, 0, 18):
            lz.elipse(x, 0, 4, 4, ORO, ORO_OSCURO, 1.5)
        lz.poligono([(-30, -40), (30, -40), (24, -50), (-24, -50)], ORO, ORO_OSCURO, 3)


def simbolo_espada(lz: Lienzo, detalle: bool = False) -> None:
    # Blade (point up), cross guard, grip and pommel.
    lz.poligono([(-9, 22), (-9, -34), (0, -50), (9, -34), (9, 22)], GRIS_ACERO, AZUL_OSCURO, 3.5)
    lz.linea([(0, -40), (0, 18)], AZUL, 3)
    lz.poligono([(-24, 22), (24, 22), (24, 30), (-24, 30)], ORO, ORO_OSCURO, 3)
    lz.elipse(-27, 26, 5, 5, ORO, ORO_OSCURO, 2.5)
    lz.elipse(27, 26, 5, 5, ORO, ORO_OSCURO, 2.5)
    lz.poligono([(-5, 30), (5, 30), (5, 44), (-5, 44)], ROJO, ROJO_OSCURO, 2.5)
    lz.elipse(0, 48, 7, 6, ORO, ORO_OSCURO, 2.5)
    if detalle:
        lz.linea(arco(0, 26, 34, 18, 200, 340, 18), ORO_OSCURO, 3)
        lz.elipse(0, 26, 4, 4, ROJO)


def simbolo_basto(lz: Lienzo, detalle: bool = False) -> None:
    # A knotty wooden club, wide at the top, with two green sprouts.
    contorno = [(-13, -46), (-17, -30), (-12, -14), (-14, 4), (-9, 22), (-10, 40), (-6, 50)]
    lz.poligono(espejo(contorno), MADERA, MADERA_OSCURA, 3)
    for x, y in ((-7, -30), (6, -12), (-5, 8), (5, 28)):
        lz.elipse(x, y, 3.5, 2.6, MADERA_OSCURA)
    for lado in (-1, 1):
        hoja = lz.con(lado * 12, -24 + lado * 18, lado * 40)
        hoja.poligono([(0, 0), (7 * lado, -6), (16 * lado, -4), (9 * lado, 3)], VERDE, VERDE_OSCURO, 1.5)
    lz.linea([(-6, -44), (-3, 40)], (226, 160, 90), 2)
    if detalle:
        for y in (-38, -2, 34):
            lz.elipse(0, y, 3, 3, VERDE_OSCURO)


SIMBOLOS = {
    "Oro": simbolo_oro,
    "Copa": simbolo_copa,
    "Espada": simbolo_espada,
    "Basto": simbolo_basto,
}
# Scale of each symbol as a pip (it is drawn about 100 units tall).
ESCALA_PIP = {"Oro": 0.74, "Copa": 0.84, "Espada": 0.98, "Basto": 0.96}
ESCALA_AS = {"Oro": 1.45, "Copa": 2.0, "Espada": 2.65, "Basto": 2.55}

# --- pip layouts (centre x, centre y) inside the frame -------------------
_L, _C, _R = 86, 131, 176
DISPOSICION: dict[int, list[Punto]] = {
    2: [(_C, 112), (_C, 288)],
    3: [(_C, 96), (_C, 200), (_C, 304)],
    4: [(_L, 112), (_R, 112), (_L, 288), (_R, 288)],
    5: [(_L, 104), (_R, 104), (_C, 200), (_L, 296), (_R, 296)],
    6: [(_L, 96), (_R, 96), (_L, 200), (_R, 200), (_L, 304), (_R, 304)],
    7: [(_L, 86), (_R, 86), (_C, 150), (_L, 214), (_R, 214), (_L, 314), (_R, 314)],
}


def _angulo_pip(palo: str, valor: int, i: int, n: int) -> float:
    """Swords and clubs lean alternately, a nod to the deck's crossed pips."""
    if palo not in ("Espada", "Basto"):
        return 0.0
    if valor <= 3:
        return (-14.0, 14.0, -14.0)[i] if n > 1 else 0.0
    if valor == 7 and i == 2:
        return 0.0
    return -16.0 if i % 2 == 0 else 16.0


def dibujar_pips(lz: Lienzo, palo: str, valor: int) -> None:
    dibujar = SIMBOLOS[palo]
    if valor == 1:
        dibujar(lz.con(131, 200, 0, ESCALA_AS[palo]), detalle=True)
        return
    pts = DISPOSICION[valor]
    for i, (x, y) in enumerate(pts):
        dibujar(lz.con(x, y, _angulo_pip(palo, valor, i, len(pts)), ESCALA_PIP[palo]))


# --- court figures -------------------------------------------------------

ROPA = {
    # (tunic, trim, hose/legs, hat or cape)
    "Oro": (VERDE, ORO, ROJO, ROJO),
    "Copa": (AZUL, ORO, ROJO, ROJO),
    "Espada": (ROJO, ORO, AZUL, AZUL),
    "Basto": (ROJO, ORO, VERDE, VERDE),
}


def _cabeza(lz: Lienzo, x: float, y: float, r: float = 17) -> None:
    lz.elipse(x, y, r, r * 1.12, PIEL, TINTA, 2.5)
    lz.elipse(x - r * 0.35, y - r * 0.1, 1.8, 2.2, TINTA)
    lz.elipse(x + r * 0.35, y - r * 0.1, 1.8, 2.2, TINTA)
    lz.linea([(x - r * 0.3, y + r * 0.45), (x + r * 0.3, y + r * 0.45)], ROJO_OSCURO, 2)


def _brazo(lz: Lienzo, pts: Sequence[Punto], color, ancho: float) -> None:
    """A sleeve: a dark outline stroke with the cloth colour on top."""
    lz.linea(pts, TINTA, ancho + 4.5)
    lz.linea(pts, color, ancho)


def figura_sota(lz: Lienzo, palo: str) -> None:
    tunica, adorno, piernas, gorro = ROPA[palo]
    # Legs and shoes.
    lz.poligono([(108, 262), (126, 262), (124, 336), (110, 336)], piernas, TINTA, 2.5)
    lz.poligono([(136, 262), (154, 262), (152, 336), (138, 336)], piernas, TINTA, 2.5)
    lz.poligono([(100, 336), (126, 336), (126, 346), (98, 346)], TINTA)
    lz.poligono([(136, 336), (162, 336), (164, 346), (136, 346)], TINTA)
    # Tunic with belt and trim.
    lz.poligono([(112, 150), (150, 150), (172, 270), (90, 270)], tunica, TINTA, 3)
    lz.poligono([(102, 214), (160, 214), (162, 224), (100, 224)], adorno, TINTA, 2)
    lz.linea([(131, 152), (131, 212)], adorno, 3)
    # Arms: the left rests on the hip, the right holds the symbol up.
    _brazo(lz, [(114, 160), (96, 196), (108, 216)], tunica, 12)
    _brazo(lz, [(148, 160), (176, 150), (186, 124)], tunica, 12)
    lz.elipse(187, 120, 7, 7, PIEL, TINTA, 2)
    # Head with a feathered cap and long hair.
    lz.poligono([(112, 118), (150, 118), (154, 152), (108, 152)], (120, 70, 30), TINTA, 2)
    _cabeza(lz, 131, 124)
    lz.poligono([(108, 112), (154, 112), (148, 96), (114, 96)], gorro, TINTA, 2.5)
    lz.linea([(150, 100), (170, 80), (174, 70)], adorno, 4)
    SIMBOLOS[palo](lz.con(190, 86, 15 if palo in ("Espada", "Basto") else 0, 0.62))


def figura_caballo(lz: Lienzo, palo: str) -> None:
    tunica, adorno, piernas, gorro = ROPA[palo]
    caballo, crin = (236, 226, 206), (130, 84, 40)
    # Horse legs, body, neck and head (facing left).
    for x0, x1 in ((84, 92), (104, 110), (156, 162), (176, 184)):
        lz.poligono([(x0, 260), (x0 + 12, 260), (x1 + 8, 336), (x1 - 2, 336)], caballo, TINTA, 2.5)
        lz.poligono([(x1 - 4, 334), (x1 + 10, 334), (x1 + 10, 344), (x1 - 4, 344)], TINTA)
    lz.elipse(134, 248, 64, 34, caballo, TINTA, 3)
    lz.linea([(84, 154), (98, 180), (102, 214)], crin, 8)
    lz.poligono([(84, 236), (98, 214), (76, 158), (58, 172)], caballo, TINTA, 3)
    lz.poligono([(76, 156), (52, 168), (36, 200), (48, 208), (62, 190), (70, 182)], caballo, TINTA, 3)
    lz.elipse(66, 170, 2.5, 2.5, TINTA)
    lz.linea([(196, 236), (214, 262), (210, 300)], crin, 7)
    # Saddle cloth.
    lz.poligono([(112, 222), (168, 222), (172, 262), (108, 262)], gorro, TINTA, 2.5)
    lz.linea([(108, 258), (172, 258)], adorno, 3)
    # Rider.
    lz.poligono([(146, 216), (160, 216), (150, 270), (138, 270)], piernas, TINTA, 2)
    lz.poligono([(118, 150), (150, 150), (158, 222), (112, 222)], tunica, TINTA, 3)
    _brazo(lz, [(122, 160), (100, 182), (78, 176)], tunica, 11)
    _brazo(lz, [(148, 158), (172, 140), (180, 112)], tunica, 11)
    lz.elipse(181, 108, 6.5, 6.5, PIEL, TINTA, 2)
    _cabeza(lz, 134, 128, 15)
    lz.poligono([(114, 118), (154, 118), (146, 98), (122, 98)], gorro, TINTA, 2.5)
    lz.elipse(134, 116, 24, 4, gorro, TINTA, 2)
    SIMBOLOS[palo](lz.con(186, 76, 15 if palo in ("Espada", "Basto") else 0, 0.56))


def figura_rey(lz: Lienzo, palo: str) -> None:
    tunica, adorno, piernas, capa = ROPA[palo]
    # Long robe and cape.
    lz.poligono([(104, 150), (158, 150), (190, 336), (72, 336)], capa, TINTA, 3)
    lz.poligono([(114, 156), (148, 156), (164, 336), (98, 336)], tunica, TINTA, 2.5)
    lz.poligono([(126, 156), (136, 156), (136, 336), (126, 336)], adorno, TINTA, 1.5)
    lz.poligono([(96, 330), (166, 330), (170, 344), (92, 344)], adorno, TINTA, 2)
    # Ermine collar.
    lz.elipse(131, 152, 30, 9, (246, 242, 236), TINTA, 2)
    for x in (112, 124, 138, 150):
        lz.elipse(x, 152, 1.8, 2.6, TINTA)
    # Arms: one on the chest, one holding the symbol.
    _brazo(lz, [(110, 166), (100, 206), (124, 222)], capa, 13)
    lz.elipse(126, 222, 7, 7, PIEL, TINTA, 2)
    _brazo(lz, [(152, 166), (178, 168), (190, 142)], capa, 13)
    lz.elipse(191, 138, 7, 7, PIEL, TINTA, 2)
    # Head, beard and crown.
    _cabeza(lz, 131, 118, 17)
    lz.poligono(arco(131, 124, 17, 18, 0, 180, 14), (232, 230, 226), TINTA, 2)
    lz.linea([(124, 128), (138, 128)], ROJO_OSCURO, 2)
    lz.poligono([(112, 102), (150, 102), (154, 78), (143, 90), (131, 72), (119, 90), (108, 78)],
                ORO, ORO_OSCURO, 2.5)
    for x, y in ((108, 78), (131, 72), (154, 78)):
        lz.elipse(x, y, 3, 3, ROJO, ROJO_OSCURO, 1)
    SIMBOLOS[palo](lz.con(196, 102, 15 if palo in ("Espada", "Basto") else 0, 0.58))


FIGURAS = {10: figura_sota, 11: figura_caballo, 12: figura_rey}


# --- frame and corner indices --------------------------------------------

MARCO = (28, 36, 234, 364)          # x0, y0, x1, y1 in base units


def _segmentos_con_cortes(x0: float, x1: float, cortes: int) -> list[tuple[float, float]]:
    """Splits a horizontal edge into pieces leaving `cortes` centred gaps."""
    if cortes == 0:
        return [(x0, x1)]
    hueco, paso = 9.0, 16.0
    total = cortes * hueco + (cortes - 1) * (paso - hueco)
    inicio = (x0 + x1) / 2 - total / 2
    tramos, cursor = [], x0
    for i in range(cortes):
        a = inicio + i * paso
        tramos.append((cursor, a))
        cursor = a + hueco
    tramos.append((cursor, x1))
    return tramos


def dibujar_marco(lz: Lienzo, palo: str) -> None:
    color = COLOR_PALO[palo]
    x0, y0, x1, y1 = MARCO
    cortes = CORTES_MARCO[palo]
    lz.linea([(x0, y0), (x0, y1)], color, 2.4)
    lz.linea([(x1, y0), (x1, y1)], color, 2.4)
    for y in (y0, y1):
        for a, b in _segmentos_con_cortes(x0, x1, cortes):
            lz.linea([(a, y), (b, y)], color, 2.4)


def dibujar_indices(lz: Lienzo, palo: str, valor: int) -> None:
    texto = str(valor)
    alto = 34.0
    ancho = ancho_numero(texto, alto)
    color = COLOR_PALO[palo]
    for rot in (0, 180):
        esquina = lz.con(131, 200, rot).con(-131, -200)
        x, y = 13.0, 12.0
        esquina.poligono([(x - 5, y - 5), (x + ancho + 5, y - 5),
                          (x + ancho + 5, y + alto + 5), (x - 5, y + alto + 5)], PAPEL)
        dibujar_numero(esquina, texto, x, y, alto, color)


# --- whole cards ---------------------------------------------------------

def _nuevo_lienzo() -> tuple[Image.Image, Lienzo]:
    k = ESCALA_SALIDA * SUPERMUESTREO
    img = Image.new("RGB", (BASE_W * k, BASE_H * k), PAPEL)
    return img, Lienzo(ImageDraw.Draw(img), k)


def _terminar(img: Image.Image) -> Image.Image:
    return img.resize(TAMANO_CARTA, Image.LANCZOS)


def dibujar_carta(valor: int, palo: str) -> Image.Image:
    """One card face as an RGB image of TAMANO_CARTA pixels."""
    palo = str(palo).capitalize()
    if palo not in PALOS or int(valor) not in VALORES:
        raise ValueError(f"No existe la carta {valor} de {palo}")
    img, lz = _nuevo_lienzo()
    lz.poligono([(3, 3), (259, 3), (259, 397), (3, 397)], None, (200, 190, 168), 2)
    dibujar_marco(lz, palo)
    if valor in FIGURAS:
        FIGURAS[valor](lz, palo)
    else:
        dibujar_pips(lz, palo, valor)
    dibujar_indices(lz, palo, valor)
    return _terminar(img)


def dibujar_dorso() -> Image.Image:
    """The project's own card back, matching the web's navy and gold back."""
    img, lz = _nuevo_lienzo()
    fondo, linea, oro = (22, 48, 99), (60, 78, 110), (255, 204, 41)
    lz.poligono([(0, 0), (262, 0), (262, 400), (0, 400)], (4, 11, 28))
    lz.poligono([(12, 12), (250, 12), (250, 388), (12, 388)], fondo)
    for i in range(-40, 70):
        d = i * 12.0
        lz.linea([(12 + d, 12), (12 + d + 376, 388)], linea, 1.6)
        lz.linea([(250 - d, 12), (250 - d - 376, 388)], linea, 1.6)
    # Re-paint the border over the lattice overflow.
    for caja in ([(0, 0), (262, 0), (262, 12), (0, 12)], [(0, 388), (262, 388), (262, 400), (0, 400)],
                 [(0, 0), (12, 0), (12, 400), (0, 400)], [(250, 0), (262, 0), (262, 400), (250, 400)]):
        lz.poligono(caja, (4, 11, 28))
    lz.poligono([(20, 20), (242, 20), (242, 380), (20, 380)], None, (224, 168, 0), 2.4)
    rombo = lz.con(131, 200, 45)
    rombo.poligono([(-46, -46), (46, -46), (46, 46), (-46, 46)], (8, 20, 46), oro, 4)
    rombo.poligono([(-36, -36), (36, -36), (36, 36), (-36, 36)], None, (224, 168, 0), 2)
    simbolo_oro(lz.con(131, 200, 0, 0.62))
    return _terminar(img)


def generar_baraja_40(dir_salida: str) -> dict[str, str]:
    """Writes the 40 faces and the back into `dir_salida`; returns name -> path."""
    os.makedirs(dir_salida, exist_ok=True)
    generadas: dict[str, str] = {}
    for palo in PALOS:
        for valor in VALORES:
            nombre = nombre_archivo(valor, palo)
            ruta = os.path.join(dir_salida, nombre)
            dibujar_carta(valor, palo).save(ruta, optimize=True)
            generadas[nombre] = ruta
    ruta = os.path.join(dir_salida, NOMBRE_DORSO)
    dibujar_dorso().save(ruta, optimize=True)
    generadas[NOMBRE_DORSO] = ruta
    return generadas


def hoja_de_contactos(dir_cartas: str, destino: str, ancho_carta: int = 131) -> str:
    """A grid of all 40 faces plus the back, for reviewing the deck at a glance."""
    alto = round(ancho_carta * BASE_H / BASE_W)
    sep = 8
    filas = [[nombre_archivo(v, p) for v in VALORES] for p in PALOS] + [[NOMBRE_DORSO]]
    hoja = Image.new("RGB", (len(VALORES) * (ancho_carta + sep) + sep,
                             len(filas) * (alto + sep) + sep), (30, 60, 40))
    for f, fila in enumerate(filas):
        for c, nombre in enumerate(fila):
            img = Image.open(os.path.join(dir_cartas, nombre)).convert("RGB")
            hoja.paste(img.resize((ancho_carta, alto), Image.LANCZOS),
                       (sep + c * (ancho_carta + sep), sep + f * (alto + sep)))
    hoja.save(destino, optimize=True)
    return destino


if __name__ == "__main__":
    import sys

    dir_cartas = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartas")
    generadas = generar_baraja_40(dir_cartas)
    print(f"[OK] {len(generadas)} imagenes en {dir_cartas}")
    if len(sys.argv) > 1:
        print(f"[OK] Hoja de contactos: {hoja_de_contactos(dir_cartas, sys.argv[1])}")
