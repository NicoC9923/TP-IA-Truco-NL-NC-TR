"""Truco cards in the 3D scene: layout, scene XML and the play-card state machine.

This module never imports MuJoCo, so the game logic of the cards can be tested
without it. `escena_cartas.py` takes what is decided here and applies it to a
loaded model (mocap poses, materials, arm joints, cameras).

Seats
-----
The table sits between two seats on the x axis:

    seat 1  (x = 0)      Ottoman, the robot already in the scene, facing +x
    seat 2  (x = 1.20)   the rival, facing -x: the human (no body), or in
                         robot-vs-robot mode a second G1 in a light-blue poncho

Each seat owns three card bodies. A robot seat holds its fan with both hands
(carried by the left palm); the human seat has no body, so its cards only appear when played.

Playing a card is kinematic, never physical grasping. For a robot seat:

    alcanzar  the right hand reaches the chosen card in the fan
    llevar    the card rides on the right hand and is laid face up on the table
    volver    the right hand goes back to rest

A human card slides in from the human's edge of the table to its slot.
"""

from __future__ import annotations

import math
import os
import threading
from collections import deque
from dataclasses import dataclass, field

PALOS = ("Espada", "Basto", "Oro", "Copa")
VALORES = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)

ASIENTOS = (1, 2)
CARTAS_POR_ASIENTO = 3

# --- geometry (metres) ---------------------------------------------------
# A bit larger than a real deck (61 x 93 mm) so they read on screen; the
# 262/400 ratio of the PNGs is kept.
CARTA_ANCHO, CARTA_ALTO = 0.088, 0.1344
CARTA_GROSOR = 0.0012
MESA_X = 0.60                                  # table centre
MESA_Z = 0.671                                 # top of the felt
ASIENTO_X = {1: 0.0, 2: 2 * MESA_X}            # where each seat's pelvis sits
# Seat 1 faces +x and seat 2 faces -x.
ASIENTO_SENTIDO = {1: 1.0, 2: -1.0}
# Bazas left to right as seen by seat 1; both seats share the same y so each
# baza's two cards face each other across the table.
SLOT_Y = (0.125, 0.0, -0.125)
SLOT_DISTANCIA = 0.15                         # from table centre toward the owner
SLOT_GIRO = (0.05, -0.03, 0.04)                # small hand-placed look, radians

# --- cameras -------------------------------------------------------------
CAMARA_GENERAL = "general"
CAMARA_DETRAS = {1: "detras_robot_1", 2: "detras_robot_2"}
CAMARA_MESA = "mesa"          # side view of the whole table, both seats in frame
CAMARAS = (CAMARA_GENERAL, CAMARA_DETRAS[1], CAMARA_DETRAS[2], CAMARA_MESA)
MODO_HUMANO = "humano"        # human vs robot: no camera may show a robot's hand
MODO_ROBOTS = "robots"        # robot vs robot (spectator mode): every camera allowed
MODOS = (MODO_HUMANO, MODO_ROBOTS)


@dataclass(frozen=True)
class Carta:
    valor: int
    palo: str

    @staticmethod
    def de(dato) -> "Carta":
        """Builds a card from a dict ({'valor', 'palo'}), a tuple or a Carta."""
        if isinstance(dato, Carta):
            return dato
        if isinstance(dato, dict):
            valor, palo = dato.get("valor"), dato.get("palo")
        else:
            valor, palo = dato
        carta = Carta(int(valor), str(palo).capitalize())
        if carta.palo not in PALOS or carta.valor not in VALORES:
            raise ValueError(f"Carta invalida: {dato!r}")
        return carta

    def como_dict(self) -> dict:
        return {"valor": self.valor, "palo": self.palo}


def nombre_textura(carta: Carta) -> str:
    """Card -> PNG file name in proyecto/cartas (the same names the web uses)."""
    return f"{carta.valor}_{carta.palo}.png"


def nombre_material(carta: Carta | None) -> str:
    """Card -> MuJoCo material name; None means the card back."""
    return "mat_dorso" if carta is None else f"mat_{carta.valor}_{carta.palo}"


def todas_las_cartas() -> list[Carta]:
    return [Carta(v, p) for p in PALOS for v in VALORES]


def cuerpo(asiento: int, indice: int) -> str:
    return f"carta_{asiento}_{indice}"


def slot_mesa(asiento: int, baza: int) -> tuple[tuple[float, float, float], float]:
    """Where seat `asiento` lays its card of baza `baza` (0-2): position and yaw.

    The card lies face up with its top pointing away from its owner, so each
    player reads their own cards upright. Later bazas sit a hair higher, so
    overlapping corners never z-fight.
    """
    s = ASIENTO_SENTIDO[asiento]
    x = MESA_X - s * SLOT_DISTANCIA
    z = MESA_Z + CARTA_GROSOR / 2 + 0.0004 + 0.0006 * baza
    yaw = (math.pi / 2 if s > 0 else -math.pi / 2) + SLOT_GIRO[baza] * s
    return (x, SLOT_Y[baza], z), yaw


def borde_humano(asiento: int) -> tuple[float, float, float]:
    """Where a human's card appears before sliding to its slot."""
    s = ASIENTO_SENTIDO[asiento]
    return (MESA_X - s * 0.30, 0.0, MESA_Z + 0.06)


# --- cameras -------------------------------------------------------------

def _mirar(desde, hacia, arriba=(0.0, 0.0, 1.0)) -> str:
    """MuJoCo `xyaxes` for a camera at `desde` looking at `hacia`."""
    f = [h - d for h, d in zip(hacia, desde)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    x = [f[1] * arriba[2] - f[2] * arriba[1],
         f[2] * arriba[0] - f[0] * arriba[2],
         f[0] * arriba[1] - f[1] * arriba[0]]
    n = math.sqrt(sum(c * c for c in x))
    x = [c / n for c in x]
    y = [x[1] * f[2] - x[2] * f[1], x[2] * f[0] - x[0] * f[2], x[0] * f[1] - x[1] * f[0]]
    return " ".join(f"{c:.4f}" for c in x + y)


def _camara_detras(asiento: int) -> tuple[tuple, tuple]:
    s, x0 = ASIENTO_SENTIDO[asiento], ASIENTO_X[asiento]
    # Over the left shoulder, high enough to clear the hat's brim.
    desde = (x0 - s * 0.28, s * 0.44, 1.40)
    hacia = (x0 + s * 0.30, s * 0.06, 0.86)
    return desde, hacia


POSICION_CAMARAS = {
    CAMARA_GENERAL: ((MESA_X + 0.10, -0.95, 1.45), (MESA_X - 0.16, 0.0, 0.80)),
    CAMARA_DETRAS[1]: _camara_detras(1),
    CAMARA_DETRAS[2]: _camara_detras(2),
    CAMARA_MESA: ((MESA_X, -1.42, 1.78), (MESA_X, 0.04, 0.70)),
}


def camaras_permitidas(modo: str) -> tuple[str, ...]:
    """Cameras a viewer may show. Human mode hides every behind-robot camera."""
    if modo == MODO_ROBOTS:
        return CAMARAS
    return (CAMARA_GENERAL,)


def camara_valida(nombre: str, modo: str) -> str:
    """The camera to actually show: `nombre` if allowed, else the overview."""
    return nombre if nombre in camaras_permitidas(modo) else CAMARA_GENERAL


# F7 in the MuJoCo viewer cycles through the allowed cameras (GLFW key code).
# MuJoCo's own [ and ] also cycle cameras; the sim loop sends a forbidden one
# back to the overview. Esc returns to the free camera, as usual.
TECLA_CAMARA = 296


def camara_siguiente(actual: str, modo: str) -> str:
    permitidas = camaras_permitidas(modo)
    if actual not in permitidas:
        return permitidas[0]
    return permitidas[(permitidas.index(actual) + 1) % len(permitidas)]


# --- scene XML -----------------------------------------------------------

def ruta_cartas_relativa(dir_escena: str, dir_cartas: str) -> str:
    return os.path.relpath(dir_cartas, dir_escena).replace(os.sep, "/")


def _malla_carta() -> str:
    """A thin slab whose top face carries the whole texture (v = 0 at the top)."""
    w, h, t = CARTA_ANCHO / 2, CARTA_ALTO / 2, CARTA_GROSOR / 4
    vertices = [(-w, -h, t), (w, -h, t), (w, h, t), (-w, h, t),
                (-w, -h, -t), (w, -h, -t), (w, h, -t), (-w, h, -t)]
    uv = [(0, 1), (1, 1), (1, 0), (0, 0)] * 2
    caras = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
             1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0]
    v = " ".join(f"{c:.5f}" for p in vertices for c in p)
    tc = " ".join(f"{c:g}" for p in uv for c in p)
    f = " ".join(str(i) for i in caras)
    return f'    <mesh name="malla_carta" vertex="{v}" texcoord="{tc}" face="{f}"/>'


def xml_assets(ruta_cartas: str) -> str:
    """Textures and materials for the 40 faces and the back, plus the card mesh."""
    lineas = [_malla_carta()]
    material = 'specular="0.05" shininess="0.1" reflectance="0"'
    for carta in todas_las_cartas():
        archivo = nombre_textura(carta)
        tex = f"tx_{carta.valor}_{carta.palo}"
        lineas.append(f'    <texture name="{tex}" type="2d" file="{ruta_cartas}/{archivo}"/>')
        lineas.append(f'    <material name="{nombre_material(carta)}" texture="{tex}" {material}/>')
    lineas.append(f'    <texture name="tx_dorso" type="2d" file="{ruta_cartas}/dorso.png"/>')
    lineas.append(f'    <material name="mat_dorso" texture="tx_dorso" {material}/>')
    lineas.append('    <material name="mat_canto" rgba="0.93 0.91 0.85 1"/>')
    return "\n".join(lineas)


def xml_cuerpos() -> str:
    """Six mocap card bodies (three per seat), parked out of sight until dealt."""
    t = CARTA_GROSOR / 4
    lineas = []
    for asiento in ASIENTOS:
        for i in range(CARTAS_POR_ASIENTO):
            n = cuerpo(asiento, i)
            lineas += [
                f'    <body name="{n}" mocap="true" pos="{MESA_X} {0.5 + 0.1 * i} -1">',
                f'      <geom name="{n}_frente" type="mesh" mesh="malla_carta" pos="0 0 {t:.5f}"'
                f' material="mat_dorso" contype="0" conaffinity="0" group="2"/>',
                f'      <geom name="{n}_dorso" type="mesh" mesh="malla_carta" pos="0 0 {-t:.5f}"'
                f' euler="0 3.14159 0" material="mat_dorso" contype="0" conaffinity="0" group="2"/>',
                "    </body>",
            ]
    return "\n".join(lineas)


def xml_camaras() -> str:
    lineas = []
    for nombre in CAMARAS:
        desde, hacia = POSICION_CAMARAS[nombre]
        pos = " ".join(f"{c:.3f}" for c in desde)
        lineas.append(f'    <camera name="{nombre}" pos="{pos}" xyaxes="{_mirar(desde, hacia)}" fovy="50"/>')
    return "\n".join(lineas)


# --- play-card state machine ---------------------------------------------

@dataclass
class Ritmo:
    """Durations in seconds. `llevar` is the place motion (step 3 tunes pacing)."""
    alcanzar: float = 0.45
    llevar: float = 0.8
    volver: float = 0.45
    deslizar: float = 0.8          # a human card sliding to its slot


@dataclass
class CartaEnEscena:
    """What one card body shows and where it is, as the scene must draw it."""
    cuerpo: str
    asiento: int
    carta: Carta | None = None
    lugar: str = "oculta"           # oculta | mano | viaje | mesa
    indice_abanico: int = 0         # position in the fan, 0 = leftmost
    cartas_en_abanico: int = 0
    baza: int = -1                  # table slot once played
    fase: str = ""                  # alcanzar | llevar | volver | deslizar while moving
    progreso: float = 0.0           # 0..1 inside the phase


@dataclass
class _Jugada:
    asiento: int
    carta: Carta
    indice: int                     # card body index
    baza: int
    fases: list = field(default_factory=list)
    t: float = 0.0


class MesaCartas:
    """Which card is where, and the animation that moves it. Thread-safe.

    The sim loop calls `avanzar(dt)` every frame; the socket thread calls
    `repartir` and `jugar`. Plays queue per seat, so a robot that answers right
    after the human never cuts the human's card short.
    """

    def __init__(self, ritmo: Ritmo | None = None, asientos_robot=(1,)):
        """`asientos_robot`: seats with a robot body that holds its cards.

        Seat 2 joins when the mode is robot vs robot (`poner_modo`).
        """
        self.ritmo = ritmo or Ritmo()
        self.asientos_robot = set(asientos_robot)
        self._lock = threading.RLock()
        self.modo = MODO_HUMANO
        self._camara = CAMARA_GENERAL
        self.cartas: dict[str, CartaEnEscena] = {}
        self._activa: dict[int, _Jugada | None] = {a: None for a in ASIENTOS}
        self._cola: dict[int, deque] = {a: deque() for a in ASIENTOS}
        self.repartir({})

    # ---- orders ----
    def repartir(self, manos: dict) -> None:
        """Starts a new hand: clears the table and gives each robot seat its cards.

        `manos` maps seat -> list of cards. A human seat's cards are ignored:
        the scene only learns them when they are played.
        """
        with self._lock:
            self._activa = {a: None for a in ASIENTOS}
            self._cola = {a: deque() for a in ASIENTOS}
            self.cartas = {}
            for asiento in ASIENTOS:
                mano = [Carta.de(c) for c in manos.get(asiento, manos.get(str(asiento), []))]
                if asiento not in self.asientos_robot:
                    mano = []
                for i in range(CARTAS_POR_ASIENTO):
                    c = CartaEnEscena(cuerpo(asiento, i), asiento)
                    if i < len(mano):
                        c.carta, c.lugar = mano[i], "mano"
                    self.cartas[c.cuerpo] = c
                self._ordenar_abanico(asiento)

    def jugar(self, asiento: int, carta) -> None:
        """Queues `carta` to be played by `asiento` on its next free table slot."""
        carta = Carta.de(carta)
        asiento = int(asiento)
        if asiento not in ASIENTOS:
            raise ValueError(f"Asiento invalido: {asiento}")
        with self._lock:
            self._cola[asiento].append(carta)
            if self._activa[asiento] is None:
                self._empezar_siguiente(asiento)

    def poner_modo(self, modo: str) -> None:
        if modo not in MODOS:
            raise ValueError(f"Modo invalido: {modo!r}. Validos: {', '.join(MODOS)}")
        with self._lock:
            self.modo = modo
            # The second G1 only sits at the table against another robot.
            if modo == MODO_ROBOTS:
                self.asientos_robot.add(2)
            else:
                self.asientos_robot.discard(2)

    def elegir_camara(self, nombre: str) -> str:
        """Asks for a camera; returns the one that will really be shown."""
        if nombre not in CAMARAS:
            raise ValueError(f"Camara desconocida: {nombre!r}. Validas: {', '.join(CAMARAS)}")
        with self._lock:
            # A forbidden request is not remembered: it never shows up later.
            self._camara = camara_valida(nombre, self.modo)
            return self._camara

    @property
    def camara(self) -> str:
        with self._lock:
            return camara_valida(self._camara, self.modo)

    def ocultar_caras_en_mano(self, asiento: int) -> bool:
        """In human mode the robot's hand never shows faces, whatever the camera."""
        return self.modo == MODO_HUMANO and asiento in self.asientos_robot

    # ---- time ----
    def avanzar(self, dt: float) -> None:
        with self._lock:
            for asiento in ASIENTOS:
                restante = dt
                while restante > 0 and self._activa[asiento] is not None:
                    restante = self._avanzar_jugada(self._activa[asiento], restante)

    def ocupado(self) -> bool:
        with self._lock:
            return any(self._activa[a] is not None or self._cola[a] for a in ASIENTOS)

    def estado(self) -> list[CartaEnEscena]:
        """A snapshot of every card body (copies; safe to read without the lock)."""
        with self._lock:
            return [CartaEnEscena(**vars(c)) for c in self.cartas.values()]

    def jugada_activa(self, asiento: int) -> tuple[str, float, int, int] | None:
        """(phase, progress, card index, baza) of the seat's current play, if any."""
        with self._lock:
            j = self._activa.get(asiento)
            if j is None or not j.fases:
                return None
            fase, dur = j.fases[0]
            return fase, (min(1.0, j.t / dur) if dur > 0 else 1.0), j.indice, j.baza

    # ---- internals ----
    def _de_asiento(self, asiento: int) -> list[CartaEnEscena]:
        return [self.cartas[cuerpo(asiento, i)] for i in range(CARTAS_POR_ASIENTO)]

    def _ordenar_abanico(self, asiento: int) -> None:
        en_mano = [c for c in self._de_asiento(asiento) if c.lugar == "mano"]
        for k, c in enumerate(en_mano):
            c.indice_abanico, c.cartas_en_abanico = k, len(en_mano)

    def _empezar_siguiente(self, asiento: int) -> None:
        if not self._cola[asiento]:
            self._activa[asiento] = None
            return
        carta = self._cola[asiento].popleft()
        propias = self._de_asiento(asiento)
        baza = sum(1 for c in propias if c.lugar in ("mesa", "viaje"))
        en_mano = next((c for c in propias if c.lugar == "mano" and c.carta == carta), None)
        if baza >= CARTAS_POR_ASIENTO:
            self._empezar_siguiente(asiento)         # a fourth card: nothing to show
            return
        if en_mano is not None and asiento in self.asientos_robot:
            destino = en_mano
            fases = [("alcanzar", self.ritmo.alcanzar), ("llevar", self.ritmo.llevar),
                     ("volver", self.ritmo.volver)]
        else:
            # Human card, or a card the robot was never shown holding.
            destino = next((c for c in propias if c.lugar in ("oculta",)), None)
            if destino is None:
                destino = next(c for c in propias if c.lugar == "mano")
            destino.carta = carta
            fases = [("deslizar", self.ritmo.deslizar)]
        indice = int(destino.cuerpo.rsplit("_", 1)[1])
        destino.baza = baza
        jugada = _Jugada(asiento, carta, indice, baza, fases)
        self._activa[asiento] = jugada
        self._entrar_fase(jugada)

    def _entrar_fase(self, j: _Jugada) -> None:
        c = self.cartas[cuerpo(j.asiento, j.indice)]
        fase = j.fases[0][0]
        c.fase, c.progreso = fase, 0.0
        if fase in ("llevar", "deslizar"):
            c.lugar = "viaje"
            self._ordenar_abanico(j.asiento)

    def _avanzar_jugada(self, j: _Jugada, dt: float) -> float:
        fase, dur = j.fases[0]
        j.t += dt
        c = self.cartas[cuerpo(j.asiento, j.indice)]
        if j.t < dur:
            c.progreso = j.t / dur
            return 0.0
        sobra = j.t - dur
        j.t = 0.0
        j.fases.pop(0)
        if fase in ("llevar", "deslizar"):
            c.lugar, c.fase, c.progreso = "mesa", "", 1.0
        if j.fases:
            self._entrar_fase(j)
        else:
            c.fase = ""
            self._empezar_siguiente(j.asiento)
        return sobra
