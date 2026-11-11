"""Offscreen screenshots of the Truco table, one per camera, at key moments.

    cd entorno && py -3 -m sim.capturas ../docs/screenshots/step1 jpg
    cd entorno && py -3 -m sim.capturas ../docs/screenshots/step2 jpg robots

Runs the same pose code as the live simulator on a simulated clock, so what
these images show is what the viewer shows. Used to review the cards in 3D
(scale, clipping, orientation, readability) without opening a window.
"""

from __future__ import annotations

import os
import sys

from . import cartas as C


def _guardar(px, ruta: str) -> None:
    from PIL import Image

    img = Image.fromarray(px)
    if ruta.lower().endswith((".jpg", ".jpeg")):
        img.save(ruta, quality=84, optimize=True)
    else:
        img.save(ruta, optimize=True)


class Fotografo:
    def __init__(self, ancho: int = 960, alto: int = 540):
        import mujoco

        from .arrancar import SimuladorOficial
        from .mundo import Mundo
        from .robots import G1

        self.mj = mujoco
        self.t = 0.0
        repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unitree_mujoco")
        self.sim = SimuladorOficial(Mundo(), G1, repo, verboso=False)
        self.sim.reloj = lambda: self.t
        self.mesa = self.sim.cartas.mesa
        self.renderer = mujoco.Renderer(self.sim.model, alto, ancho)

    def correr(self, segundos: float, fps: int = 50) -> None:
        for _ in range(max(1, round(segundos * fps))):
            self.t += 1.0 / fps
            self.sim._escribir_pose()

    def foto(self, camara: str):
        self.renderer.update_scene(self.sim.data, camera=camara)
        return self.renderer.render()


def capturar(destino: str, camaras=(C.CAMARA_GENERAL, C.CAMARA_DETRAS[1], C.CAMARA_DETRAS[2]), ancho: int = 960, alto: int = 540,
             extension: str = "png") -> list[str]:
    os.makedirs(destino, exist_ok=True)
    f = Fotografo(ancho, alto)
    rutas: list[str] = []

    def momento(nombre: str) -> None:
        for cam in camaras:
            ruta = os.path.join(destino, f"{nombre}_{cam}.{extension}")
            _guardar(f.foto(cam), ruta)
            rutas.append(ruta)

    # Robot vs robot view rules, so the behind-robot camera shows the faces.
    f.mesa.poner_modo(C.MODO_ROBOTS)
    robot = [C.Carta(1, "Espada"), C.Carta(7, "Oro"), C.Carta(12, "Copa")]
    humano = [C.Carta(3, "Basto"), C.Carta(1, "Basto"), C.Carta(10, "Espada")]
    f.mesa.repartir({1: robot})
    f.correr(1.5)
    momento("1_cartas_en_mano")

    f.mesa.jugar(2, humano[0])
    f.mesa.jugar(1, robot[1])
    r = f.mesa.ritmo
    f.correr(r.alcanzar * 0.95)
    momento("2_alcanza_la_carta")
    f.correr(r.alcanzar * 0.05 + r.llevar * 0.5)
    momento("3_lleva_la_carta")
    f.correr(r.llevar * 0.5 + r.volver + 0.5)
    momento("4_carta_en_la_mesa")

    for k in (1, 2):
        f.mesa.jugar(1, robot[[0, 2][k - 1]])
        f.mesa.jugar(2, humano[k])
        f.correr(r.alcanzar + r.llevar + r.volver + 0.6)
    momento("5_tres_bazas")

    # Human vs robot: the only camera allowed is the overview, and from there
    # the robot must visibly hold three cards (backs toward the human).
    f.mesa.poner_modo(C.MODO_HUMANO)
    f.mesa.repartir({1: robot})
    f.correr(1.5)
    ruta = os.path.join(destino, f"6_modo_humano_{C.CAMARA_GENERAL}.{extension}")
    _guardar(f.foto(C.CAMARA_GENERAL), ruta)
    rutas.append(ruta)
    return rutas


def capturar_robots(destino: str, camaras=C.CAMARAS, ancho: int = 960, alto: int = 540,
                    extension: str = "png") -> list[str]:
    """Spectator mode: both G1s hold their cards and play them with the arm."""
    os.makedirs(destino, exist_ok=True)
    f = Fotografo(ancho, alto)
    rutas: list[str] = []

    def momento(nombre: str, solo=None) -> None:
        for cam in solo or camaras:
            ruta = os.path.join(destino, f"{nombre}_{cam}.{extension}")
            _guardar(f.foto(cam), ruta)
            rutas.append(ruta)

    f.mesa.poner_modo(C.MODO_ROBOTS)
    uno = [C.Carta(1, "Espada"), C.Carta(7, "Oro"), C.Carta(12, "Copa")]
    dos = [C.Carta(3, "Basto"), C.Carta(1, "Basto"), C.Carta(10, "Espada")]
    f.mesa.repartir({1: uno, 2: dos})
    f.correr(1.5)
    momento("1_dos_robots_con_cartas")

    r = f.mesa.ritmo
    f.mesa.jugar(1, uno[1])
    f.correr(r.alcanzar + r.llevar * 0.5)
    momento("2_robot_1_lleva_la_carta", (C.CAMARA_MESA, C.CAMARA_DETRAS[1]))
    f.correr(r.llevar * 0.5 + r.volver + 0.3)
    f.mesa.jugar(2, dos[0])
    f.correr(r.alcanzar * 0.95)
    momento("3_robot_2_alcanza_la_carta", (C.CAMARA_MESA, C.CAMARA_DETRAS[2]))
    f.correr(r.alcanzar * 0.05 + r.llevar * 0.5)
    momento("4_robot_2_lleva_la_carta", (C.CAMARA_MESA, C.CAMARA_DETRAS[2]))
    f.correr(r.llevar * 0.5 + r.volver + 0.5)

    for k in (1, 2):
        f.mesa.jugar(2, dos[k])
        f.mesa.jugar(1, uno[[0, 2][k - 1]])
        f.correr(r.alcanzar + r.llevar + r.volver + 0.6)
    momento("5_tres_bazas")

    # Back to a human: the rival G1 leaves the table.
    f.mesa.poner_modo(C.MODO_HUMANO)
    f.mesa.repartir({1: uno})
    f.correr(1.0)
    momento("6_modo_humano_sin_rival", (C.CAMARA_GENERAL,))
    return rutas


if __name__ == "__main__":
    salida = sys.argv[1] if len(sys.argv) > 1 else "capturas"
    extension = sys.argv[2] if len(sys.argv) > 2 else "png"
    hacer = capturar_robots if len(sys.argv) > 3 and sys.argv[3] == "robots" else capturar
    for ruta in hacer(salida, extension=extension):
        print(ruta)
