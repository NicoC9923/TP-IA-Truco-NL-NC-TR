"""El simulador por defecto: MuJoCo + socket local, SIN DDS.

Es el mismo simulador -- el modelo OFICIAL de Unitree, la misma cinematica y el
mismo visor -- con el transporte cambiado. Hereda de `SimuladorOficial` y solo
reemplaza dos cosas:

    iniciar_transporte()      levanta el socket en vez del bridge DDS
    _publicar_telemetria()    arma un diccionario con el estado de MuJoCo

Por que sin DDS: ver la cabecera de `local.py`. En dos lineas -- CycloneDDS no
tiene wheels para Python 3.11+, y el SDK de Unitree solo corre en Linux.
"""

from __future__ import annotations

from .arrancar import SimuladorOficial
from .local import ServidorLocal


class SimuladorLocal(SimuladorOficial):
    """El simulador por socket. No importa nada de `unitree_sdk2py`."""

    def __init__(self, mundo, robot, repo: str, verboso: bool = True,
                 puerto: int | None = None):
        # domain e interfaz no se usan en este modo, pero el padre arma con
        # ellos su `cfg` (que ademas trae SIMULATE_DT y VIEWER_DT).
        super().__init__(mundo, robot, repo, domain=0, interfaz="",
                         verboso=verboso)
        self.puerto = puerto
        self.servidor = None
        self._ultima_telemetria: dict = {}

    # ---------- transporte ----------
    def iniciar_transporte(self):
        """Levanta el socket. Es IDEMPOTENTE a proposito.

        Si la ventana 3D se cae a mitad de camino, el simulador vuelve al modo
        consola y ese camino llama de nuevo aca. Sin esta guarda, el segundo
        intento chocaria con el puerto ya abierto.
        """
        if self.servidor is not None:
            return

        from .local import PUERTO

        self.servidor = ServidorLocal(
            self.mundo, self.robot, self.mundo.limites,
            telemetria=lambda: self._ultima_telemetria,
            verboso=self.verboso,
            puerto=self.puerto if self.puerto is not None else PUERTO)
        self.servidor.arrancar_en_hilo()

    def cerrar(self) -> None:
        if self.servidor is not None:
            try:
                self.servidor.shutdown()
                self.servidor.server_close()
            except Exception:                                 # noqa: BLE001
                pass
            self.servidor = None

    # ---------- telemetria ----------
    def _publicar_telemetria(self) -> None:
        """Deja lista la foto que sirve el socket, leida directo de MuJoCo."""
        nu = self.model.nu
        qpos = self.data.qpos
        qvel = self.data.qvel
        self._ultima_telemetria = {
            "base": {"pos": [float(v) for v in qpos[0:3]],
                     "quat": [float(v) for v in qpos[3:7]]},
            "motor_state": [
                {"id": i,
                 "q": float(qpos[7 + i]) if 7 + i < len(qpos) else 0.0,
                 "dq": float(qvel[6 + i]) if 6 + i < len(qvel) else 0.0}
                for i in range(nu)],
        }
