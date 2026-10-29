"""Levanta el simulador OFICIAL de Unitree.

Que hace, en orden:

  1. Inyecta nuestra configuracion (robot, dominio, interfaz) donde el codigo
     oficial la espera. El bridge oficial hace `import config` y ramifica segun
     config.ROBOT para elegir los mensajes unitree_hg (G1) o unitree_go (Go2),
     asi que hay que dejarsela puesta ANTES de importarlo.
  2. Carga la escena oficial del robot elegido.
  3. Levanta el UnitreeSdk2Bridge OFICIAL: publica rt/lowstate y escucha
     rt/lowcmd, igual que contra el robot real.
  4. Levanta el servicio "sport", que el simulador oficial NO trae. Sin esto
     LocoClient.Move() da timeout: el controlador que hace caminar al G1 corre
     en la PC interna del robot y Unitree no lo publica.
  5. Corre el bucle y la ventana.

Sobre la fisica: el bucle usa mj_forward, no mj_step. El G1 es un humanoide y
SE CAE SOLO sin un controlador de locomocion. La marcha de las patas es
cosmetica: esta es la capa de alto nivel (Move), no la de control articular.
"""

from __future__ import annotations

import math
import os
import sys
import time
import types


def preparar_config(robot, repo: str, domain: int, interfaz: str):
    """Deja un modulo `config` en sys.modules para el codigo oficial."""
    cfg = types.ModuleType("config")
    cfg.ROBOT = robot.clave
    # OJO: hay que pedirle la ruta al robot, NO armarla con un join.
    # `ruta_escena()` GENERA la escena limpia del Go2 si todavia no existe
    # (`scene_limpia.xml`, que no es un archivo del repo oficial).
    ruta = robot.ruta_escena()
    if not ruta:
        raise FileNotFoundError(
            f"No encuentro el modelo de {robot.clave} ni pude generar su "
            f"escena. Fijate que este 'entorno/sim/unitree_mujoco/"
            f"unitree_robots/{robot.clave}/'.")
    cfg.ROBOT_SCENE = ruta
    cfg.DOMAIN_ID = domain
    cfg.INTERFACE = interfaz
    cfg.USE_JOYSTICK = 0
    cfg.JOYSTICK_TYPE = "xbox"
    cfg.JOYSTICK_DEVICE = 0
    cfg.PRINT_SCENE_INFORMATION = False
    cfg.ENABLE_ELASTIC_BAND = False  # no hace falta: no corremos fisica
    cfg.SIMULATE_DT = 0.005
    cfg.VIEWER_DT = 0.02
    sys.modules["config"] = cfg
    return cfg


class SimuladorOficial:
    def __init__(self, mundo, robot, repo: str, domain: int = 0,
                 interfaz: str = "lo", verboso: bool = True):
        self.mundo = mundo
        self.robot = robot
        self.repo = repo
        self.verboso = verboso

        self.cfg = preparar_config(robot, repo, domain, interfaz)
        if not os.path.exists(self.cfg.ROBOT_SCENE):
            raise FileNotFoundError(
                f"No encuentro la escena oficial:\n  {self.cfg.ROBOT_SCENE}")

        # El bridge oficial vive en el repo de Unitree.
        ruta_oficial = os.path.join(repo, "simulate_python")
        if ruta_oficial not in sys.path:
            sys.path.insert(0, ruta_oficial)

        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(self.cfg.ROBOT_SCENE)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.cfg.SIMULATE_DT
        self.model.opt.gravity[:] = 0.0   # sin fisica, el robot no se cae

        self.bridge = None
        self.servicio = None
        self._qpos_previo = None

    # ---------- transporte ----------
    def iniciar_transporte(self):
        """Punto de enganche: `SimuladorLocal` lo reemplaza por el socket.

        Los bucles de `correr_*` llaman aca y no directo a `iniciar_dds`, asi
        el modo local reusa TODO el resto -- MuJoCo, el visor, la pose -- sin
        duplicar una linea.
        """
        self.iniciar_dds()

    def iniciar_dds(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(self.cfg.DOMAIN_ID, self.cfg.INTERFACE)

        # Bridge OFICIAL: rt/lowstate, rt/lowcmd, rt/sportmodestate
        from unitree_sdk2py_bridge import UnitreeSdk2Bridge

        self.bridge = UnitreeSdk2Bridge(self.model, self.data)
        if self.verboso:
            print("  [DDS] Bridge oficial activo: rt/lowstate, rt/lowcmd")

        # Servicio de locomocion, que el oficial NO trae.
        if self.robot.clave == "g1":
            from .servicio_sport import ServicioSportSimulado

            self.servicio = ServicioSportSimulado(self.mundo, verboso=self.verboso)
            if self.verboso:
                print("  [DDS] Servicio de locomocion activo: LocoClient responde")
        else:
            from .servicio_sport_go2 import ServicioSportGo2

            self.servicio = ServicioSportGo2(self.mundo, verboso=self.verboso)
            if self.verboso:
                print("  [DDS] Servicio de locomocion activo: SportClient responde")

    # ---------- pose ----------
    def _escribir_pose(self):
        e = self.mundo.leer()
        q = self.data.qpos

        q[0] = e["x"]
        q[1] = e["y"]
        q[2] = self.robot.altura + e.get("altura", 0.0)
        media = e["yaw"] / 2.0
        q[3], q[4], q[5], q[6] = math.cos(media), 0.0, 0.0, math.sin(media)

        art = q[7:]
        art[:] = 0.0
        for idx, valor in self.robot.pose_de_pie.items():
            if idx < len(art):
                art[idx] = valor

        if not e.get("de_pie", True):
            for idx, valor in self.robot.pose_sentado.items():
                if idx < len(art):
                    art[idx] = valor
        elif e["accion"] == "saludando":
            for idx, valor in self.robot.saludo.items():
                if idx < len(art):
                    art[idx] = valor
        elif e["moviendose"]:
            f = e["fase"]
            for idx, amplitud, desfase in self.robot.marcha:
                if idx < len(art):
                    art[idx] += amplitud * math.sin(f + desfase)

        # Velocidad de las articulaciones por diferencia finita. De aca salen
        # el gyro y la velocidad de motor que publica el bridge oficial.
        ahora = time.perf_counter()
        if self._qpos_previo is not None:
            dt = ahora - self._qpos_previo[0]
            if dt > 1e-6:
                self.data.qvel[6:] = (q[7:] - self._qpos_previo[1]) / dt
                self.data.qvel[:6] = 0.0
            else:
                self.data.qvel[:] = 0.0
        else:
            self.data.qvel[:] = 0.0
        self._qpos_previo = (ahora, q[7:].copy())

        self.mj.mj_forward(self.model, self.data)
        self._publicar_telemetria()

    def _publicar_telemetria(self) -> None:
        """Por DDS no hace falta nada: el bridge oficial publica el LowState
        leyendo `data` en su propio hilo. `SimuladorLocal` lo reemplaza."""

    # ---------- bucles ----------
    def correr_con_ventana(self):
        """Abre la ventana 3D. Si no se puede, sigue en consola.

        Que el import de MuJoCo funcione NO garantiza que se pueda abrir una
        ventana: hace falta un driver con OpenGL 3.3. En una maquina virtual sin
        GPU, o por escritorio remoto, `launch_passive` muere con
        `GLFWError: (65542) WGL: The driver does not appear to support OpenGL`.
        """
        try:
            self._correr_con_ventana()
        except Exception as exc:                              # noqa: BLE001
            print()
            print("*" * 62)
            print("  NO SE PUDO ABRIR LA VENTANA 3D")
            print(f"  Motivo: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            # conda trae su propio glfw y su propio libstdc++, que chocan con
            # los del sistema. La pista es "EGL" en el mensaje.
            if "EGL" in str(exc) and os.environ.get("CONDA_PREFIX"):
                print("  Probable causa: conda (glfw propio). Proba con el")
                print("  Python del sistema: conda deactivate")
            print("  Sigo en modo consola.")
            print("*" * 62)
            print()
            self.correr_sin_ventana()

    def _correr_con_ventana(self):
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as v:
            v.cam.distance = 3.5 if self.robot.tipo == "humanoide" else 2.6
            v.cam.elevation = -20
            self.iniciar_transporte()
            periodo = self.cfg.VIEWER_DT
            while v.is_running():
                inicio = time.perf_counter()
                self.mundo.avanzar()
                self._escribir_pose()
                self._volcar_avisos()
                v.sync()
                resto = periodo - (time.perf_counter() - inicio)
                if resto > 0:
                    time.sleep(resto)

    def correr_sin_ventana(self):
        """El bucle cuando no hay ventana 3D. Dibuja el recorrido en texto."""
        from .consola import VistaConsola

        vista = VistaConsola(verboso=self.verboso) if self.verboso else None
        self.iniciar_transporte()
        while True:
            self.mundo.avanzar()
            self._escribir_pose()
            self._volcar_avisos()
            if vista is not None:
                vista.actualizar(self.mundo.leer())
            time.sleep(self.cfg.VIEWER_DT)

    def _volcar_avisos(self):
        if self.mundo.avisos and self.verboso:
            for aviso in self.mundo.avisos:
                print(f"  [LIMITE] {aviso}")
            self.mundo.avisos.clear()
