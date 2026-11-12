"""Levanta el simulador OFICIAL de Unitree.

Que hace, en orden:

  1. Inyecta nuestra configuracion (robot, dominio, interfaz) donde el codigo
     oficial la espera. El bridge oficial hace `import config` y ramifica segun
     config.ROBOT para elegir los mensajes unitree_hg (G1) o unitree_go (Go2),
     asi que hay que dejarsela puesta ANTES de importarlo.
  2. Carga la escena oficial del robot elegido (con silla y mesa de Truco para G1).
  3. Levanta el UnitreeSdk2Bridge OFICIAL: publica rt/lowstate y escucha
     rt/lowcmd, igual que contra el robot real.
  4. Levanta el servicio "sport", que el simulador oficial NO trae. Sin esto
     LocoClient.Move() da timeout: el controlador que hace caminar al G1 corre
     en la PC interna del robot y Unitree no lo publica.
  5. Corre el bucle y la ventana.
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
        self._art_suave = None
        self._art_rival = None
        self.reloj = time.perf_counter     # replaceable for offscreen captures
        self.cartas = self._crear_escena_cartas()
        # Ottoman's joints are the first ones; a second G1 (seat 2) comes after.
        rival = self.cartas.robots.get(2) if self.cartas is not None else None
        self._n_art = (rival.qpos_base if rival is not None else self.model.nq) - 7

    def _crear_escena_cartas(self):
        """The Truco cards, when the scene has them (G1 at the Truco table)."""
        if self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY,
                              "carta_1_0") < 0:
            return None
        from .escena_cartas import EscenaCartas

        escena = EscenaCartas(self.mj, self.model, self.data)
        escena.calibrar(self._qpos_sentado())
        return escena

    def _qpos_sentado(self):
        """qpos of the robot (and the rival G1, if any) seated at the table, with no gesture."""
        from .robots import PREFIJO_RIVAL

        q = self.data.qpos.copy()
        q[:] = 0.0
        bases = [(0, 0.0, (1.0, 0.0, 0.0, 0.0))]
        junta = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT,
                                   PREFIJO_RIVAL + "floating_base_joint")
        if junta >= 0:
            from .cartas import ASIENTO_X
            bases.append((int(self.model.jnt_qposadr[junta]), ASIENTO_X[2], (0.0, 0.0, 0.0, 1.0)))
        for adr, x, quat in bases:
            q[adr:adr + 3] = [x, 0.0, self.robot.altura - 0.23]
            q[adr + 3:adr + 7] = quat
            for pose in (self.robot.pose_de_pie, self.robot.pose_sentado):
                for idx, valor in pose.items():
                    q[adr + 7 + idx] = valor
        return q

    def _escribir_pose_rival(self, dt: float) -> None:
        """The second G1: seated in seat 2 against another robot, else under the floor.

        It does not walk: it sits, holds its cards, plays them and makes the
        gestures of its calls (`Mundo.gesto_rival`). While it holds cards a
        gesture lifts both arms with the fan instead (`ajustar_articulaciones`).
        """
        rival = self.cartas.robots.get(2) if self.cartas is not None else None
        if rival is None:
            return
        from .cartas import ASIENTO_X, MODO_ROBOTS

        q = self.data.qpos
        b = rival.qpos_base
        n = self._n_art
        if self.cartas.mesa.modo != MODO_ROBOTS:
            q[b:b + 3] = [ASIENTO_X[2], 0.0, -10.0]
            q[b + 3:b + 7] = [0.0, 0.0, 0.0, 1.0]
            self._art_rival = None
            return
        q[b:b + 3] = [ASIENTO_X[2], 0.0, self.robot.altura - 0.23]
        q[b + 3:b + 7] = [0.0, 0.0, 0.0, 1.0]          # facing -x, toward Ottoman
        objetivo = [0.0] * n
        for pose in (self.robot.pose_de_pie, self.robot.pose_sentado):
            for idx, valor in pose.items():
                objetivo[idx] = valor
        accion = self.mundo.leer().get("accion_rival", "quieto")
        gestos = getattr(self.robot, "gestos", None) or {}
        if accion in gestos:
            for idx, valor in gestos[accion].items():
                if idx < n:
                    objetivo[idx] = valor
            if accion == "mazo" and 12 < n:
                objetivo[12] = 0.28 * math.sin(self.reloj() * 5.5 * 0.9)
        actual = self._art_rival if self._art_rival is not None else objetivo
        exactas = self.cartas.ajustar_articulaciones(objetivo, actual, asiento=2)
        if self._art_rival is None:
            self._art_rival = list(objetivo)
        else:
            alpha = min(1.0, max(0.08, dt * 10.0))
            for i in range(n):
                self._art_rival[i] += (objetivo[i] - self._art_rival[i]) * alpha
        for idx, valor in exactas.items():
            self._art_rival[idx] = valor
        q[rival.qpos_art:rival.qpos_art + n] = self._art_rival
        self.data.qvel[rival.qvel_art - 6:rival.qvel_art + n] = 0.0

    # ---------- transporte ----------
    def iniciar_transporte(self):
        self.iniciar_dds()

    def iniciar_dds(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(self.cfg.DOMAIN_ID, self.cfg.INTERFACE)

        from unitree_sdk2py_bridge import UnitreeSdk2Bridge

        self.bridge = UnitreeSdk2Bridge(self.model, self.data)
        if self.verboso:
            print("  [DDS] Bridge oficial activo: rt/lowstate, rt/lowcmd")

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

        art = q[7:7 + self._n_art]
        objetivo = [0.0] * len(art)
        for idx, valor in self.robot.pose_de_pie.items():
            if idx < len(objetivo):
                objetivo[idx] = valor

        accion = e.get("accion", "quieto")
        t_osc = self.reloj() * 5.5

        # 1. Si está sentado en la silla, fijar las piernas en pose_sentado (y brazos base sobre la mesa)
        if not e.get("de_pie", True):
            for idx, valor in self.robot.pose_sentado.items():
                if idx < len(objetivo):
                    objetivo[idx] = valor

        # 2. Aun estando sentado o de pie, el tren superior (torso + brazos) ejecuta los gestos activos
        if getattr(self.robot, "gestos", None) and accion in self.robot.gestos:
            for idx, valor in self.robot.gestos[accion].items():
                if idx < len(objetivo):
                    objetivo[idx] = valor
            if accion == "repartiendo" and 25 < len(objetivo):
                # Mano derecha toma carta del mazo (mano izq) y la reparte hacia adelante/costados
                fase_rep = math.sin(t_osc * 1.35)
                objetivo[12] = 0.14 * math.sin(t_osc * 0.67)
                objetivo[22] += 0.18 * fase_rep
                objetivo[23] += 0.16 * max(0.0, -fase_rep)
                objetivo[24] += 0.32 * fase_rep
                objetivo[25] += 0.25 * max(0.0, fase_rep)
            elif accion in ("saludando", "saludo") and 25 < len(objetivo):
                objetivo[23] += 0.14 * math.sin(t_osc)
                objetivo[24] += 0.25 * math.sin(t_osc)
                objetivo[25] += 0.12 * math.sin(t_osc)
            elif accion == "dar_la_mano" and 25 < len(objetivo):
                objetivo[22] += 0.10 * math.sin(t_osc * 1.1)
                objetivo[25] += 0.18 * math.sin(t_osc * 1.1)
            elif accion == "festejando" and 25 < len(objetivo):
                objetivo[15] += 0.18 * math.sin(t_osc)
                objetivo[22] += 0.18 * math.sin(t_osc)
                objetivo[18] += 0.20 * math.sin(t_osc)
                objetivo[25] += 0.20 * math.sin(t_osc)
                objetivo[12] = 0.10 * math.sin(t_osc * 0.5)
            elif accion == "desafiando" and 25 < len(objetivo):
                objetivo[22] += 0.06 * math.sin(t_osc * 0.8)
                objetivo[25] += 0.08 * math.sin(t_osc * 0.8)
            elif accion == "mazo" and 12 < len(objetivo):
                objetivo[12] = 0.28 * math.sin(t_osc * 0.9)
        elif accion == "saludando":
            for idx, valor in self.robot.saludo.items():
                if idx < len(objetivo):
                    objetivo[idx] = valor
        elif e.get("de_pie", True) and e["moviendose"]:
            f = e["fase"]
            for idx, amplitud, desfase in self.robot.marcha:
                if idx < len(objetivo):
                    objetivo[idx] += amplitud * math.sin(f + desfase)

        ahora = self.reloj()
        dt_suave = (ahora - self._qpos_previo[0]) if self._qpos_previo else self.cfg.VIEWER_DT
        exactas = {}
        if self.cartas is not None:
            self.cartas.mesa.avanzar(max(0.0, dt_suave))
            actual = self._art_suave if self._art_suave is not None else objetivo
            exactas = self.cartas.ajustar_articulaciones(objetivo, actual)
        if self._art_suave is None or len(self._art_suave) != len(objetivo):
            self._art_suave = list(objetivo)
        else:
            alpha = min(1.0, max(0.08, dt_suave * 10.0))
            for i in range(len(objetivo)):
                self._art_suave[i] += (objetivo[i] - self._art_suave[i]) * alpha
        # The play animation already interpolates its joints: no extra lag,
        # so the card on the palm and the palm stay together.
        for idx, valor in exactas.items():
            self._art_suave[idx] = valor

        art[:] = self._art_suave

        n = self._n_art
        if self._qpos_previo is not None:
            dt = ahora - self._qpos_previo[0]
            if dt > 1e-6:
                self.data.qvel[6:6 + n] = (q[7:7 + n] - self._qpos_previo[1]) / dt
                self.data.qvel[:6] = 0.0
            else:
                self.data.qvel[:] = 0.0
        else:
            self.data.qvel[:] = 0.0
        self._qpos_previo = (ahora, q[7:7 + n].copy())
        self._escribir_pose_rival(max(0.0, dt_suave))

        self.mj.mj_forward(self.model, self.data)
        if self.cartas is not None:
            # Cards hang from the palms, so they go after the arms are posed.
            self.cartas.aplicar_cartas()
            self.mj.mj_kinematics(self.model, self.data)
        self._publicar_telemetria()

    def _publicar_telemetria(self) -> None:
        """Por DDS no hace falta nada: el bridge oficial publica el LowState
        leyendo `data` en su propio hilo. `SimuladorLocal` lo reemplaza."""

    # ---------- bucles ----------
    def correr_con_ventana(self):
        try:
            self._correr_con_ventana()
        except Exception as exc:                              # noqa: BLE001
            print()
            print("*" * 62)
            print("  NO SE PUDO ABRIR LA VENTANA 3D")
            print(f"  Motivo: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            if "EGL" in str(exc) and os.environ.get("CONDA_PREFIX"):
                print("  Probable causa: conda (glfw propio). Proba con el")
                print("  Python del sistema: conda deactivate")
            print("  Sigo en modo consola.")
            print("*" * 62)
            print()
            self.correr_sin_ventana()

    def _correr_con_ventana(self):
        import mujoco.viewer

        tecla = self.cartas.tecla if self.cartas is not None else None
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False,
            key_callback=tecla,
        ) as v:
            if self.robot.tipo == "humanoide":
                v.cam.lookat[:] = [0.26, 0.0, 0.62]
                v.cam.distance = 2.25
                v.cam.elevation = -20
                v.cam.azimuth = 135
            else:
                v.cam.distance = 2.6
                v.cam.elevation = -20
            self.iniciar_transporte()
            if self.cartas is not None and self.verboso:
                print("  [CARTAS] F8 cambia de camara (general / detras de cada robot).")
                print("           Contra un humano, las camaras detras del robot no se muestran.")
            periodo = self.cfg.VIEWER_DT
            while v.is_running():
                inicio = time.perf_counter()
                self.mundo.avanzar()
                self._escribir_pose()
                self._volcar_avisos()
                if self.cartas is not None:
                    with v.lock():
                        self.cartas.sincronizar_visor(v.cam)
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
