"""Applies `MesaCartas` to a loaded MuJoCo scene: card poses, faces, arms, cameras.

Everything is kinematic. Cards are mocap bodies whose pose is written every
frame; a card "in a hand" is placed relative to that hand's palm site. The
robot's arm poses for holding the fan, reaching a card and laying it on the
table are solved once with a small damped least-squares IK on a scratch copy
of the model, then interpolated.
"""

from __future__ import annotations

import math

import numpy as np

from . import cartas as C
from .robots import (G1_L_ELBOW, G1_L_SHOULDER_P, G1_L_SHOULDER_R, G1_L_SHOULDER_Y,
                     G1_L_WRIST_P, G1_L_WRIST_R, G1_L_WRIST_Y, G1_R_ELBOW,
                     G1_R_SHOULDER_P, G1_R_SHOULDER_R, G1_R_SHOULDER_Y,
                     G1_R_WRIST_P, G1_R_WRIST_R, G1_R_WRIST_Y, G1_WAIST_PITCH)

BRAZO_IZQ = (G1_L_SHOULDER_P, G1_L_SHOULDER_R, G1_L_SHOULDER_Y, G1_L_ELBOW,
             G1_L_WRIST_R, G1_L_WRIST_P, G1_L_WRIST_Y)
BRAZO_DER = (G1_R_SHOULDER_P, G1_R_SHOULDER_R, G1_R_SHOULDER_Y, G1_R_ELBOW,
             G1_R_WRIST_R, G1_R_WRIST_P, G1_R_WRIST_Y)
OFFSET_ART = 7            # qpos index of the first joint (after the free base)

# Where the fan is held, in the seat's frame (x forward, y left, z up).
MANO_ABANICO = (0.25, 0.13, 0.89)
INCLINACION_ABANICO = math.radians(38)     # face tilted up toward the eyes
APERTURA_ABANICO = math.radians(24)        # angle between neighbouring cards
ALTURA_SOBRE_MESA = 0.045                   # palm above the slot when laying a card
ELEVACION_VIAJE = 0.07                      # extra lift halfway to the table

# A natural seated arm to start the IK from and to bias it toward.
SEMILLA_IZQ = {G1_L_SHOULDER_P: -0.75, G1_L_SHOULDER_R: 0.25, G1_L_SHOULDER_Y: -0.2,
               G1_L_ELBOW: 0.1, G1_L_WRIST_R: 0.0, G1_L_WRIST_P: 0.0, G1_L_WRIST_Y: 0.0}
SEMILLA_DER = {G1_R_SHOULDER_P: -0.6, G1_R_SHOULDER_R: -0.2, G1_R_SHOULDER_Y: 0.2,
               G1_R_ELBOW: 0.3, G1_R_WRIST_R: 0.0, G1_R_WRIST_P: 0.0, G1_R_WRIST_Y: 0.0}


def suave(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _mat_a_quat(mj, m: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mj.mju_mat2Quat(q, np.ascontiguousarray(m).reshape(9))
    return q


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def orientacion_abanico(asiento: int) -> np.ndarray:
    """Rotation (columns: card right, card up, card face normal) of the fan."""
    s = C.ASIENTO_SENTIDO[asiento]
    a = INCLINACION_ABANICO
    normal = np.array([-s * math.cos(a), 0.0, math.sin(a)])     # toward the owner
    arriba = np.array([s * math.sin(a), 0.0, math.cos(a)])
    derecha = np.cross(arriba, normal)
    return np.column_stack([derecha, arriba, normal])


def orientacion_mesa(asiento: int, baza: int) -> np.ndarray:
    _, yaw = C.slot_mesa(asiento, baza)
    return _rot_z(yaw)


class EscenaCartas:
    """Bridges the card state machine and a MuJoCo model with the Truco table."""

    def __init__(self, mj, model, data, mesa: C.MesaCartas | None = None):
        self.mj, self.model, self.data = mj, model, data
        self.mesa = mesa or C.MesaCartas()
        nombre = lambda tipo, n: mj.mj_name2id(model, tipo, n)          # noqa: E731
        obj = mj.mjtObj
        self.mocap = {}
        self.geom_frente = {}
        for asiento in C.ASIENTOS:
            for i in range(C.CARTAS_POR_ASIENTO):
                n = C.cuerpo(asiento, i)
                b = nombre(obj.mjOBJ_BODY, n)
                if b < 0:
                    raise ValueError(f"La escena no tiene el cuerpo de carta {n}")
                self.mocap[n] = int(model.body_mocapid[b])
                self.geom_frente[n] = nombre(obj.mjOBJ_GEOM, f"{n}_frente")
        self.material = {c: nombre(obj.mjOBJ_MATERIAL, C.nombre_material(c))
                         for c in C.todas_las_cartas()}
        self.material_dorso = nombre(obj.mjOBJ_MATERIAL, "mat_dorso")
        self.sitio_izq = nombre(obj.mjOBJ_SITE, "mano_izq")
        self.sitio_der = nombre(obj.mjOBJ_SITE, "mano_der")
        self.camaras = {n: nombre(obj.mjOBJ_CAMERA, n) for n in C.CAMARAS}
        self._calibrado = False
        self._pose_sostener: dict[int, float] = {}
        self._pose_alcanzar: dict[tuple[int, int], dict[int, float]] = {}
        self._pose_elevar: dict[tuple[int, int, int], dict[int, float]] = {}
        self._pose_apoyar: dict[int, dict[int, float]] = {}
        self._inicio_fase: dict[int, float] | None = None
        self._fase_previa = None
        self._enganche: dict | None = None
        self._abanico_jugada = (0, 1)       # (k, n) of the card being played

    # ------------------------------------------------------------------ IK
    def calibrar(self, base_qpos: np.ndarray) -> None:
        """Solves every arm pose from the seated robot whose qpos is given."""
        m, mj = self.model, self.mj
        d = mj.MjData(m)
        d.qpos[:] = base_qpos
        objetivo = np.array(MANO_ABANICO)            # seat 1 frame == world here
        self._pose_sostener = self._ik(d, self.sitio_izq, BRAZO_IZQ, objetivo, SEMILLA_IZQ)
        self._fijar(d, self._pose_sostener)
        mj.mj_kinematics(m, d)
        for n in range(1, C.CARTAS_POR_ASIENTO + 1):
            for k in range(n):
                pos, rot = self.pose_en_abanico(1, k, n, d)
                agarre = pos + rot[:, 1] * C.CARTA_ALTO * 0.30 - rot[:, 2] * 0.035
                self._pose_alcanzar[(n, k)] = self._ik(d, self.sitio_der, BRAZO_DER,
                                                        agarre, SEMILLA_DER)
        for baza in range(C.CARTAS_POR_ASIENTO):
            (x, y, z), _ = C.slot_mesa(1, baza)
            apoyo = np.array([x, y, z + ALTURA_SOBRE_MESA])
            self._pose_apoyar[baza] = self._ik(d, self.sitio_der, BRAZO_DER, apoyo, SEMILLA_DER)
        self._calibrado = True

    def _fijar(self, d, pose: dict[int, float]) -> None:
        for idx, valor in pose.items():
            d.qpos[OFFSET_ART + idx] = valor

    def _ik(self, d, sitio: int, cadena, objetivo, semilla, iteraciones: int = 160) -> dict[int, float]:
        mj, m = self.mj, self.model
        cols = [m.jnt_dofadr[self._junta(idx)] for idx in cadena]
        rangos = [m.jnt_range[self._junta(idx)] for idx in cadena]
        q = np.array([semilla[idx] for idx in cadena], dtype=float)
        nominal = q.copy()
        jacp = np.zeros((3, m.nv))
        for _ in range(iteraciones):
            for idx, v in zip(cadena, q):
                d.qpos[OFFSET_ART + idx] = v
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            error = objetivo - d.site_xpos[sitio]
            if np.linalg.norm(error) < 1e-4:
                break
            mj.mj_jacSite(m, d, jacp, None, sitio)
            J = jacp[:, cols]
            lam, mu = 1e-3, 2e-3
            A = J.T @ J + (lam + mu) * np.eye(len(cadena))
            dq = np.linalg.solve(A, J.T @ error + mu * (nominal - q))
            q = q + np.clip(dq, -0.2, 0.2)
            q = np.array([min(r[1], max(r[0], v)) for v, r in zip(q, rangos)])
        return {idx: float(v) for idx, v in zip(cadena, q)}

    def _junta(self, idx: int) -> int:
        # Joint ids follow qpos order after the free joint (id 0).
        return idx + 1

    # ------------------------------------------------------------ geometry
    def pose_en_abanico(self, asiento: int, k: int, n: int, d=None):
        """World position and rotation of card k of a fan of n in the left hand."""
        d = self.data if d is None else d
        rot0 = orientacion_abanico(asiento)
        pivote = d.site_xpos[self.sitio_izq] + rot0[:, 2] * 0.03 - rot0[:, 1] * 0.01
        giro = -(k - (n - 1) / 2) * APERTURA_ABANICO
        c, s = math.cos(giro), math.sin(giro)
        en_plano = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        rot = rot0 @ en_plano
        centro = pivote + rot[:, 1] * C.CARTA_ALTO * 0.36 + rot0[:, 2] * 0.0016 * k
        return centro, rot

    # --------------------------------------------------------------- arms
    def ajustar_articulaciones(self, objetivo: list, actual) -> dict[int, float]:
        """Overrides arm targets for holding and playing. Returns exact joints.

        `objetivo` (the gesture/seated targets) is edited in place for the left
        arm's holding pose, which then blends in with the normal smoothing. The
        returned joints come from the play animation and are written as is.
        """
        if not self._calibrado:
            return {}
        estado = self.mesa.estado()
        sostiene = any(c.asiento == 1 and c.lugar == "mano" for c in estado)
        jugada = self.mesa.jugada_activa(1)
        if sostiene or (jugada and jugada[0] in ("alcanzar", "llevar")):
            for idx, v in self._pose_sostener.items():
                objetivo[idx] = v
        if jugada is None or jugada[0] == "deslizar":
            self._fase_previa = None
            return {}
        fase, p, indice, baza = jugada
        clave = (fase, indice, baza)
        if clave != self._fase_previa:
            self._inicio_fase = {idx: float(actual[idx]) for idx in BRAZO_DER}
            self._fase_previa = clave
            if fase == "alcanzar":
                carta = next(c for c in estado if c.cuerpo == C.cuerpo(1, indice))
                self._abanico_jugada = (carta.indice_abanico, max(1, carta.cartas_en_abanico))
        t = suave(p)
        if fase == "alcanzar":
            fin = self._pose_alcanzar.get(self._abanico_jugada, next(iter(self._pose_alcanzar.values())))
            return self._mezclar(self._inicio_fase, fin, t)
        if fase == "llevar":
            inicio = self._pose_alcanzar.get(self._abanico_jugada, self._inicio_fase)
            medio = self._pose_elevada(baza)
            if p < 0.5:
                return self._mezclar(inicio, medio, suave(p * 2))
            return self._mezclar(medio, self._pose_apoyar[baza], suave(p * 2 - 1))
        # volver: from the table back to whatever the arm would otherwise do.
        reposo = {idx: objetivo[idx] for idx in BRAZO_DER}
        return self._mezclar(self._pose_apoyar[baza], reposo, t)

    def _pose_elevada(self, baza: int) -> dict[int, float]:
        clave = (0, 0, baza)
        if clave not in self._pose_elevar:
            m, mj = self.model, self.mj
            d = mj.MjData(m)
            d.qpos[:] = self.data.qpos
            (x, y, z), _ = C.slot_mesa(1, baza)
            fan, _ = self.pose_en_abanico(1, 0, 1)
            medio = (np.array([x, y, z + ALTURA_SOBRE_MESA]) + fan) / 2 + [0, 0, ELEVACION_VIAJE]
            self._pose_elevar[clave] = self._ik(d, self.sitio_der, BRAZO_DER, medio, SEMILLA_DER)
        return self._pose_elevar[clave]

    @staticmethod
    def _mezclar(a: dict, b: dict, t: float) -> dict[int, float]:
        return {idx: a[idx] + (b[idx] - a[idx]) * t for idx in b}

    # -------------------------------------------------------------- cards
    def aplicar_cartas(self) -> None:
        """Writes every card's mocap pose and face material from the state."""
        mj, d = self.mj, self.data
        for c in self.mesa.estado():
            mid = self.mocap[c.cuerpo]
            pos, rot = self._pose_carta(c)
            d.mocap_pos[mid] = pos
            d.mocap_quat[mid] = _mat_a_quat(mj, rot)
            oculta = c.carta is None or (c.lugar == "mano" and self.mesa.ocultar_caras_en_mano(c.asiento))
            self.model.geom_matid[self.geom_frente[c.cuerpo]] = (
                self.material_dorso if oculta else self.material[c.carta])

    def _pose_carta(self, c: C.CartaEnEscena):
        if c.lugar == "oculta":
            return np.array([C.MESA_X, 0.0, -1.0]), np.eye(3)
        if c.lugar == "mano":
            return self.pose_en_abanico(c.asiento, c.indice_abanico, max(1, c.cartas_en_abanico))
        (x, y, z), _ = C.slot_mesa(c.asiento, max(0, c.baza))
        destino = np.array([x, y, z])
        rot_mesa = orientacion_mesa(c.asiento, max(0, c.baza))
        if c.lugar == "mesa":
            if self._enganche and self._enganche["cuerpo"] == c.cuerpo:
                self._enganche = None
            return destino, rot_mesa
        p = c.progreso
        if c.fase == "deslizar":
            origen = np.array(C.borde_humano(c.asiento))
            pos = origen + (destino - origen) * suave(p)
            pos[2] += 0.05 * math.sin(math.pi * p) * (1 - p)
            return pos, rot_mesa
        # llevar: the card rides on the right palm, then settles in its slot.
        mano = self.data.site_xpos[self.sitio_der].copy()
        if self._enganche is None or self._enganche["cuerpo"] != c.cuerpo:
            pos0, rot0 = self.pose_en_abanico(c.asiento, *self._abanico_jugada)
            self._enganche = {"cuerpo": c.cuerpo, "offset": pos0 - mano,
                              "q0": _mat_a_quat(self.mj, rot0)}
        e = self._enganche
        colgada = mano + e["offset"] * (1 - suave(min(1.0, p / 0.35))) + \
            np.array([0, 0, -0.03]) * suave(min(1.0, p / 0.35))
        asentar = suave((p - 0.62) / 0.38)
        pos = colgada + (destino - colgada) * asentar
        q = _slerp(e["q0"], _mat_a_quat(self.mj, rot_mesa), suave(min(1.0, p / 0.8)))
        rot = np.zeros(9)
        self.mj.mju_quat2Mat(rot, q)
        return pos, rot.reshape(3, 3)

    # ------------------------------------------------------------ cameras
    def id_camara(self, nombre: str | None = None) -> int:
        return self.camaras[nombre or self.mesa.camara]

    def tecla(self, codigo: int) -> None:
        """Viewer key callback: F7 switches to the next allowed camera."""
        if codigo == C.TECLA_CAMARA:
            self.mesa.elegir_camara(C.camara_siguiente(self.mesa.camara, self.mesa.modo))
            self._camara_mostrada = None          # apply even if it is the same

    def sincronizar_visor(self, cam) -> None:
        """Puts the viewer on the camera asked for, and never on a forbidden one."""
        fija = self.mj.mjtCamera.mjCAMERA_FIXED
        pedida = self.mesa.camara
        if pedida != getattr(self, "_camara_mostrada", None):
            cam.type, cam.fixedcamid = fija, self.camaras[pedida]
            self._camara_mostrada = pedida
            return
        if cam.type == fija:
            permitidas = {self.camaras[n] for n in C.camaras_permitidas(self.mesa.modo)}
            if cam.fixedcamid not in permitidas:
                cam.fixedcamid = self.camaras[C.CAMARA_GENERAL]
