"""Applies `MesaCartas` to a loaded MuJoCo scene: card poses, faces, arms, cameras.

Everything is kinematic. Cards are mocap bodies whose pose is written every
frame; a card "in a hand" is placed relative to that hand's palm site. The
robot's arm poses for holding the fan, reaching a card and laying it on the
table are solved once with a small damped least-squares IK on a scratch copy
of the model, then interpolated. The fan is held with both hands in front of
the chest, like a person holding a hand of cards (ADR 0067): the left palm
carries it and the right palm grips its other side, leaving it only to pull a
card out and lay it on the table. Each seated G1 (Ottoman in seat 1 and, in
robot-vs-robot mode, the rival in seat 2) has its own arms and play state.
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

# Where the fan is held, in the seat's frame (x forward, y left, z up): the
# bottom of the fan (where the cards meet), in front of the chest.
CENTRO_ABANICO = (0.27, 0.0, 0.88)
SEPARACION_MANOS = 0.15                    # palm to palm, one on each side of the fan
MANOS_DETRAS = 0.055                        # palms behind the cards (fingers on the backs)
MANOS_ARRIBA = 0.04                       # palms a little above the bottom of the fan
INCLINACION_ABANICO = math.radians(15)     # face tilted up a little toward the eyes
GIRO_ABANICO = math.radians(50)            # backs toward the overview camera, faces toward the behind-robot one
APERTURA_ABANICO = math.radians(24)        # angle between neighbouring cards
ELEVACION_GESTO = 0.45                     # most a gesture may lift both arms while holding
ANGULO_DEDOS = math.radians(5)            # fingers lean in from vertical along the fan
LARGO_MANO = 0.10                          # wrist to palm site
ALTURA_SOBRE_MESA = 0.045                   # palm above the slot when laying a card
ELEVACION_VIAJE = 0.07                      # extra lift halfway to the table

# A natural seated arm to start the IK from and to bias it toward.
SEMILLA_IZQ = {G1_L_SHOULDER_P: -0.75, G1_L_SHOULDER_R: 0.25, G1_L_SHOULDER_Y: -0.2,
               G1_L_ELBOW: 0.1, G1_L_WRIST_R: 0.0, G1_L_WRIST_P: 0.0, G1_L_WRIST_Y: 0.0}
SEMILLA_DER = {G1_R_SHOULDER_P: -0.6, G1_R_SHOULDER_R: -0.2, G1_R_SHOULDER_Y: 0.2,
               G1_R_ELBOW: 0.3, G1_R_WRIST_R: 0.0, G1_R_WRIST_P: 0.0, G1_R_WRIST_Y: 0.0}
SEMILLA_SOSTENER_DER = {G1_R_SHOULDER_P: -0.75, G1_R_SHOULDER_R: -0.25, G1_R_SHOULDER_Y: 0.2,
                        G1_R_ELBOW: 0.1, G1_R_WRIST_R: 0.0, G1_R_WRIST_P: 0.0, G1_R_WRIST_Y: 0.0}


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
    base = np.column_stack([derecha, arriba, normal])
    # Seat 2 is seat 1 turned half a turn (same hands), so the turn is the same.
    return _rot_z(-GIRO_ABANICO) @ base


def orientacion_mesa(asiento: int, baza: int) -> np.ndarray:
    _, yaw = C.slot_mesa(asiento, baza)
    return _rot_z(yaw)


def a_mundo(asiento: int, punto) -> np.ndarray:
    """A point in the seat's frame (x forward, y left, z up) in world coordinates."""
    s = C.ASIENTO_SENTIDO[asiento]
    x, y, z = punto
    return np.array([C.ASIENTO_X[asiento] + s * x, s * y, z])


def manos_en_abanico(asiento: int) -> tuple[np.ndarray, np.ndarray]:
    """Where the left and right palms grip the fan (world): one on each side of
    its bottom, a little behind the cards, in the fan's plane."""
    rot0 = orientacion_abanico(asiento)
    normal = rot0[:, 2]
    izquierda = np.array([0.0, C.ASIENTO_SENTIDO[asiento], 0.0])        # the owner's left
    izquierda = izquierda - normal * float(izquierda @ normal)
    izquierda /= np.linalg.norm(izquierda)
    base = (a_mundo(asiento, CENTRO_ABANICO) + rot0[:, 1] * MANOS_ARRIBA - normal * MANOS_DETRAS)
    lado = izquierda * SEPARACION_MANOS / 2
    return base + lado, base - lado


def _plano_abanico(asiento: int):
    rot0 = orientacion_abanico(asiento)
    normal = rot0[:, 2]
    izquierda = np.array([0.0, C.ASIENTO_SENTIDO[asiento], 0.0])
    izquierda = izquierda - normal * float(izquierda @ normal)
    izquierda /= np.linalg.norm(izquierda)
    arriba = np.cross(normal, izquierda)
    if arriba[2] < 0:
        arriba = -arriba
    return izquierda, arriba, normal


def munecas_en_abanico(asiento: int) -> tuple[np.ndarray, np.ndarray]:
    """Where each wrist goes so the fingers point up along the fan's sides,
    leaning in toward its middle (a person's grip on a hand of cards)."""
    izquierda, arriba, normal = _plano_abanico(asiento)
    izq, der = manos_en_abanico(asiento)
    a = ANGULO_DEDOS
    hacia_izq = arriba * math.cos(a) - izquierda * math.sin(a)        # left fingers lean right
    hacia_der = arriba * math.cos(a) + izquierda * math.sin(a)
    return izq - hacia_izq * LARGO_MANO - normal * 0.01, der - hacia_der * LARGO_MANO - normal * 0.01


class BrazosRobot:
    """One seated G1: where its joints and palms are, its solved arm poses and
    the state of the card it is playing. Seat 1 is Ottoman; seat 2 the rival G1
    (names prefixed with `robots.PREFIJO_RIVAL`)."""

    def __init__(self, mj, model, asiento: int, prefijo: str = ""):
        obj = mj.mjtObj
        self.asiento = asiento
        base = mj.mj_name2id(model, obj.mjOBJ_JOINT, prefijo + "floating_base_joint")
        if base < 0:
            raise ValueError(f"La escena no tiene el robot del asiento {asiento}")
        self.junta_base = base
        self.qpos_base = int(model.jnt_qposadr[base])
        self.qpos_art = self.qpos_base + 7          # first hinge right after the free joint
        self.qvel_art = int(model.jnt_dofadr[base]) + 6
        self.sitio_izq = mj.mj_name2id(model, obj.mjOBJ_SITE, prefijo + "mano_izq")
        self.sitio_der = mj.mj_name2id(model, obj.mjOBJ_SITE, prefijo + "mano_der")
        self.pose_sostener: dict[int, float] = {}       # left arm, carrying the fan
        self.pose_sostener_der: dict[int, float] = {}   # right arm, gripping its other side
        self.sentado: dict[int, float] = {}             # both arms seated, no gesture
        self.offset_abanico = np.zeros(3)               # fan bottom relative to the left palm
        self.pose_alcanzar: dict[tuple[int, int], dict[int, float]] = {}
        self.pose_elevar: dict[int, dict[int, float]] = {}
        self.pose_apoyar: dict[int, dict[int, float]] = {}
        self.inicio_fase: dict[int, float] | None = None
        self.fase_previa = None
        self.enganche: dict | None = None
        self.abanico_jugada = (0, 1)             # (k, n) of the card being played

    def junta(self, idx: int) -> int:
        # Joint ids follow qpos order after the robot's free joint.
        return self.junta_base + 1 + idx


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
        from .robots import PREFIJO_RIVAL
        self.robots = {1: BrazosRobot(mj, model, 1)}
        if nombre(obj.mjOBJ_JOINT, PREFIJO_RIVAL + "floating_base_joint") >= 0:
            self.robots[2] = BrazosRobot(mj, model, 2, PREFIJO_RIVAL)
        self.sitio_izq = self.robots[1].sitio_izq
        self.sitio_der = self.robots[1].sitio_der
        self.camaras = {n: nombre(obj.mjOBJ_CAMERA, n) for n in C.CAMARAS}
        self._calibrado = False

    @property
    def hay_rival(self) -> bool:
        """True when the scene has the second G1 for seat 2."""
        return 2 in self.robots

    # ------------------------------------------------------------------ IK
    def calibrar(self, base_qpos: np.ndarray) -> None:
        """Solves every arm pose from the seated robots whose qpos is given."""
        m, mj = self.model, self.mj
        d = mj.MjData(m)
        d.qpos[:] = base_qpos
        for asiento, r in self.robots.items():
            r.sentado = {idx: float(base_qpos[r.qpos_art + idx]) for idx in BRAZO_IZQ + BRAZO_DER}
            izq, der = manos_en_abanico(asiento)
            muneca_izq, muneca_der = munecas_en_abanico(asiento)
            r.pose_sostener = self._ik(d, r, r.sitio_izq, BRAZO_IZQ, izq, SEMILLA_IZQ, muneca=muneca_izq)
            r.pose_sostener_der = self._ik(d, r, r.sitio_der, BRAZO_DER, der, SEMILLA_SOSTENER_DER,
                                           muneca=muneca_der)
            self._fijar(d, r, r.pose_sostener)
            self._fijar(d, r, r.pose_sostener_der)
            mj.mj_kinematics(m, d)
            # The fan rides on the left palm; the right one only grips it.
            r.offset_abanico = a_mundo(asiento, CENTRO_ABANICO) - d.site_xpos[r.sitio_izq]
            fan, _ = self.pose_en_abanico(asiento, 0, 1, d)
            for n in range(1, C.CARTAS_POR_ASIENTO + 1):
                for k in range(n):
                    pos, rot = self.pose_en_abanico(asiento, k, n, d)
                    agarre = pos + rot[:, 1] * C.CARTA_ALTO * 0.30 - rot[:, 2] * 0.035
                    r.pose_alcanzar[(n, k)] = self._ik(d, r, r.sitio_der, BRAZO_DER,
                                                       agarre, SEMILLA_DER)
            for baza in range(C.CARTAS_POR_ASIENTO):
                (x, y, z), _ = C.slot_mesa(asiento, baza)
                apoyo = np.array([x, y, z + ALTURA_SOBRE_MESA])
                r.pose_apoyar[baza] = self._ik(d, r, r.sitio_der, BRAZO_DER, apoyo, SEMILLA_DER)
                medio = (apoyo + fan) / 2 + [0, 0, ELEVACION_VIAJE]
                r.pose_elevar[baza] = self._ik(d, r, r.sitio_der, BRAZO_DER, medio, SEMILLA_DER)
        self._calibrado = True

    def _fijar(self, d, r: BrazosRobot, pose: dict[int, float]) -> None:
        for idx, valor in pose.items():
            d.qpos[r.qpos_art + idx] = valor

    def _ik(self, d, r: BrazosRobot, sitio: int, cadena, objetivo, semilla,
            iteraciones: int = 160, muneca=None) -> dict[int, float]:
        """Joints that put the palm `sitio` on `objetivo`. With `muneca` (a world
        point), the wrist is also pulled there, which sets where the fingers point."""
        mj, m = self.mj, self.model
        cols = [m.jnt_dofadr[r.junta(idx)] for idx in cadena]
        rangos = [m.jnt_range[r.junta(idx)] for idx in cadena]
        q = np.array([semilla[idx] for idx in cadena], dtype=float)
        nominal = q.copy()
        jacp = np.zeros((3, m.nv))
        jacm = np.zeros((3, m.nv))
        cuerpo = int(m.site_bodyid[sitio])
        for _ in range(iteraciones * (2 if muneca is not None else 1)):
            for idx, v in zip(cadena, q):
                d.qpos[r.qpos_art + idx] = v
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            error = objetivo - d.site_xpos[sitio]
            mj.mj_jacSite(m, d, jacp, None, sitio)
            J = jacp[:, cols]
            if muneca is not None:
                mj.mj_jac(m, d, jacm, None, d.xpos[cuerpo], cuerpo)
                peso = 0.6
                error = np.concatenate([error, peso * (np.asarray(muneca) - d.xpos[cuerpo])])
                J = np.vstack([J, peso * jacm[:, cols]])
            if np.linalg.norm(error) < 1e-4:
                break
            lam, mu = 1e-3, 2e-3
            A = J.T @ J + (lam + mu) * np.eye(len(cadena))
            dq = np.linalg.solve(A, J.T @ error + mu * (nominal - q))
            q = q + np.clip(dq, -0.2, 0.2)
            q = np.array([min(rg[1], max(rg[0], v)) for v, rg in zip(q, rangos)])
        return {idx: float(v) for idx, v in zip(cadena, q)}

    # ------------------------------------------------------------ geometry
    def pose_en_abanico(self, asiento: int, k: int, n: int, d=None):
        """World position and rotation of card k of a fan of n (held by both hands,
        carried by the left one)."""
        d = self.data if d is None else d
        rot0 = orientacion_abanico(asiento)
        r = self.robots[asiento]
        pivote = d.site_xpos[r.sitio_izq] + r.offset_abanico
        giro = -(k - (n - 1) / 2) * APERTURA_ABANICO
        c, s = math.cos(giro), math.sin(giro)
        en_plano = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        rot = rot0 @ en_plano
        centro = pivote + rot[:, 1] * C.CARTA_ALTO * 0.36 + rot0[:, 2] * 0.0016 * k
        return centro, rot

    # --------------------------------------------------------------- arms
    def ajustar_articulaciones(self, objetivo: list, actual, asiento: int = 1) -> dict[int, float]:
        """Overrides `asiento`'s arm targets for holding and playing. Returns exact joints.

        `objetivo` (the gesture/seated targets, indexed like the robot's 29
        joints) is edited in place for the two-hand holding pose, which then
        blends in with the normal smoothing. The returned joints come from the
        play animation and are written as is.

        While holding, a gesture cannot wave an arm (it would drop the cards):
        if it raises the arms, both lift together with the fan instead, up to
        `ELEVACION_GESTO` of shoulder pitch.
        """
        r = self.robots.get(asiento)
        if not self._calibrado or r is None:
            return {}
        estado = self.mesa.estado()
        sostiene = any(c.asiento == asiento and c.lugar == "mano" for c in estado)
        jugada = self.mesa.jugada_activa(asiento)
        elevar = 0.0
        if r.sentado:
            elevar = min(objetivo[G1_L_SHOULDER_P] - r.sentado[G1_L_SHOULDER_P],
                         objetivo[G1_R_SHOULDER_P] - r.sentado[G1_R_SHOULDER_P])
            elevar = max(-ELEVACION_GESTO, min(0.0, elevar))
        if sostiene or (jugada and jugada[0] in ("alcanzar", "llevar")):
            for idx, v in r.pose_sostener.items():
                objetivo[idx] = v
            objetivo[G1_L_SHOULDER_P] += elevar
        if sostiene:
            # The free hand goes back to the fan (also where "volver" ends).
            for idx, v in r.pose_sostener_der.items():
                objetivo[idx] = v
            objetivo[G1_R_SHOULDER_P] += elevar
        if jugada is None or jugada[0] == "deslizar":
            r.fase_previa = None
            return {}
        fase, p, indice, baza = jugada
        clave = (fase, indice, baza)
        if clave != r.fase_previa:
            r.inicio_fase = {idx: float(actual[idx]) for idx in BRAZO_DER}
            r.fase_previa = clave
            if fase == "alcanzar":
                carta = next(c for c in estado if c.cuerpo == C.cuerpo(asiento, indice))
                r.abanico_jugada = (carta.indice_abanico, max(1, carta.cartas_en_abanico))
        t = suave(p)
        if fase == "alcanzar":
            fin = r.pose_alcanzar.get(r.abanico_jugada, next(iter(r.pose_alcanzar.values())))
            return self._mezclar(r.inicio_fase, fin, t)
        if fase == "llevar":
            inicio = r.pose_alcanzar.get(r.abanico_jugada, r.inicio_fase)
            medio = r.pose_elevar[baza]
            if p < 0.5:
                return self._mezclar(inicio, medio, suave(p * 2))
            return self._mezclar(medio, r.pose_apoyar[baza], suave(p * 2 - 1))
        # volver: from the table back to whatever the arm would otherwise do.
        reposo = {idx: objetivo[idx] for idx in BRAZO_DER}
        return self._mezclar(r.pose_apoyar[baza], reposo, t)

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
            r = self.robots.get(c.asiento)
            if r is not None and r.enganche and r.enganche["cuerpo"] == c.cuerpo:
                r.enganche = None
            return destino, rot_mesa
        p = c.progreso
        if c.fase == "deslizar":
            origen = np.array(C.borde_humano(c.asiento))
            pos = origen + (destino - origen) * suave(p)
            pos[2] += 0.05 * math.sin(math.pi * p) * (1 - p)
            return pos, rot_mesa
        # llevar: the card rides on the right palm, then settles in its slot.
        r = self.robots[c.asiento]
        mano = self.data.site_xpos[r.sitio_der].copy()
        if r.enganche is None or r.enganche["cuerpo"] != c.cuerpo:
            pos0, rot0 = self.pose_en_abanico(c.asiento, *r.abanico_jugada)
            r.enganche = {"cuerpo": c.cuerpo, "offset": pos0 - mano,
                          "q0": _mat_a_quat(self.mj, rot0)}
        e = r.enganche
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
