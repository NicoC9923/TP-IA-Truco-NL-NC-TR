"""Los dos robots reales: donde esta su modelo oficial y como se anima cada uno.

Usamos los modelos OFICIALES de Unitree (repo unitreerobotics/unitree_mujoco).

En la capa de alto nivel el movimiento es CINEMATICO: escribimos la pose y
llamamos mj_forward, sin correr fisica. El G1 es un humanoide y SE CAE SOLO sin
un controlador de locomocion, que el simulador oficial no trae (vive en la PC
interna del robot y Unitree no lo publica).

La animacion de las patas es COSMETICA: el robot se desliza.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

# Los modelos oficiales vienen en el repo. Solo se buscan aca, para que nunca
# se cargue otra copia que haya quedado en la maquina.
UBICACIONES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "unitree_mujoco", "unitree_robots"),
]


ESCENA_LIMPIA = """<mujoco model="{modelo} - escena limpia">
  <!-- Generado por sim/robots.py. NO es un archivo oficial de Unitree.
       La escena oficial del Go2 trae 8 cajas de obstaculo, incluida una
       escalera, y dos vigas. Esta escena tiene el mismo robot y el mismo
       piso, sin esos obstaculos.
       Se regenera solo si se borra. No modifica ningun archivo oficial. -->
  <include file="{incluye}"/>

  <statistic center="0 0 0.1" extent="1.2"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def _generar_escena_limpia(destino: str, incluye: str) -> str:
    """Escribe la escena sin obstaculos al lado del modelo oficial."""
    contenido = ESCENA_LIMPIA.format(
        modelo=os.path.splitext(incluye)[0], incluye=incluye)
    with open(destino, "w", encoding="utf-8") as f:
        f.write(contenido)
    return destino


@dataclass
class Robot:
    clave: str
    nombre: str
    tipo: str
    escena: str          # ruta relativa dentro de unitree_robots/
    altura: float        # altura de la base al estar de pie, en metros
    incluye: str = ""    # si esta, generamos una escena limpia que lo incluya
    pose_de_pie: dict = field(default_factory=dict)   # indice art -> radianes
    # Indices (dentro de qpos, contando desde la primera articulacion) que se
    # animan al caminar: (indice, amplitud, desfase)
    marcha: list = field(default_factory=list)
    saludo: dict = field(default_factory=dict)
    pose_sentado: dict = field(default_factory=dict)
    gestos: dict = field(default_factory=dict)

    def ruta_escena(self) -> str | None:
        for base in UBICACIONES:
            ruta = os.path.join(base, self.escena)
            if os.path.exists(ruta):
                return ruta
            # Escena limpia: la generamos al lado del modelo oficial si falta.
            if self.incluye:
                modelo = os.path.join(base, os.path.dirname(self.escena),
                                      self.incluye)
                if os.path.exists(modelo):
                    return _generar_escena_limpia(ruta, self.incluye)
        return None


# --- G1: humanoide de 29 grados de libertad -----------------------------
# Indices relativos (qpos real = 7 + indice)
G1_L_HIP_P, G1_L_KNEE = 0, 3
G1_R_HIP_P, G1_R_KNEE = 6, 9
G1_WAIST_YAW, G1_WAIST_ROLL, G1_WAIST_PITCH = 12, 13, 14
G1_L_SHOULDER_P, G1_L_SHOULDER_R, G1_L_SHOULDER_Y, G1_L_ELBOW = 15, 16, 17, 18
G1_L_WRIST_R, G1_L_WRIST_P, G1_L_WRIST_Y = 19, 20, 21
G1_R_SHOULDER_P, G1_R_SHOULDER_R, G1_R_SHOULDER_Y, G1_R_ELBOW = 22, 23, 24, 25
G1_R_WRIST_R, G1_R_WRIST_P, G1_R_WRIST_Y = 26, 27, 28

# Nota cinematica del modelo oficial G1 29-DOF:
# En elbow_joint = 0 rad, el antebrazo apunta hacia adelante (+X, flexionado 90°).
# Valores positivos (~0.9 rad) extienden el brazo hacia abajo al costado del torso;
# valores negativos pliegan el antebrazo hacia arriba.
G1 = Robot(
    clave="g1",
    nombre="Unitree G1",
    tipo="humanoide",
    escena="g1/scene_29dof.xml",   # scene.xml tiene ~100 obstaculos del terrain_tool
    altura=0.793,
    pose_de_pie={
        G1_L_SHOULDER_P: 0.15, G1_R_SHOULDER_P: 0.15,
        G1_L_SHOULDER_R: 0.12, G1_R_SHOULDER_R: -0.12,
        G1_L_ELBOW: 0.90, G1_R_ELBOW: 0.90,
    },
    marcha=[
        (G1_L_HIP_P, 0.40, 0.0),
        (G1_R_HIP_P, 0.40, math.pi),
        (G1_L_KNEE, 0.45, 1.1),
        (G1_R_KNEE, 0.45, 1.1 + math.pi),
        (G1_L_SHOULDER_P, 0.30, math.pi),
        (G1_R_SHOULDER_P, 0.30, 0.0),
    ],
    saludo={
        G1_R_SHOULDER_P: -1.35,
        G1_R_SHOULDER_R: -0.55,
        G1_R_SHOULDER_Y: -0.30,
        G1_R_ELBOW: -0.35,
    },
    # Agachado (FSM Sit / Damp). No es un desplome: es una postura.
    pose_sentado={G1_L_HIP_P: -1.2, G1_R_HIP_P: -1.2,
                  G1_L_KNEE: 1.8, G1_R_KNEE: 1.8},
    gestos={
        "saludando": {
            G1_R_SHOULDER_P: -1.35,
            G1_R_SHOULDER_R: -0.55,
            G1_R_SHOULDER_Y: -0.30,
            G1_R_ELBOW: -0.35,
        },
        "saludo": {
            G1_R_SHOULDER_P: -1.35,
            G1_R_SHOULDER_R: -0.55,
            G1_R_SHOULDER_Y: -0.30,
            G1_R_ELBOW: -0.35,
        },
        "dar_la_mano": {
            G1_WAIST_PITCH: 0.06,
            G1_R_SHOULDER_P: -0.65,
            G1_R_SHOULDER_R: -0.10,
            G1_R_SHOULDER_Y: 0.0,
            G1_R_ELBOW: 0.35,
            G1_R_WRIST_R: 0.0,
            G1_R_WRIST_P: 0.0,
        },
        "festejando": {
            G1_L_SHOULDER_P: -1.75, G1_R_SHOULDER_P: -1.75,
            G1_L_SHOULDER_R: 0.55, G1_R_SHOULDER_R: -0.55,
            G1_L_SHOULDER_Y: 0.25, G1_R_SHOULDER_Y: -0.25,
            G1_L_ELBOW: 0.20, G1_R_ELBOW: 0.20,
        },
        "desafiando": {
            G1_WAIST_PITCH: 0.12,
            G1_R_SHOULDER_P: -1.05,
            G1_R_SHOULDER_R: -0.15,
            G1_R_ELBOW: 0.85,
            G1_L_SHOULDER_P: 0.15,
            G1_L_SHOULDER_R: 0.22,
            G1_L_ELBOW: 0.45,
        },
        "mazo": {
            G1_L_SHOULDER_P: -0.55, G1_R_SHOULDER_P: -0.55,
            G1_L_SHOULDER_R: 0.35, G1_R_SHOULDER_R: -0.35,
            G1_L_ELBOW: -0.35, G1_R_ELBOW: -0.35,
        },
    },
)

# --- Go2: cuadrupedo de 12 grados de libertad ---------------------------
GO2_FL_T, GO2_FL_C = 1, 2
GO2_FR_T, GO2_FR_C = 4, 5
GO2_RL_T, GO2_RL_C = 7, 8
GO2_RR_T, GO2_RR_C = 10, 11

# Pose de perro parado. Sin esto el modelo aparece con las patas estiradas.
_GO2_MUSLO, _GO2_PANTORRILLA = 0.80, -1.55

GO2 = Robot(
    clave="go2",
    nombre="Unitree Go2",
    tipo="cuadrupedo",
    # La escena oficial del Go2 trae una escalera y vigas. Usamos una escena
    # limpia que se genera sola. La del G1 ya viene limpia.
    escena="go2/scene_limpia.xml",
    incluye="go2.xml",
    altura=0.33,
    pose_de_pie={
        GO2_FL_T: _GO2_MUSLO, GO2_FR_T: _GO2_MUSLO,
        GO2_RL_T: _GO2_MUSLO, GO2_RR_T: _GO2_MUSLO,
        GO2_FL_C: _GO2_PANTORRILLA, GO2_FR_C: _GO2_PANTORRILLA,
        GO2_RL_C: _GO2_PANTORRILLA, GO2_RR_C: _GO2_PANTORRILLA,
    },
    # Trote: las diagonales van juntas (FL con RR, FR con RL).
    marcha=[
        (GO2_FL_T, 0.28, 0.0), (GO2_RR_T, 0.28, 0.0),
        (GO2_FR_T, 0.28, math.pi), (GO2_RL_T, 0.28, math.pi),
        (GO2_FL_C, 0.22, 0.0), (GO2_RR_C, 0.22, 0.0),
        (GO2_FR_C, 0.22, math.pi), (GO2_RL_C, 0.22, math.pi),
    ],
    # El perro "saluda" levantando la pata delantera izquierda.
    saludo={GO2_FL_T: -0.6, GO2_FL_C: -0.9},
    pose_sentado={GO2_FL_T: 1.3, GO2_FR_T: 1.3, GO2_RL_T: 1.3, GO2_RR_T: 1.3,
                  GO2_FL_C: -2.4, GO2_FR_C: -2.4, GO2_RL_C: -2.4, GO2_RR_C: -2.4},
)

ROBOTS = {"g1": G1, "go2": GO2}


def obtener(clave: str) -> Robot:
    c = clave.lower().strip()
    if c not in ROBOTS:
        raise ValueError(f"Robot desconocido: '{clave}'. Validos: g1, go2.")
    return ROBOTS[c]
