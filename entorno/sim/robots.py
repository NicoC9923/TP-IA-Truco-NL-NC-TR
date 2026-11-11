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
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# Los modelos oficiales vienen en el repo. Solo se buscan aca, para que nunca
# se cargue otra copia que haya quedado en la maquina.
# The card PNGs the web serves are also the 3D scene's textures.
DIR_CARTAS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "proyecto", "cartas")

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

ESCENA_MESA_TRUCO = """<mujoco model="{modelo} - mesa de truco">
  <!-- Generado por sim/robots.py. Incluye el modelo oficial g1_29dof.xml y
       agrega las sillas, la mesa de Truco con paño verde, los naipes
       texturados, las camaras y el segundo G1 (poncho celeste) del asiento 2,
       que solo se sienta a la mesa en el modo robot contra robot. -->
  <include file="{incluye}"/>

  <statistic center="0.28 0 0.6" extent="1.8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-135" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
{cartas_assets}
    <model name="rival" file="{rival}"/>
  </asset>

  <worldbody>
    <light pos="0.4 0 2.0" dir="0 0 -1" directional="true"/>
    <light pos="0.6 -0.6 1.8" dir="0 0.5 -1" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
{camaras}
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- Silla de madera donde se sienta Ottoman -->
    <body name="silla_ottoman" pos="-0.04 0 0">
      <geom name="silla_asiento" type="box" pos="0 0 0.46" size="0.18 0.22 0.025" rgba="0.38 0.24 0.13 1"/>
      <geom name="silla_respaldo" type="box" pos="-0.16 0 0.72" size="0.022 0.21 0.23" rgba="0.38 0.24 0.13 1"/>
      <geom name="silla_pata1" type="cylinder" pos="0.14 0.18 0.22" size="0.02 0.22" rgba="0.22 0.15 0.08 1"/>
      <geom name="silla_pata2" type="cylinder" pos="0.14 -0.18 0.22" size="0.02 0.22" rgba="0.22 0.15 0.08 1"/>
      <geom name="silla_pata3" type="cylinder" pos="-0.14 0.18 0.22" size="0.02 0.22" rgba="0.22 0.15 0.08 1"/>
      <geom name="silla_pata4" type="cylinder" pos="-0.14 -0.18 0.22" size="0.02 0.22" rgba="0.22 0.15 0.08 1"/>
    </body>

    <!-- Mesa de Truco con paño verde, cartas, mate criollo y porotos de tanteo -->
    <body name="mesa_truco" pos="0.60 0 0">
      <geom name="mesa_tablero" type="box" pos="0 0 0.64" size="0.32 0.52 0.025" rgba="0.48 0.30 0.16 1"/>
      <geom name="mesa_pano" type="box" pos="0 0 0.667" size="0.26 0.44 0.004" rgba="0.12 0.44 0.23 1"/>
      <!-- El mazo: un taco de canto crema con nuestro dorso arriba -->
      <geom name="mazo_cartas" type="box" pos="0.02 0.33 0.6815" size="0.0445 0.0677 0.0105" rgba="0.93 0.91 0.85 1"/>
      <geom name="mazo_dorso" type="mesh" mesh="malla_carta" pos="0.02 0.33 0.6925" material="mat_dorso" contype="0" conaffinity="0"/>
      <!-- Mate criollo con virola de plata y bombilla -->
      <geom name="mate_base" type="cylinder" pos="-0.22 -0.40 0.676" size="0.022 0.006" rgba="0.25 0.15 0.08 1"/>
      <geom name="mate_calabaza" type="ellipsoid" pos="-0.22 -0.40 0.706" size="0.034 0.034 0.032" rgba="0.45 0.24 0.11 1"/>
      <geom name="mate_virola" type="cylinder" pos="-0.22 -0.40 0.733" size="0.028 0.005" rgba="0.88 0.90 0.94 1"/>
      <geom name="mate_yerba" type="cylinder" pos="-0.22 -0.40 0.736" size="0.024 0.003" rgba="0.26 0.42 0.16 1"/>
      <geom name="mate_bombilla" type="capsule" pos="-0.235 -0.40 0.758" euler="0 -0.45 0" size="0.004 0.028" rgba="0.90 0.92 0.96 1"/>
      <!-- Termo / Pava criolla al costado -->
      <geom name="termo_cuerpo" type="cylinder" pos="0.22 -0.42 0.745" size="0.036 0.075" rgba="0.72 0.74 0.78 1"/>
      <geom name="termo_tapa" type="cylinder" pos="0.22 -0.42 0.828" size="0.033 0.012" rgba="0.15 0.15 0.15 1"/>
      <geom name="termo_pico" type="cylinder" pos="0.22 -0.42 0.844" size="0.012 0.008" rgba="0.82 0.12 0.12 1"/>
      <!-- Porotos de tanteador sobre el paño -->
      <geom name="poroto_1" type="ellipsoid" pos="-0.18 0.28 0.674" size="0.009 0.006 0.004" rgba="0.88 0.78 0.56 1"/>
      <geom name="poroto_2" type="ellipsoid" pos="-0.16 0.30 0.674" size="0.009 0.006 0.004" rgba="0.88 0.78 0.56 1"/>
      <geom name="poroto_3" type="ellipsoid" pos="-0.19 0.31 0.674" size="0.009 0.006 0.004" rgba="0.88 0.78 0.56 1"/>
      <geom name="mesa_pata1" type="cylinder" pos="-0.27 0.46 0.31" size="0.028 0.31" rgba="0.32 0.20 0.10 1"/>
      <geom name="mesa_pata2" type="cylinder" pos="-0.27 -0.46 0.31" size="0.028 0.31" rgba="0.32 0.20 0.10 1"/>
      <geom name="mesa_pata3" type="cylinder" pos="0.27 0.46 0.31" size="0.028 0.31" rgba="0.32 0.20 0.10 1"/>
      <geom name="mesa_pata4" type="cylinder" pos="0.27 -0.46 0.31" size="0.028 0.31" rgba="0.32 0.20 0.10 1"/>
    </body>

    <!-- Asiento 2, enfrente: el humano, o el segundo G1 contra otro robot -->
    <body name="silla_asiento_2" pos="1.24 0 0" euler="0 0 3.14159">
      <geom name="silla2_asiento" type="box" pos="0 0 0.46" size="0.18 0.22 0.025" rgba="0.38 0.24 0.13 1"/>
      <geom name="silla2_respaldo" type="box" pos="-0.16 0 0.72" size="0.022 0.21 0.23" rgba="0.38 0.24 0.13 1"/>
      <geom name="silla2_pata1" type="cylinder" pos="0.14 0.18 0.22" size="0.02 0.22" rgba="0.22 0.15 0.08 1"/>
      <geom name="silla2_pata2" type="cylinder" pos="0.14 -0.18 0.22" size="0.02 0.22" rgba="0.22 0.15 0.08 1"/>
      <geom name="silla2_pata3" type="cylinder" pos="-0.14 0.18 0.22" size="0.02 0.22" rgba="0.22 0.15 0.08 1"/>
      <geom name="silla2_pata4" type="cylinder" pos="-0.14 -0.18 0.22" size="0.02 0.22" rgba="0.22 0.15 0.08 1"/>
    </body>

    <!-- Cartas del Truco: cuerpos mocap que mueve sim/escena_cartas.py -->
{cartas_cuerpos}

    <!-- Segundo G1 (asiento 2). Cinematico: sim/arrancar.py escribe su pose y lo
         guarda bajo el piso cuando juega un humano. -->
    <frame pos="1.24 0 0" euler="0 0 3.14159">
      <attach model="rival" body="pelvis" prefix="{prefijo_rival}"/>
    </frame>
  </worldbody>
</mujoco>
"""


def _agregar_geom_gaucho(body: ET.Element, **atributos: str) -> None:
    """Agrega un geom decorativo (sin masa ni colisión) a un body del robot."""
    attrs = {
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
        "density": "0",
    }
    attrs.update(atributos)
    ET.SubElement(body, "geom", attrs)


def _generar_modelo_gaucho(ruta_original: str) -> str:
    """Genera g1_29dof_gaucho.xml vistiendo a Ottoman de gaucho criollo para el Truco."""
    carpeta = os.path.dirname(ruta_original)
    nombre_gaucho = "g1_29dof_gaucho.xml"
    ruta_gaucho = os.path.join(carpeta, nombre_gaucho)

    tree = ET.parse(ruta_original)
    root = tree.getroot()

    # 1. Colorear las mallas base del cuerpo para que parezcan tela de camisa criolla,
    #    bombachas de campo y alpargatas (respetando el visor negro de la cabeza y las manos)
    colores_malla = {
        "torso_link": "0.93 0.90 0.82 1",              # Camisa criolla color crema
        "waist_yaw_link": "0.78 0.10 0.10 1",          # Faja pampa roja
        "waist_roll_link": "0.78 0.10 0.10 1",
        "waist_support_link": "0.78 0.10 0.10 1",
        "pelvis": "0.58 0.50 0.38 1",                  # Bombacha de campo beige/tierra
        "pelvis_contour_link": "0.28 0.16 0.08 1",     # Tirador de cuero criollo
        "left_shoulder_pitch_link": "0.48 0.12 0.10 1",# Hombrera del chaleco/poncho bordó
        "right_shoulder_pitch_link": "0.48 0.12 0.10 1",
        "left_shoulder_roll_link": "0.93 0.90 0.82 1", # Mangas de camisa criolla
        "right_shoulder_roll_link": "0.93 0.90 0.82 1",
        "left_shoulder_yaw_link": "0.93 0.90 0.82 1",
        "right_shoulder_yaw_link": "0.93 0.90 0.82 1",
        "left_elbow_link": "0.93 0.90 0.82 1",
        "right_elbow_link": "0.93 0.90 0.82 1",
        "left_wrist_roll_link": "0.48 0.12 0.10 1",    # Puños bordó
        "right_wrist_roll_link": "0.48 0.12 0.10 1",
        "left_hip_pitch_link": "0.58 0.50 0.38 1",     # Bombachas de campo gauchas
        "right_hip_pitch_link": "0.58 0.50 0.38 1",
        "left_hip_roll_link": "0.58 0.50 0.38 1",
        "right_hip_roll_link": "0.58 0.50 0.38 1",
        "left_hip_yaw_link": "0.58 0.50 0.38 1",
        "right_hip_yaw_link": "0.58 0.50 0.38 1",
        "left_knee_link": "0.58 0.50 0.38 1",
        "right_knee_link": "0.58 0.50 0.38 1",
        "left_ankle_pitch_link": "0.18 0.12 0.08 1",   # Botas / Alpargatas criollas
        "right_ankle_pitch_link": "0.18 0.12 0.08 1",
        "left_ankle_roll_link": "0.15 0.17 0.22 1",
        "right_ankle_roll_link": "0.15 0.17 0.22 1",
    }

    for geom in root.iter("geom"):
        mesh_name = geom.attrib.get("mesh", "")
        if mesh_name in colores_malla:
            geom.set("rgba", colores_malla[mesh_name])

    # 2. Agregar prendas 3D gauchescas adheridas a cada articulación
    bodies = {b.attrib.get("name"): b for b in root.iter("body") if "name" in b.attrib}

    torso = bodies.get("torso_link")
    if torso is not None:
        # --- SOMBRERO CHAMBERGO GAUCHO (con cinta punzó y hebilla dorada) ---
        _agregar_geom_gaucho(
            torso, name="chambergo_ala", type="cylinder",
            pos="0.002 0 0.448", euler="0 -0.07 0", size="0.148 0.0055",
            rgba="0.13 0.11 0.10 1",
        )
        _agregar_geom_gaucho(
            torso, name="chambergo_copa", type="cylinder",
            pos="-0.001 0 0.488", euler="0 -0.07 0", size="0.076 0.038",
            rgba="0.15 0.12 0.11 1",
        )
        _agregar_geom_gaucho(
            torso, name="chambergo_tope", type="ellipsoid",
            pos="-0.003 0 0.524", euler="0 -0.07 0", size="0.075 0.075 0.015",
            rgba="0.15 0.12 0.11 1",
        )
        _agregar_geom_gaucho(
            torso, name="chambergo_cinta", type="cylinder",
            pos="0.001 0 0.461", euler="0 -0.07 0", size="0.0785 0.010",
            rgba="0.80 0.10 0.10 1",
        )
        _agregar_geom_gaucho(
            torso, name="chambergo_hebilla", type="box",
            pos="0.079 0 0.465", euler="0 -0.07 0", size="0.004 0.018 0.009",
            rgba="0.94 0.78 0.22 1",
        )

        # --- PAÑUELO GAUCHO ROJO PUNZÓ AL CUELLO CON PASADOR DE PLATA ---
        _agregar_geom_gaucho(
            torso, name="panuelo_cuello", type="cylinder",
            pos="0.008 0 0.278", size="0.066 0.017",
            rgba="0.82 0.10 0.10 1",
        )
        _agregar_geom_gaucho(
            torso, name="panuelo_nudo", type="ellipsoid",
            pos="0.074 0 0.264", size="0.018 0.024 0.018",
            rgba="0.86 0.12 0.12 1",
        )
        _agregar_geom_gaucho(
            torso, name="panuelo_pasador_plata", type="cylinder",
            pos="0.079 0 0.253", euler="0 0.35 0", size="0.014 0.006",
            rgba="0.90 0.92 0.95 1",
        )
        _agregar_geom_gaucho(
            torso, name="panuelo_punta_izq", type="capsule",
            pos="0.084 0.022 0.214", euler="0.30 0.18 0", size="0.012 0.036",
            rgba="0.82 0.10 0.10 1",
        )
        _agregar_geom_gaucho(
            torso, name="panuelo_punta_der", type="capsule",
            pos="0.084 -0.022 0.214", euler="-0.30 0.18 0", size="0.012 0.036",
            rgba="0.82 0.10 0.10 1",
        )

        # --- CHALECO / PONCHO CRIOLLO CON GUARDAS PAMPA ---
        _agregar_geom_gaucho(
            torso, name="camisa_pechera", type="box",
            pos="0.012 0 0.152", size="0.074 0.042 0.105",
            rgba="0.95 0.93 0.86 1",
        )
        _agregar_geom_gaucho(
            torso, name="chaleco_frente_izq", type="box",
            pos="0.010 0.064 0.150", size="0.078 0.046 0.112",
            rgba="0.46 0.11 0.09 1",
        )
        _agregar_geom_gaucho(
            torso, name="chaleco_frente_der", type="box",
            pos="0.010 -0.064 0.150", size="0.078 0.046 0.112",
            rgba="0.46 0.11 0.09 1",
        )
        _agregar_geom_gaucho(
            torso, name="chaleco_espalda", type="box",
            pos="-0.032 0 0.150", size="0.038 0.108 0.112",
            rgba="0.46 0.11 0.09 1",
        )
        # Franjas de Guarda Pampa (crema con rombos oscuros)
        _agregar_geom_gaucho(
            torso, name="guarda_pampa_izq", type="box",
            pos="0.088 0.056 0.150", size="0.003 0.015 0.108",
            rgba="0.93 0.86 0.70 1",
        )
        _agregar_geom_gaucho(
            torso, name="guarda_pampa_der", type="box",
            pos="0.088 -0.056 0.150", size="0.003 0.015 0.108",
            rgba="0.93 0.86 0.70 1",
        )
        for idx_z, z_rombo in enumerate((0.215, 0.150, 0.085)):
            _agregar_geom_gaucho(
                torso, name=f"rombo_pampa_izq_{idx_z}", type="box",
                pos=f"0.090 0.056 {z_rombo}", euler="0.785 0 0", size="0.003 0.009 0.009",
                rgba="0.14 0.11 0.09 1",
            )
            _agregar_geom_gaucho(
                torso, name=f"rombo_pampa_der_{idx_z}", type="box",
                pos=f"0.090 -0.056 {z_rombo}", euler="0.785 0 0", size="0.003 0.009 0.009",
                rgba="0.14 0.11 0.09 1",
            )

        # --- FAJA PAMPA TEJIDA, TIRADOR DE CUERO Y RASTRA GAUCHA DE PLATA Y ORO ---
        _agregar_geom_gaucho(
            torso, name="faja_pampa", type="cylinder",
            pos="0.006 0 0.018", size="0.079 0.030",
            rgba="0.80 0.10 0.10 1",
        )
        _agregar_geom_gaucho(
            torso, name="tirador_cuero", type="cylinder",
            pos="0.007 0 0.018", size="0.081 0.016",
            rgba="0.28 0.16 0.08 1",
        )
        _agregar_geom_gaucho(
            torso, name="rastra_plata_central", type="cylinder",
            pos="0.087 0 0.018", euler="0 1.5708 0", size="0.028 0.005",
            rgba="0.88 0.90 0.94 1",
        )
        _agregar_geom_gaucho(
            torso, name="rastra_flor_oro", type="cylinder",
            pos="0.091 0 0.018", euler="0 1.5708 0", size="0.013 0.004",
            rgba="0.94 0.78 0.22 1",
        )
        for idx_m, y_mon in enumerate((-0.062, -0.038, 0.038, 0.062)):
            _agregar_geom_gaucho(
                torso, name=f"rastra_moneda_{idx_m}", type="cylinder",
                pos=f"0.081 {y_mon} 0.018", euler="0 1.5708 0", size="0.009 0.004",
                rgba="0.88 0.90 0.94 1",
            )

    # Mangas criollas abullonadas en brazos y antebrazos
    for lado in ("left", "right"):
        sh_yaw = bodies.get(f"{lado}_shoulder_yaw_link")
        if sh_yaw is not None:
            _agregar_geom_gaucho(
                sh_yaw, name=f"manga_brazo_{lado}", type="cylinder",
                pos="0.006 0 -0.036", size="0.043 0.055",
                rgba="0.93 0.90 0.82 1",
            )
        elbow = bodies.get(f"{lado}_elbow_link")
        if elbow is not None:
            _agregar_geom_gaucho(
                elbow, name=f"manga_antebrazo_{lado}", type="capsule",
                pos="0.045 0 -0.016", euler="0 1.35 0", size="0.036 0.034",
                rgba="0.93 0.90 0.82 1",
            )

    # Bombachas de campo anchas en muslos y pantorrillas + alpargatas con suela de yute
    for lado in ("left", "right"):
        hip_yaw = bodies.get(f"{lado}_hip_yaw_link")
        if hip_yaw is not None:
            _agregar_geom_gaucho(
                hip_yaw, name=f"bombacha_muslo_{lado}", type="ellipsoid",
                pos="-0.025 0 -0.105", size="0.056 0.064 0.120",
                rgba="0.58 0.50 0.38 1",
            )
        knee = bodies.get(f"{lado}_knee_link")
        if knee is not None:
            _agregar_geom_gaucho(
                knee, name=f"bombacha_pierna_{lado}", type="ellipsoid",
                pos="0.010 0 -0.125", size="0.052 0.058 0.130",
                rgba="0.58 0.50 0.38 1",
            )
            _agregar_geom_gaucho(
                knee, name=f"bota_cana_{lado}", type="cylinder",
                pos="0.006 0 -0.245", size="0.036 0.035",
                rgba="0.20 0.13 0.08 1",
            )
        ankle = bodies.get(f"{lado}_ankle_roll_link")
        if ankle is not None:
            _agregar_geom_gaucho(
                ankle, name=f"alpargata_capellada_{lado}", type="box",
                pos="0.035 0 -0.012", size="0.078 0.040 0.015",
                rgba="0.15 0.17 0.22 1",
            )
            _agregar_geom_gaucho(
                ankle, name=f"alpargata_yute_{lado}", type="box",
                pos="0.035 0 -0.026", size="0.082 0.043 0.007",
                rgba="0.82 0.72 0.50 1",
            )

    # Sitios en las palmas: de aca cuelgan las cartas que el robot sostiene
    # o lleva a la mesa (sim/escena_cartas.py). No son visibles.
    for lado, nombre in (("left", "mano_izq"), ("right", "mano_der")):
        muneca = bodies.get(f"{lado}_wrist_yaw_link")
        if muneca is not None:
            ET.SubElement(muneca, "site", {"name": nombre, "pos": "0.10 0 0",
                                           "size": "0.008", "group": "5"})

    tree.write(ruta_gaucho, encoding="utf-8")
    return nombre_gaucho


PREFIJO_RIVAL = "r2_"     # names of the second G1's bodies, joints and sites

# The rival's poncho: the reds of Ottoman's outfit turned light blue and navy.
_COLORES_RIVAL = {
    "0.48 0.12 0.10 1": "0.42 0.68 0.92 1",     # hombreras y punos
    "0.46 0.11 0.09 1": "0.42 0.68 0.92 1",     # chaleco / poncho
    "0.78 0.10 0.10 1": "0.16 0.30 0.62 1",     # faja
    "0.80 0.10 0.10 1": "0.16 0.30 0.62 1",     # faja, cinta del sombrero
    "0.82 0.10 0.10 1": "0.93 0.95 0.98 1",     # panuelo blanco
    "0.86 0.12 0.12 1": "0.93 0.95 0.98 1",
}


def _generar_modelo_rival(carpeta: str, nombre_gaucho: str) -> str:
    """Genera g1_29dof_gaucho_rival.xml: el mismo gaucho con poncho celeste.

    Sin actuadores ni sensores: es cinematico, y asi el modelo sigue teniendo los
    29 motores de Ottoman que esperan el bridge DDS y la telemetria.
    """
    nombre = "g1_29dof_gaucho_rival.xml"
    tree = ET.parse(os.path.join(carpeta, nombre_gaucho))
    root = tree.getroot()
    for etiqueta in ("actuator", "sensor", "keyframe"):
        for e in root.findall(etiqueta):
            root.remove(e)
    for geom in root.iter("geom"):
        color = geom.attrib.get("rgba")
        if color in _COLORES_RIVAL:
            geom.set("rgba", _COLORES_RIVAL[color])
    root.set("model", "g1_29dof_gaucho_rival")
    tree.write(os.path.join(carpeta, nombre), encoding="utf-8")
    return nombre


def _generar_escena_limpia(destino: str, incluye: str) -> str:
    """Escribe la escena al lado del modelo oficial."""
    es_mesa_truco = "mesa_truco" in os.path.basename(destino)
    plantilla = ESCENA_MESA_TRUCO if es_mesa_truco else ESCENA_LIMPIA
    archivo_incluido = incluye
    if es_mesa_truco:
        ruta_original = os.path.join(os.path.dirname(destino), incluye)
        if os.path.exists(ruta_original):
            archivo_incluido = _generar_modelo_gaucho(ruta_original)
    extras = {}
    if es_mesa_truco:
        from . import cartas
        ruta = cartas.ruta_cartas_relativa(os.path.dirname(destino), DIR_CARTAS)
        carpeta = os.path.dirname(destino)
        rival = (_generar_modelo_rival(carpeta, archivo_incluido)
                 if os.path.exists(os.path.join(carpeta, archivo_incluido)) else archivo_incluido)
        extras = {"cartas_assets": cartas.xml_assets(ruta),
                  "cartas_cuerpos": cartas.xml_cuerpos(),
                  "camaras": cartas.xml_camaras(),
                  "rival": rival, "prefijo_rival": PREFIJO_RIVAL}
    contenido = plantilla.format(
        modelo=os.path.splitext(incluye)[0], incluye=archivo_incluido, **extras)
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
            if self.incluye:
                modelo = os.path.join(base, os.path.dirname(self.escena),
                                      self.incluye)
                if os.path.exists(modelo):
                    return _generar_escena_limpia(ruta, self.incluye)
            if os.path.exists(ruta):
                return ruta
        return None


# --- G1: humanoide de 29 grados de libertad -----------------------------
# Indices relativos (qpos real = 7 + indice)
G1_L_HIP_P, G1_L_KNEE, G1_L_ANKLE_P = 0, 3, 4
G1_R_HIP_P, G1_R_KNEE, G1_R_ANKLE_P = 6, 9, 10
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
    escena="g1/scene_mesa_truco.xml",
    incluye="g1_29dof.xml",
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
    # Sentado en la silla frente a la mesa de Truco (piernas a 90° y manos sobre la mesa)
    pose_sentado={
        G1_L_HIP_P: -1.42, G1_R_HIP_P: -1.42,
        G1_L_KNEE: 1.48, G1_R_KNEE: 1.48,
        G1_L_ANKLE_P: -0.05, G1_R_ANKLE_P: -0.05,
        G1_L_SHOULDER_P: -0.38, G1_R_SHOULDER_P: -0.38,
        G1_L_SHOULDER_R: 0.18, G1_R_SHOULDER_R: -0.18,
        G1_L_ELBOW: 0.15, G1_R_ELBOW: 0.15,
    },
    gestos={
        "repartiendo": {
            G1_WAIST_PITCH: 0.08,
            G1_L_SHOULDER_P: -0.65,
            G1_L_SHOULDER_R: 0.22,
            G1_L_SHOULDER_Y: -0.35,
            G1_L_ELBOW: -0.10,
            G1_R_SHOULDER_P: -0.75,
            G1_R_SHOULDER_R: -0.18,
            G1_R_SHOULDER_Y: 0.10,
            G1_R_ELBOW: 0.20,
        },
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
            G1_L_SHOULDER_P: -0.38,
            G1_L_SHOULDER_R: 0.22,
            G1_L_ELBOW: 0.18,
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
