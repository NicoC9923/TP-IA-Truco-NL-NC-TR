# Robots Unitree (G1 / Go2) en MuJoCo

Simulador de los modelos oficiales de Unitree con una API de alto nivel en
Python. Proyecto personal.

## Estructura

| Carpeta | Que hay |
|---|---|
| `entorno/sim/` | el simulador: MuJoCo, transporte (socket local o DDS), API `Robot` |
| `entorno/sim/unitree_mujoco/` | modelos oficiales G1 y Go2 (BSD 3-Clause, ver su `LICENSE`) |
| `proyecto/` | tu codigo. `from robot import Robot` |

## Arrancar

Una vez por máquina, doble click en `instalar.bat` (o `./instalar.sh` en
Linux/macOS). Instala lo que lista `requirements.txt` (MuJoCo para la vista 3D,
NumPy, OpenCV para la cámara y Pillow para dibujar la baraja) y después revisa el entorno. Es lo mismo que:

```bash
py -3 -m pip install --user -r requirements.txt
```

Los lanzadores nunca instalan nada: si falta algo, avisan y dicen qué correr.
Sin OpenCV el Truco funciona igual, pero la cámara queda desactivada.
Sin MuJoCo la web del Truco funciona igual, pero no hay vista 3D: el servidor
avisa al arrancar con el comando para instalarlo.

Cada vez, en dos terminales:

```bash
sim --robot g1                   # o doble click en sim.bat (abre el G1); deja la ventana abierta
jugar_truco.bat                  # abre la interfaz web de Truco + Voz en http://127.0.0.1:8080
```

O para correr el ejemplo minimo por consola:

```bash
cd proyecto && py -3 ejemplo.py
```

`sim.bat` (Windows) y `./sim.sh` (Linux/macOS) pasan cualquier flag al
simulador. En Linux/macOS, `python3` en vez de `py -3`. `sim --solo-revisar`
chequea el entorno sin abrir nada.

En Windows, si `import mujoco` falla con *DLL load failed*, falta el
[Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

## Limites

Por defecto **no hay ninguno**. Se ponen al abrir el simulador y el cliente
los adopta al conectar:

```bash
py -3 -m sim --robot go2 --vmax 1.0 --wmax 2.0 --tmax 30
```

O del lado cliente: `Robot(limites=Limites(velocidad_max=0.5))`.

Lo que se conserva siempre: cada `Move()` vence en ~1 s (el cliente lo refresca
cada 0.1 s), asi que si tu programa muere el robot frena solo.

## Estado actual del simulador

La capa de alto nivel es **cinematica**: gravedad en cero, `mj_forward`, la
base se desliza a la velocidad pedida y las patas se animan. No hay control
articular ni fisica todavia. `robot.telemetria()` devuelve el `q`/`dq` real
de cada articulacion del modelo.

Flags: `--sin-ventana` (modo consola), `--dds` (SDK real de Unitree, solo
Linux + Python 3.10), `--puerto`, `--solo-revisar`.

## Truco cards in 3D

The deck in `proyecto/cartas/` is drawn by code (`py -3 proyecto/generar_baraja.py`);
the same PNGs texture the cards on the 3D table. Ottoman holds its cards fanned in
the left hand and lays each one on the table with the right hand; your cards
appear on your side as you play them. In the viewer, **F7** cycles the cameras
(`general`, `detras_robot_1`, `detras_robot_2`); against a human only `general`
is available, and the robot's hand shows card backs. Offscreen screenshots:
`cd entorno && py -3 -m sim.capturas ../docs/screenshots/step1 jpg`. See ADR 0067.

## Phase 2 network dependency

The phase 2 network and policy require PyTorch on CPU. Install it separately with
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
The game, phase 1, and the phase 2 observation encoder do not require PyTorch.
