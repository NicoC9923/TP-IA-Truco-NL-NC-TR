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
NumPy, OpenCV para la cámara y Pillow para dibujar la baraja), además de PyTorch
para la IA de aprendizaje por refuerzo, y después revisa el entorno. En Windows,
los comandos equivalentes son:

```bash
py -3 -m pip install --user -r requirements.txt
py -3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
```

En Linux se usa `python3` con el mismo índice de CPU; en macOS se instala
PyTorch desde el índice predeterminado de pip. Si PyTorch ya se puede importar,
el instalador omite ese paso. Si falla su instalación, las otras tres IA siguen
disponibles.

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
the same PNGs texture the cards on the 3D table. Ottoman holds its fan of cards with
both hands, pulls each card out with the right hand and lays it on the table; your cards
appear on your side as you play them. In the viewer, **F7** cycles the cameras
(`general`, `detras_robot_1`, `detras_robot_2`); against a human only `general`
is available, and the robot's hand shows card backs. Offscreen screenshots:
`cd entorno && py -3 -m sim.capturas ../docs/screenshots/step1 jpg`. See ADR 0067.

## Spectator mode (two AIs)

With the game server running (`jugar_truco.bat`), open
<http://127.0.0.1:8080/espectador> (or **Modo espectador** in the main page's top
bar). Pick an AI for each seat (Refuerzo corrida 3 · gen 73, Utilidad adaptativa,
Genético publicado (gen 48) or Reactivo; the same one twice is fine), the length
(A 15, A 30, or a short match of 1 to 10 hands) and press **Empezar**. Controls:
Pausa / Seguir, Siguiente jugada (one action), speed ×0.5 / ×1 / ×2, Ver cartas
(both hands face up by default), Qué piensa cada IA (on-screen only, never
spoken), the 3D camera, and at the end Revancha or Cambiar IAs.

The match runs on the training engine (`proyecto/espectador.py`, ADR 0064), so the
AIs play exactly as they were measured, plus the envido rule of ADR 0068. With the
simulator open, a second G1 in a light-blue poncho sits in seat 2 and both robots
hold and play their cards; every camera is allowed (F7 cycles `general`,
`detras_robot_1`, `detras_robot_2`, `mesa`). Without MuJoCo the page works alone.
Short matches are also available against Ottoman from the **Corta** selector next to
A 15 / A 30 (ADR 0069). Two-robot screenshots:
`cd entorno && py -3 -m sim.capturas ../docs/screenshots/step2 jpg robots`.

## Phase 2 network dependency

The phase 2 network and policy require PyTorch. The one-time installers install
it automatically (using the CPU index on Windows and Linux), unless it is already
available. If installation fails, rerun the installer or use the commands above.
The game, phase 1, and the phase 2 observation encoder do not require PyTorch.
