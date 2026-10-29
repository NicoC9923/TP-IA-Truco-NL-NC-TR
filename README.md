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

Una vez:

```bash
py -3 -m pip install --user mujoco
```

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
