"""Ejemplo minimo. Abri antes INICIAR_SIMULADOR y despues: python ejemplo.py"""

from robot import Robot

robot = Robot()
robot.conectar()
try:
    robot.avanzar(velocidad=0.8, tiempo=2.0)
    robot.mover(vx=0.5, vyaw=0.8, tiempo=3.0)   # curva
    robot.saludar()
    print(robot.verificar_estado())
    motores = robot.telemetria()["motor_state"]
    print(f"{len(motores)} motores; q[0] = {motores[0]['q']:+.3f} rad")
finally:
    robot.desconectar()
