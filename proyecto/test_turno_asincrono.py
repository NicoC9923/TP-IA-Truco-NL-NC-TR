"""Tests for the robot's asynchronous, human-paced turn in PartidaTruco."""

from __future__ import annotations

import random
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import voz_jugador as vj
from partida_truco import MENSAJE_ROBOT_OCUPADO, PartidaTruco
from ritmo import Ritmo

RAPIDO = Ritmo(pensar=(0.05, 0.05), pensar_grande=(0.08, 0.08), alcanzar=0.02, colocar=0.03,
               pausa_baza=0.01, pausa_mano=0.01)
LENTO = Ritmo(pensar=(0.5, 0.5), pensar_grande=(0.5, 0.5), alcanzar=0.05, colocar=0.05,
              pausa_baza=0.01, pausa_mano=0.01)


class RobotFalso:
    """Records the 3D hooks and gestures in the order they arrive."""

    def __init__(self):
        self.llamadas = []

    def repartir_cartas(self, mano, rival=None):
        self.llamadas.append(("repartir_cartas",))
        return True

    def tirar_carta(self, carta, asiento=1):
        self.llamadas.append(("tirar_carta", carta["valor"], carta["palo"], asiento))
        return True

    def modo_mesa(self, modo):
        return True

    def desafiar(self, esperar=True):
        self.llamadas.append(("desafiar",))

    def gesto_mazo(self, esperar=True):
        self.llamadas.append(("gesto_mazo",))


def esperar(condicion, limite=5.0):
    fin = time.time() + limite
    while time.time() < fin:
        if condicion():
            return True
        time.sleep(0.005)
    return False


class Base(unittest.TestCase):
    def crear(self, ritmo=RAPIDO, robot=None, semilla=11):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        partida = PartidaTruco(robot=robot, historial_path=str(Path(tmp.name) / "h.json"),
                               rng=random.Random(semilla), ritmo=ritmo, rng_voz=random.Random(5))
        self.addCleanup(partida.cancelar_turno_robot)
        return partida

    def robot_mano(self, partida):
        """Starts a hand in which the robot is mano (so it moves first)."""
        partida.mano_terminada = True
        partida.nueva_mano()

    def sin_cantos(self, partida):
        """The robot never calls: it only plays cards."""
        for nombre in ("decidir_iniciar_envido", "decidir_iniciar_truco"):
            parche = patch.object(partida.ia, nombre, return_value=(None, ""))
            parche.start()
            self.addCleanup(parche.stop)

    def libre(self, partida):
        return esperar(lambda: not partida.estado_publico()["robot"]["ocupado"])


class TestTurnoAsincrono(Base):
    def test_el_modo_por_defecto_es_instantaneo_y_sincrono(self):
        partida = PartidaTruco(historial_path=str(Path(tempfile.mkdtemp()) / "h.json"), rng=random.Random(1))
        self.assertTrue(partida.estado_publico()["robot"]["instantaneo"])
        self.robot_mano(partida)
        est = partida.estado_publico()
        self.assertFalse(est["robot"]["ocupado"])
        self.assertTrue(est["mesa"] or est["canto_pendiente"])

    def test_pensando_y_luego_actua(self):
        partida = self.crear(LENTO)
        self.sin_cantos(partida)
        self.robot_mano(partida)
        est = partida.estado_publico()
        self.assertEqual(est["robot"]["estado"], "pensando")
        self.assertTrue(est["robot"]["pensando"])
        self.assertEqual(est["mesa"], [])
        self.assertEqual(est["turno_carta"], "robot")
        self.assertEqual(est["cartas_robot_restantes"], 3)
        self.assertTrue(self.libre(partida))
        est = partida.estado_publico()
        self.assertEqual(est["robot"]["estado"], "ocioso")
        self.assertEqual(len(est["mesa"]), 1)
        self.assertEqual(est["mesa"][0]["quien"], "robot")
        self.assertEqual(est["cartas_robot_restantes"], 2)
        self.assertEqual(est["turno_carta"], "humano")

    def test_el_estado_congelado_no_adelanta_la_carta(self):
        partida = self.crear(LENTO)
        self.sin_cantos(partida)
        self.robot_mano(partida)
        # Through the whole thinking time (0.5 s) the card is not shown.
        for _ in range(8):
            est = partida.estado_publico()
            if est["robot"]["estado"] != "pensando":
                break
            self.assertEqual(est["mesa"], [])
            time.sleep(0.04)
        self.assertTrue(self.libre(partida))
        self.assertEqual(len(partida.estado_publico()["mesa"]), 1)

    def test_el_humano_no_puede_actuar_mientras_el_robot_piensa(self):
        partida = self.crear(LENTO)
        self.sin_cantos(partida)
        self.robot_mano(partida)
        for llamada in (
            lambda: partida.jugar_carta_humano(partida.cartas_humano[0]),
            lambda: partida.cantar_truco_humano(),
            lambda: partida.cantar_envido_humano("ENVIDO"),
            lambda: partida.irse_al_mazo_humano(),
            lambda: partida.responder_canto_humano("QUIERO"),
            lambda: partida.nueva_mano(),
        ):
            res = llamada()
            self.assertEqual(res.get("error"), MENSAJE_ROBOT_OCUPADO)
            self.assertEqual(res["estado"]["robot"]["estado"], "pensando")
        self.assertEqual(len(partida.cartas_humano_disponibles), 3)
        self.assertTrue(self.libre(partida))
        # Once the robot is done the human can play.
        res = partida.jugar_carta_humano(partida.cartas_humano[0])
        self.assertNotIn("error", res)

    def test_reiniciar_cancela_el_turno_pendiente(self):
        partida = self.crear(LENTO)
        self.sin_cantos(partida)
        self.robot_mano(partida)
        self.assertTrue(partida.estado_publico()["robot"]["pensando"])
        hilo = partida._hilo_robot
        partida.reiniciar_partido(15)
        est = partida.estado_publico()
        self.assertEqual(est["robot"]["estado"], "ocioso")
        self.assertEqual(est["turno_carta"], "humano")
        ultimo_id = max([v["id"] for v in est["voz"]] or [0])
        hilo.join(2)
        self.assertFalse(hilo.is_alive())
        time.sleep(0.7)
        est = partida.estado_publico()
        self.assertEqual(est["mesa"], [])
        self.assertEqual(est["cartas_robot_restantes"], 3)
        self.assertEqual(max([v["id"] for v in est["voz"]] or [0]), ultimo_id)

    def test_nueva_partida_tras_cancelar_vuelve_a_funcionar(self):
        partida = self.crear(RAPIDO)
        self.sin_cantos(partida)
        self.robot_mano(partida)
        partida.reiniciar_partido(15)
        self.robot_mano(partida)
        self.assertTrue(self.libre(partida))
        self.assertEqual(len(partida.estado_publico()["mesa"]), 1)

    def test_orden_de_la_mesa_3d_humano_antes_que_robot(self):
        robot = RobotFalso()
        partida = self.crear(LENTO, robot=robot)
        self.sin_cantos(partida)
        carta = partida.cartas_humano[0]
        res = partida.jugar_carta_humano(carta)
        self.assertTrue(res["estado"]["robot"]["pensando"])
        self.assertTrue(esperar(lambda: any(l[0] == "tirar_carta" for l in robot.llamadas)))
        # The human's card goes out at once, the robot's only after its think time.
        tiradas = [l for l in robot.llamadas if l[0] == "tirar_carta"]
        self.assertEqual(tiradas, [("tirar_carta", carta["valor"], carta["palo"], 2)])
        self.assertTrue(self.libre(partida))
        tiradas = [l for l in robot.llamadas if l[0] == "tirar_carta"]
        self.assertGreaterEqual(len(tiradas), 2)  # the robot may win the baza and play again
        self.assertEqual(tiradas[0][3], 2)
        self.assertEqual(tiradas[1][3], 1)

    def test_la_voz_sale_una_sola_vez_y_despues_de_pensar(self):
        partida = self.crear(LENTO)
        self.sin_cantos(partida)
        antes = partida.estado_publico()["voz"]
        partida.jugar_carta_humano(partida.cartas_humano[0])
        self.assertEqual(partida.estado_publico()["voz"], antes)
        self.assertTrue(self.libre(partida))
        ids = []
        for _ in range(5):
            ids.append([v["id"] for v in partida.estado_publico()["voz"]])
        self.assertTrue(all(i == ids[0] for i in ids))
        frases = [v for v in partida.estado_publico()["voz"] if v["accion"] == "carta"]
        self.assertLessEqual(len(frases), 1)
        self.assertEqual(len(set(ids[0])), len(ids[0]))

    def test_el_canto_del_robot_usa_el_catalogo_y_no_la_frase_de_la_ia(self):
        partida = self.crear(RAPIDO)
        with patch.object(partida.ia, "decidir_iniciar_envido", return_value=(None, "")), \
                patch.object(partida.ia, "decidir_iniciar_truco", return_value=("TRUCO", "Tengo 33 de envido y un 1 de espada")):
            self.robot_mano(partida)
            self.assertTrue(self.libre(partida))
        est = partida.estado_publico()
        self.assertEqual(est["canto_pendiente"]["canto"], "TRUCO")
        dichas = [v["frase"] for v in est["voz"]]
        self.assertEqual(dichas, ["¡Truco!"])
        self.assertEqual(est["voz"][0]["perfil"], vj.perfil_voz(1))

    def test_las_decisiones_grandes_piensan_mas(self):
        ritmo = Ritmo(pensar=(0.05, 0.05), pensar_grande=(0.4, 0.4), alcanzar=0.01, colocar=0.01,
                      pausa_baza=0, pausa_mano=0)
        for canto, minimo, maximo in (("TRUCO", 0.35, 5.0), (None, 0.0, 0.3)):
            partida = self.crear(ritmo)
            with patch.object(partida.ia, "decidir_iniciar_envido", return_value=(None, "")), \
                    patch.object(partida.ia, "decidir_iniciar_truco", return_value=(canto, "")):
                inicio = time.time()
                self.robot_mano(partida)
                self.assertTrue(self.libre(partida))
                duracion = time.time() - inicio
            self.assertTrue(minimo <= duracion <= maximo, (canto, duracion))

    def test_el_robot_responde_al_truco_del_humano_despues_de_pensar(self):
        partida = self.crear(LENTO)
        with patch.object(partida.ia, "decidir_iniciar_envido", return_value=(None, "")), \
                patch.object(partida.ia, "responder_canto_truco", return_value=("QUIERO", "")):
            res = partida.cantar_truco_humano()
            self.assertTrue(res["pensando"])
            self.assertEqual(partida.estado_publico()["robot"]["estado"], "pensando")
            self.assertEqual(partida.estado_publico()["voz"], [])
            self.assertTrue(self.libre(partida))
        est = partida.estado_publico()
        self.assertEqual(est["apuesta_truco"], "TRUCO")
        self.assertEqual([v["frase"] for v in est["voz"]], ["¡Quiero!"])

    def test_el_robot_se_va_al_mazo_con_frase_y_gesto(self):
        robot = RobotFalso()
        partida = self.crear(RAPIDO, robot=robot)
        with patch.object(partida.ia, "decidir_iniciar_envido", return_value=(None, "")), \
                patch.object(partida.ia, "responder_canto_truco", return_value=("NO_QUIERO", "")):
            partida.cantar_truco_humano()
            self.assertTrue(self.libre(partida))
        est = partida.estado_publico()
        self.assertTrue(est["mano_terminada"])
        self.assertEqual([v["frase"] for v in est["voz"]], ["No quiero."])
        self.assertTrue(esperar(lambda: ("gesto_mazo",) in robot.llamadas))

    def test_una_decision_rota_no_congela_la_partida(self):
        partida = self.crear(RAPIDO)
        with patch.object(partida, "_robot_tira_carta", side_effect=RuntimeError("boom")):
            self.sin_cantos(partida)
            self.robot_mano(partida)
            self.assertTrue(self.libre(partida))
        self.assertEqual(partida.estado_publico()["robot"]["estado"], "ocioso")

    def test_varios_pasos_encadenados_el_robot_sigue_jugando(self):
        """After winning a baza the robot moves again, each step with its own think."""
        partida = self.crear(RAPIDO)
        self.sin_cantos(partida)
        self.robot_mano(partida)
        self.assertTrue(self.libre(partida))
        for _ in range(3):
            if partida.mano_terminada:
                break
            res = partida.jugar_carta_humano(partida.cartas_humano_disponibles[0])
            self.assertNotIn("error", res)
            self.assertTrue(self.libre(partida))
        est = partida.estado_publico()
        self.assertTrue(est["mano_terminada"] or est["turno_carta"] == "humano")

    def test_el_servidor_comparte_el_lock_de_la_partida(self):
        from servidor_truco import crear_servidor
        partida = self.crear(RAPIDO)
        servidor = crear_servidor(partida, puerto=0, sin_robot=True)
        try:
            self.assertIs(servidor.lock, partida.lock)
        finally:
            servidor.server_close()

    def test_cambiar_la_velocidad_en_caliente(self):
        partida = self.crear(Ritmo.humano(1.0))
        partida.ritmo = partida.ritmo.con_velocidad(2.0)
        self.assertEqual(partida.estado_publico()["robot"]["velocidad"], 2.0)


class TestConcurrencia(Base):
    def test_sondeos_concurrentes_durante_el_turno(self):
        partida = self.crear(RAPIDO)
        self.sin_cantos(partida)
        errores = []

        def sondear():
            fin = time.time() + 0.6
            while time.time() < fin:
                try:
                    partida.estado_publico()
                except Exception as e:  # pragma: no cover
                    errores.append(e)

        hilos = [threading.Thread(target=sondear) for _ in range(4)]
        for h in hilos:
            h.start()
        self.robot_mano(partida)
        for h in hilos:
            h.join()
        self.assertEqual(errores, [])
        self.assertTrue(self.libre(partida))


if __name__ == "__main__":
    unittest.main()
