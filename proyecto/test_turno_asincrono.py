"""Tests for the robot's asynchronous, human-paced turn in PartidaTruco."""

from __future__ import annotations

import random
import tempfile
import threading
import time
import unittest
from pathlib import Path
import unittest.mock
from unittest.mock import patch

import truco_reglas as tr
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

    def repartir(self, partida, humano, robot, robot_es_mano=False):
        """Deals a chosen hand: `humano` and `robot` are lists of (valor, palo)."""
        elegidas = list(humano) + list(robot)

        def ordenar(mazo):
            por_clave = {(c["valor"], c["palo"]): c for c in mazo}
            mazo[:] = [por_clave[k] for k in elegidas] + [c for c in mazo if (c["valor"], c["palo"]) not in elegidas]

        partida.mano_terminada = True
        partida.es_mano_humano = robot_es_mano  # nueva_mano switches the mano
        with patch.object(partida.rng, "shuffle", side_effect=ordenar):
            partida.nueva_mano()
        self.assertEqual(partida.es_mano_humano, not robot_es_mano)


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
        with patch.object(partida.ia, "elegir_carta_a_jugar", side_effect=RuntimeError("boom")):
            self.sin_cantos(partida)
            self.robot_mano(partida)
            self.assertTrue(self.libre(partida))
        est = partida.estado_publico()
        self.assertEqual(est["robot"]["estado"], "ocioso")
        # The robot fell back to a legal card, so the human can go on.
        self.assertTrue(est["turno_carta"] == "humano" or est["mano_terminada"], est["turno_carta"])
        self.assertEqual([m["quien"] for m in est["mesa"]], ["robot"])
        self.assertTrue(any("[ERROR]" in linea for linea in est["log"]))

    def test_una_decision_rota_no_revela_lo_que_iba_a_hacer(self):
        partida = self.crear(RAPIDO)

        def canta_y_rompe(*_a, **_k):
            partida._hablar(vj.TRUCO)
            raise RuntimeError("boom")

        with patch.object(partida.ia, "decidir_iniciar_envido", return_value=(None, "")), \
                patch.object(partida.ia, "decidir_iniciar_truco", side_effect=canta_y_rompe):
            self.robot_mano(partida)
            self.assertTrue(self.libre(partida))
        frases = [v["frase"] for v in partida.estado_publico()["voz"]]
        self.assertNotIn("¡Truco!", frases)

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


class TestRitmoDeLasCadenas(Base):
    def test_una_cadena_no_vuelve_a_pensar_todo(self):
        """The robot wins a baza and plays again: the second card takes SEGUIR, not a full think."""
        ritmo = Ritmo(pensar=(0.4, 0.4), pensar_grande=(0.4, 0.4), seguir=(0.02, 0.02),
                      alcanzar=0.01, colocar=0.01, pausa_baza=0.01, pausa_mano=0.01)
        partida = self.crear(ritmo)
        self.sin_cantos(partida)
        self.repartir(partida, [(4, "Copa"), (5, "Copa"), (6, "Copa")],
                      [(1, "Espada"), (1, "Basto"), (7, "Espada")])
        inicio = time.time()
        partida.jugar_carta_humano({"valor": 4, "palo": "Copa"})
        self.assertTrue(self.libre(partida))
        duracion = time.time() - inicio
        robot = [m for m in partida.estado_publico()["mesa"] if m["quien"] == "robot"]
        self.assertEqual(len(robot), 2)  # it won the first baza and led the second
        self.assertLess(duracion, 0.7)   # two full thinks would be >= 0.8 s

    def test_responder_el_truco_del_humano_es_una_decision_grande(self):
        ritmo = Ritmo(pensar=(0.02, 0.02), pensar_grande=(0.4, 0.4), seguir=(0.02, 0.02),
                      alcanzar=0.01, colocar=0.01, pausa_baza=0, pausa_mano=0)
        partida = self.crear(ritmo)
        with patch.object(partida.ia, "decidir_iniciar_envido", return_value=(None, "")), \
                patch.object(partida.ia, "responder_canto_truco", return_value=("QUIERO", "")):
            inicio = time.time()
            partida.cantar_truco_humano()
            self.assertTrue(self.libre(partida))
        self.assertGreaterEqual(time.time() - inicio, 0.35)

    def test_querer_real_envido_piensa_mas_que_querer_envido(self):
        ritmo = Ritmo(pensar=(0.02, 0.02), pensar_grande=(0.4, 0.4), seguir=(0.02, 0.02),
                      alcanzar=0.01, colocar=0.01, pausa_baza=0, pausa_mano=0)
        for canto, minimo, maximo in (("REAL_ENVIDO", 0.35, 5.0), ("ENVIDO", 0.0, 0.3)):
            partida = self.crear(ritmo)
            with patch.object(partida.ia, "responder_envido", return_value=("QUIERO", "")):
                inicio = time.time()
                partida.cantar_envido_humano(canto)
                self.assertTrue(self.libre(partida))
                duracion = time.time() - inicio
            self.assertTrue(minimo <= duracion <= maximo, (canto, duracion))


class TestNadaSeAdelanta(Base):
    def test_mientras_piensa_nada_revela_la_decision(self):
        """Log, bubble, reasoning panel and pending call stay as before until the wait ends."""
        partida = self.crear(LENTO)

        def decide_truco(*_a, **_k):
            partida.ia._registrar_razonamiento("Truco", "TRUCO (MENTIRA)", "m", "u", "Mentira", "x")
            return "TRUCO", "Tengo 33 de envido y un 1 de espada"

        with patch.object(partida.ia, "decidir_iniciar_envido", return_value=(None, "")), \
                patch.object(partida.ia, "decidir_iniciar_truco", side_effect=decide_truco):
            antes = partida.estado_publico()
            partida.jugar_carta_humano(partida.cartas_humano[0])
            visto = []
            while partida.estado_publico()["robot"]["estado"] == "pensando":
                visto.append(partida.estado_publico())
                time.sleep(0.03)
            self.assertTrue(visto)
            for est in visto:
                self.assertIsNone(est["canto_pendiente"])
                self.assertEqual(est["ultima_frase_robot"], antes["ultima_frase_robot"])
                self.assertEqual(est["aprendizaje_ia"]["traza_razonamiento"],
                                 antes["aprendizaje_ia"]["traza_razonamiento"])
                nuevas = [l for l in est["log"] if l not in antes["log"]]
                self.assertTrue(all(l.startswith("👤 Humano jugó") for l in nuevas), nuevas)
                self.assertEqual(est["voz"], antes["voz"])
            self.assertTrue(self.libre(partida))
        est = partida.estado_publico()
        self.assertEqual(est["canto_pendiente"]["canto"], "TRUCO")
        self.assertEqual(est["ultima_frase_robot"], "¡Truco!")
        self.assertIn('🤖 Ottoman: "¡Truco!"', est["log"])
        textos = est["log"] + [est["ultima_frase_robot"]] + [v["frase"] for v in est["voz"]]
        self.assertFalse(any("Tengo 33" in t for t in textos))

    def test_reconectar_el_simulador_mientras_piensa_no_adelanta_ni_duplica_la_carta(self):
        robot = RobotFalso()
        partida = self.crear(LENTO, robot=robot)
        self.sin_cantos(partida)
        self.robot_mano(partida)
        self.assertEqual(partida.estado_publico()["robot"]["estado"], "pensando")
        partida.sincronizar_mesa_3d()  # what the server does when the simulator comes back
        self.assertTrue(esperar(lambda: robot.llamadas.count(("repartir_cartas",)) >= 2))
        self.assertFalse(any(l[0] == "tirar_carta" for l in robot.llamadas))
        self.assertTrue(self.libre(partida))
        self.assertTrue(esperar(lambda: any(l[0] == "tirar_carta" for l in robot.llamadas)))
        time.sleep(0.1)
        self.assertEqual([l[3] for l in robot.llamadas if l[0] == "tirar_carta"], [1])

    def test_cancelar_mientras_coloca_la_carta(self):
        ritmo = Ritmo(pensar=(0.02, 0.02), pensar_grande=(0.02, 0.02), alcanzar=0.05, colocar=0.6,
                      pausa_baza=0.01, pausa_mano=0.01)
        partida = self.crear(ritmo)
        self.sin_cantos(partida)
        self.robot_mano(partida)
        self.assertTrue(esperar(lambda: partida.estado_publico()["robot"]["estado"] == "actuando"))
        hilo = partida._hilo_robot
        partida.reiniciar_partido(15)
        hilo.join(2)
        self.assertFalse(hilo.is_alive())
        time.sleep(0.2)
        est = partida.estado_publico()
        self.assertEqual(est["robot"]["estado"], "ocioso")
        self.assertEqual(est["mesa"], [])
        self.assertEqual(est["cartas_robot_restantes"], 3)
        self.assertEqual(est["turno_carta"], "humano")


class TestTantoDeLaPartida(Base):
    """The tanto said out loud after an accepted envido, through the whole game."""

    HUMANO_33 = [(7, "Oro"), (6, "Oro"), (1, "Copa")]
    HUMANO_27 = [(7, "Copa"), (12, "Copa"), (1, "Oro")]
    HUMANO_30 = [(7, "Copa"), (3, "Copa"), (1, "Oro")]
    ROBOT_32 = [(7, "Espada"), (5, "Espada"), (4, "Basto")]
    ROBOT_27 = [(7, "Basto"), (12, "Basto"), (4, "Espada")]
    ROBOT_30 = [(6, "Basto"), (4, "Basto"), (5, "Espada")]

    def jugar_envido(self, humano, robot, robot_es_mano):
        partida = self.crear(Ritmo.instantaneo())
        self.sin_cantos(partida)
        self.repartir(partida, humano, robot, robot_es_mano)
        with patch.object(partida.ia, "responder_envido", return_value=("QUIERO", "")):
            res = partida.cantar_envido_humano("ENVIDO")
        self.assertNotIn("error", res)
        est = partida.estado_publico()
        self.assertTrue(est["envido_resuelto"])
        return partida, est

    def tanto_dicho(self, est):
        return [v["frase"] for v in est["voz"] if v["accion"] == vj.TANTO]

    def test_el_robot_mano_dice_su_numero_aunque_pierda(self):
        partida, est = self.jugar_envido(self.HUMANO_33, self.ROBOT_32, robot_es_mano=True)
        self.assertEqual(est["resultado_envido"]["ganador"], "humano")
        self.assertEqual(self.tanto_dicho(est), ["Treinta y dos."])

    def test_el_robot_pie_dice_su_numero_solo_si_gana(self):
        _, est = self.jugar_envido(self.HUMANO_27, self.ROBOT_32, robot_es_mano=False)
        self.assertEqual(est["resultado_envido"]["ganador"], "robot")
        self.assertEqual(self.tanto_dicho(est), ["Treinta y dos."])

    def test_el_robot_pie_que_pierde_dice_son_buenas_y_no_muestra_su_numero(self):
        partida, est = self.jugar_envido(self.HUMANO_33, self.ROBOT_27, robot_es_mano=False)
        self.assertEqual(est["resultado_envido"]["ganador"], "humano")
        self.assertEqual(self.tanto_dicho(est), ["Son buenas."])
        self.assertEqual(est["ultima_frase_robot"], "Son buenas.")
        self.assertFalse(any("27" in linea or "veintisiete" in linea.lower() for linea in est["log"]), est["log"])

    def test_empate_gana_la_mano(self):
        _, est = self.jugar_envido(self.HUMANO_30, self.ROBOT_30, robot_es_mano=False)
        self.assertEqual(est["resultado_envido"]["ganador"], "humano")
        self.assertEqual(self.tanto_dicho(est), ["Son buenas."])
        _, est = self.jugar_envido(self.HUMANO_30, self.ROBOT_30, robot_es_mano=True)
        self.assertEqual(est["resultado_envido"]["ganador"], "robot")
        self.assertEqual(self.tanto_dicho(est), ["Treinta."])

    def test_las_manos_de_prueba_tienen_el_tanto_esperado(self):
        cartas = lambda m: [{"valor": v, "palo": p} for v, p in m]
        for mano, tanto in ((self.HUMANO_33, 33), (self.HUMANO_27, 27), (self.HUMANO_30, 30),
                            (self.ROBOT_32, 32), (self.ROBOT_27, 27), (self.ROBOT_30, 30)):
            self.assertEqual(tr.calcular_envido(cartas(mano)), tanto, mano)


class TestServidorRitmo(unittest.TestCase):
    def crear(self, ritmo):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        partida = PartidaTruco(historial_path=str(Path(tmp.name) / "h.json"), rng=random.Random(3), ritmo=ritmo)
        self.addCleanup(partida.cancelar_turno_robot)
        return partida

    def servir(self, partida, sin_robot=True):
        from servidor_truco import crear_servidor
        servidor = crear_servidor(partida, puerto=0, sin_robot=sin_robot)
        hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
        hilo.start()
        self.addCleanup(servidor.server_close)
        self.addCleanup(servidor.shutdown)
        return servidor

    def post(self, servidor, ruta, cuerpo):
        import http.client
        import json
        conn = http.client.HTTPConnection("127.0.0.1", servidor.server_port, timeout=5)
        conn.request("POST", ruta, json.dumps(cuerpo), {"Content-Type": "application/json"})
        r = conn.getresponse()
        datos = json.loads(r.read())
        conn.close()
        return r.status, datos

    def test_elegir_velocidad_sale_del_modo_instantaneo(self):
        partida = self.crear(Ritmo.instantaneo())
        servidor = self.servir(partida)
        codigo, datos = self.post(servidor, "/truco/ritmo", {"velocidad": 2})
        self.assertEqual(codigo, 200)
        self.assertFalse(partida.ritmo.es_instantaneo)
        self.assertEqual(partida.ritmo.velocidad, 2.0)
        self.assertEqual(datos["estado"]["robot"], dict(datos["estado"]["robot"], instantaneo=False, velocidad=2.0))

    def test_cambiar_velocidad_conserva_el_ritmo_humano(self):
        partida = self.crear(Ritmo.humano(1.0))
        servidor = self.servir(partida)
        self.assertEqual(self.post(servidor, "/truco/ritmo", {"velocidad": 0.5})[0], 200)
        self.assertEqual(partida.ritmo.velocidad, 0.5)
        self.assertFalse(partida.ritmo.es_instantaneo)

    def test_velocidad_invalida(self):
        partida = self.crear(Ritmo.instantaneo())
        servidor = self.servir(partida)
        for valor in (3, True, "2", None):
            self.assertEqual(self.post(servidor, "/truco/ritmo", {"velocidad": valor})[0], 400, valor)
        self.assertTrue(partida.ritmo.es_instantaneo)

    def test_los_sondeos_no_reintentan_conectar_a_cada_rato_ni_con_el_lock(self):
        import servidor_truco
        partida = self.crear(Ritmo.instantaneo())
        servidor = servidor_truco.ServidorTruco(("127.0.0.1", 0), partida, False)
        self.addCleanup(servidor.server_close)
        lock_libre = []

        def probar_lock():
            tomado = partida.lock.acquire(timeout=1)
            if tomado:
                partida.lock.release()
            lock_libre.append(tomado)

        def conectar():
            t = threading.Thread(target=probar_lock)
            t.start()
            t.join()
            raise OSError("sin simulador")

        falso = unittest.mock.MagicMock()
        falso.conectar.side_effect = conectar
        with patch.object(servidor_truco, "Robot", return_value=falso) as Robot:
            for _ in range(5):
                self.assertIsNone(servidor.asegurar_robot())
            self.assertEqual(Robot.call_count, 1)
            self.assertEqual(lock_libre, [True])
            servidor._ultimo_intento -= servidor_truco.INTERVALO_RECONEXION + 0.1
            servidor.asegurar_robot()
            self.assertEqual(Robot.call_count, 2)


if __name__ == "__main__":
    unittest.main()
