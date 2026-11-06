"""Tests integrales del sistema de Truco Argentino, Visión, IA (474k manos) y Control por Voz."""

from __future__ import annotations

import base64
import random
import copy
import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import unittest

PROYECTO_DIR = Path(__file__).resolve().parent
RAIZ_DIR = PROYECTO_DIR.parent
if str(RAIZ_DIR) not in sys.path:
    sys.path.insert(0, str(RAIZ_DIR))

import truco_reglas as tr
from truco_ia import AgenteTrucoIA
from partida_truco import PartidaTruco
from truco_vision import DetectorCartas
from control_voz import AgenteRobot
from entorno.sim.robots import G1
from entorno.sim.mundo import Mundo


class TestTrucoReglasYMotor(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_hist = str(Path(self._tmpdir.name) / "test_historial.json")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_partidas_con_misma_semilla_reproducen_reparto_y_decision(self):
        partidas = [
            PartidaTruco(robot=None, historial_path=str(Path(self._tmpdir.name) / f"{i}.json"),
                         rng=random.Random(42))
            for i in range(2)
        ]
        self.assertEqual(partidas[0].estado_publico(), partidas[1].estado_publico())
        resultados = [partida.cantar_envido_humano("ENVIDO") for partida in partidas]
        self.assertEqual(resultados[0], resultados[1])
        self.assertEqual(partidas[0].estado_publico(), partidas[1].estado_publico())
    def test_carta_ajena_no_cambia_mano_mesa_ni_turno(self):
        partida = PartidaTruco(robot=None, historial_path=self.tmp_hist)
        antes = copy.deepcopy(partida.estado_publico())
        carta = next({"valor": valor, "palo": palo} for palo in ("Espada", "Basto", "Oro", "Copa")
                     for valor in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)
                     if not any(c["valor"] == valor and c["palo"] == palo for c in antes["cartas_humano_disponibles"]))
        resultado = partida.jugar_carta_humano(carta)
        self.assertEqual(resultado.get("error"), "Esa carta no está en tu mano")
        for campo in ("cartas_humano_disponibles", "mesa", "turno_carta"):
            self.assertEqual(resultado["estado"][campo], antes[campo])

    def test_carta_malformada_es_rechazada(self):
        partida = PartidaTruco(robot=None, historial_path=self.tmp_hist)
        for carta in ({"valor": "incorrecto", "palo": "Oro"}, {"valor": 1, "palo": "Inexistente"},
                      {"valor": 1.5, "palo": "Oro"}, {"valor": True, "palo": "Oro"}, None):
            with self.subTest(carta=carta):
                resultado = partida.jugar_carta_humano(carta)
                self.assertEqual(resultado.get("error"), "Esa carta no está en tu mano")

    def test_voz_rechaza_carta_ajena(self):
        partida = PartidaTruco(robot=None, historial_path=self.tmp_hist)
        antes = copy.deepcopy(partida.estado_publico())
        carta = next({"valor": valor, "palo": palo} for palo in ("Espada", "Basto", "Oro", "Copa")
                     for valor in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)
                     if not any(c["valor"] == valor and c["palo"] == palo for c in antes["cartas_humano_disponibles"]))
        resultado = partida.procesar_voz_truco(f"juego {carta['valor']} de {carta['palo']}")
        self.assertEqual(resultado.get("error"), "Esa carta no está en tu mano")
        for campo in ("cartas_humano_disponibles", "mesa", "turno_carta"):
            self.assertEqual(resultado["estado"][campo], antes[campo])

    def test_voz_rechaza_carta_incompleta(self):
        partida = PartidaTruco(robot=None, historial_path=self.tmp_hist)
        resultado = partida.procesar_voz_truco("juego una carta")
        self.assertEqual(resultado.get("error"), "Esa carta no está en tu mano")

    def test_carta_repartida_se_juega(self):
        partida = PartidaTruco(robot=None, historial_path=self.tmp_hist)
        carta = copy.deepcopy(partida.estado_publico()["cartas_humano_disponibles"][0])
        resultado = partida.jugar_carta_humano(carta)
        self.assertNotIn("error", resultado)
        self.assertEqual(resultado["carta_jugada"], carta)
        self.assertTrue(any(j["quien"] == "humano" and j["carta"] == carta for j in resultado["estado"]["mesa"]))

    def test_camara_envia_deteccion_sin_sustituir_carta(self):
        import servidor_truco as servidor
        partida = PartidaTruco(robot=None, historial_path=self.tmp_hist)
        antes = copy.deepcopy(partida.estado_publico())
        carta = next({"valor": valor, "palo": palo} for palo in ("Espada", "Basto", "Oro", "Copa")
                     for valor in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)
                     if not any(c["valor"] == valor and c["palo"] == palo for c in antes["cartas_humano_disponibles"]))

        class Camara:
            def procesar_frame_base64(self, imagen, cartas_mano=None):
                return cartas_mano[0] if cartas_mano else carta

        anteriores = servidor.partida_truco, servidor.detector_vision, servidor.sin_robot_flag
        servidor.partida_truco, servidor.detector_vision, servidor.sin_robot_flag = partida, Camara(), True
        server = ThreadingHTTPServer(("127.0.0.1", 0), servidor.Handler)
        hilo = threading.Thread(target=server.serve_forever, daemon=True)
        hilo.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
            conn.request("POST", "/truco/detectar", json.dumps({"imagen": "frame", "usar_mano": True}))
            detectada = json.loads(conn.getresponse().read())
            conn.request("POST", "/truco/jugar", json.dumps({"carta": detectada}))
            resultado = json.loads(conn.getresponse().read())
            conn.close()
            self.assertEqual(resultado.get("error"), "Esa carta no está en tu mano")
            for campo in ("cartas_humano_disponibles", "mesa", "turno_carta"):
                self.assertEqual(resultado["estado"][campo], antes[campo])
        finally:
            server.shutdown()
            server.server_close()
            hilo.join()
            servidor.partida_truco, servidor.detector_vision, servidor.sin_robot_flag = anteriores

    def test_jerarquia_cartas(self):
        self.assertGreater(tr.valor_carta_truco(1, "Espada"), tr.valor_carta_truco(1, "Basto"))
        self.assertGreater(tr.valor_carta_truco(1, "Basto"), tr.valor_carta_truco(7, "Espada"))
        self.assertGreater(tr.valor_carta_truco(7, "Espada"), tr.valor_carta_truco(7, "Oro"))
        self.assertGreater(tr.valor_carta_truco(7, "Oro"), tr.valor_carta_truco(3, "Copa"))
        self.assertEqual(tr.valor_carta_truco(4, "Copa"), 0)

    def test_calculo_envido(self):
        mano_33 = [{"valor": 7, "palo": "Espada"}, {"valor": 6, "palo": "Espada"}, {"valor": 1, "palo": "Oro"}]
        self.assertEqual(tr.calcular_envido(mano_33), 33)

        mano_20 = [{"valor": 10, "palo": "Copa"}, {"valor": 12, "palo": "Copa"}, {"valor": 4, "palo": "Basto"}]
        self.assertEqual(tr.calcular_envido(mano_20), 20)

        mano_distintos = [{"valor": 7, "palo": "Oro"}, {"valor": 5, "palo": "Basto"}, {"valor": 12, "palo": "Espada"}]
        self.assertEqual(tr.calcular_envido(mano_distintos), 7)

    def test_ia_probabilidades_474k(self):
        ia = AgenteTrucoIA(prob_farol=0.0, historial_path=self.tmp_hist)
        self.assertEqual(ia.stats["total_manos"], 474000)

        mano_fuerte = [{"valor": 7, "palo": "Espada"}, {"valor": 6, "palo": "Espada"}, {"valor": 1, "palo": "Espada"}]
        tantos, prob = ia.prob_envido(mano_fuerte, es_mano=True)
        self.assertEqual(tantos, 33)
        self.assertEqual(prob, 1.0)
        self.assertGreater(ia.percentil_fuerza_truco(mano_fuerte), 0.98)

    def test_partida_flujo_completo_y_turnos(self):
        partida = PartidaTruco(robot=None, meta_puntos=15, historial_path=self.tmp_hist)
        partida.ia.prob_farol = 0.0
        partida.ia.prob_farol_envido = 0.0
        partida.ia.prob_farol_truco = 0.0

        # Forzar una mano controlada donde Humano es mano
        partida.es_mano_humano = True
        partida.nueva_mano(primera_vez=True)
        partida.cartas_humano = [
            {"valor": 1, "palo": "Espada"},
            {"valor": 1, "palo": "Basto"},
            {"valor": 7, "palo": "Espada"},
        ]
        partida.cartas_humano_disponibles = list(partida.cartas_humano)
        partida.cartas_robot = [
            {"valor": 4, "palo": "Copa"},
            {"valor": 5, "palo": "Oro"},
            {"valor": 6, "palo": "Basto"},
        ]
        partida.cartas_robot_disponibles = list(partida.cartas_robot)

        # Antes de terminar la mano, las cartas de Ottoman están ocultas
        self.assertIsNone(partida.estado_publico()["cartas_robot_reveladas"])
        self.assertIsNone(partida.estado_publico()["tantos_robot"])

        # Jugar 1ra carta del humano; el robot debe responder automáticamente
        res1 = partida.jugar_carta_humano({"valor": 7, "palo": "Espada"})
        self.assertEqual(len(res1["estado"]["mesa"]), 2)
        self.assertEqual(res1["estado"]["ganadores_ronda"], ["humano"])
        self.assertEqual(res1["estado"]["ronda"], 2)

        # Jugar 2da carta del humano; humano gana 2 bazas seguidas y se lleva la mano
        res2 = partida.jugar_carta_humano({"valor": 1, "palo": "Basto"})
        self.assertTrue(res2["estado"]["mano_terminada"])
        self.assertEqual(res2["estado"]["ganador_mano"], "humano")
        self.assertEqual(res2["estado"]["tanteador"]["humano"], 1)

        # Al terminar la mano, se revelan las 3 cartas de Ottoman y sus tantos reales de Envido
        reveladas = res2["estado"]["cartas_robot_reveladas"]
        self.assertEqual(len(reveladas), 3)
        self.assertEqual(res2["estado"]["tantos_robot"], 6)
        jugadas_count = sum(1 for c in reveladas if c.get("jugada"))
        self.assertEqual(jugadas_count, 2)

    def test_deteccion_y_estadisticas_de_bluff(self):
        partida = PartidaTruco(robot=None, meta_puntos=15, historial_path=self.tmp_hist)
        partida.es_mano_humano = True
        partida.nueva_mano(primera_vez=True)
        # Forzar farol de Envido y Truco al 100% con cartas bajas para Ottoman
        partida.ia.prob_farol_envido = 1.0
        partida.ia.prob_farol_truco = 1.0
        partida.cartas_robot = [
            {"valor": 4, "palo": "Copa"},
            {"valor": 5, "palo": "Oro"},
            {"valor": 6, "palo": "Basto"},
        ]
        partida.cartas_robot_disponibles = list(partida.cartas_robot)

        canto_env, frase_env = partida.ia.decidir_iniciar_envido(
            partida.cartas_robot, es_mano=False, puntos_humano=0, puntos_robot=0, meta_puntos=15
        )
        self.assertEqual(canto_env, "ENVIDO")
        self.assertTrue(partida.ia.ultimo_canto_fue_farol)
        partida._robot_canta_envido(
            canto_env,
            frase_env,
            es_farol=partida.ia.ultimo_canto_fue_farol,
            detalle_farol=partida.ia.ultimo_detalle_farol,
        )
        # El humano dice NO QUIERO: el farol de Ottoman resulta exitoso
        partida.responder_canto_humano("NO_QUIERO")

        self.assertEqual(partida.estadisticas_partida["envido"]["bluffs_intentados"], 1)
        self.assertEqual(partida.estadisticas_partida["envido"]["bluffs_exitosos"], 1)
        self.assertEqual(len(partida.bluffs_mano_actual), 1)
        self.assertEqual(partida.bluffs_mano_actual[0]["resultado"], "exitoso")

    def test_aprendizaje_adaptativo_e_historial_persistente(self):
        ia = AgenteTrucoIA(prob_farol=0.18, historial_path=self.tmp_hist)
        p_truco_ini = ia.prob_farol_truco

        # Si el humano siempre se retira ("NO QUIERO") ante cantos de Truco del robot,
        # la IA incrementa su frecuencia de farol en Truco
        for _ in range(5):
            ia.registrar_respuesta_humano_a_canto("TRUCO", quiso=False, era_farol_robot=True)
        self.assertGreater(ia.prob_farol_truco, p_truco_ini)

        # Registrar fin de partida y verificar que quede persistido en disco
        snap = ia.registrar_partida_finalizada(
            ganador="robot",
            puntos_humano=11,
            puntos_robot=15,
            meta_puntos=15,
            estadisticas_partida={
                "manos_jugadas": 8,
                "envido": {
                    "cantados_robot": 3,
                    "bluffs_intentados": 2,
                    "bluffs_exitosos": 1,
                    "bluffs_descubiertos": 1,
                },
                "truco": {
                    "cantados_robot": 4,
                    "bluffs_intentados": 3,
                    "bluffs_exitosos": 2,
                    "bluffs_descubiertos": 1,
                },
            },
        )
        self.assertEqual(snap["ganador"], "robot")

        # Recargar desde disco y comprobar que conserve la partida y los parámetros aprendidos
        ia2 = AgenteTrucoIA(prob_farol=0.18, historial_path=self.tmp_hist)
        resumen = ia2.resumen_aprendizaje()
        self.assertEqual(resumen["partidas_jugadas"], 1)
        self.assertEqual(resumen["victorias_robot"], 1)
        self.assertEqual(len(resumen["partidas"]), 1)
        self.assertEqual(resumen["partidas"][0]["bluffs_truco_intentados"], 3)
        self.assertAlmostEqual(ia2.prob_farol_truco, ia.prob_farol_truco, places=3)

    def test_escena_mesa_truco_y_animacion_repartiendo(self):
        ruta = Path(G1.ruta_escena())
        self.assertTrue(ruta.exists())
        self.assertEqual(ruta.name, "scene_mesa_truco.xml")
        contenido = ruta.read_text(encoding="utf-8")
        self.assertIn("mesa_truco", contenido)
        self.assertIn("silla_ottoman", contenido)
        self.assertIn("repartiendo", G1.gestos)

        mundo = Mundo()
        self.assertFalse(mundo.leer()["de_pie"])
        mundo.gesto("repartiendo", 2.0)
        self.assertEqual(mundo.leer()["accion"], "repartiendo")
        self.assertFalse(mundo.leer()["de_pie"])

    def test_modos_arquitectura_agente_y_xai(self):
        partida = PartidaTruco(robot=None, meta_puntos=15, historial_path=self.tmp_hist)
        mano_media = [
            {"valor": 4, "palo": "Copa"},
            {"valor": 7, "palo": "Oro"},
            {"valor": 1, "palo": "Espada"},
        ]

        # 1. En modo Reactivo Simple, tira siempre su carta más alta (1 de Espada)
        partida.cambiar_modo_ia("reactivo")
        self.assertEqual(partida.ia.modo_agente, "reactivo")
        c_reac = partida.ia.elegir_carta_a_jugar(mano_media, ronda=1)
        self.assertEqual((c_reac["valor"], c_reac["palo"]), (1, "Espada"))
        self.assertEqual(partida.ia.ultimo_razonamiento["modo_codigo"], "reactivo")

        # 2. En modo Basado en Utilidad (474k), abre con su intermedia fuerte (7 de Oro) y reserva el Ancho
        partida.cambiar_modo_ia("utilidad")
        self.assertEqual(partida.ia.modo_agente, "utilidad")
        c_util = partida.ia.elegir_carta_a_jugar(mano_media, ronda=1)
        self.assertEqual((c_util["valor"], c_util["palo"]), (7, "Oro"))
        self.assertGreaterEqual(len(partida.ia.traza_razonamiento), 2)

        # 3. Volver a modo Aprendizaje Adaptativo
        est = partida.cambiar_modo_ia("aprendizaje")
        self.assertEqual(est["aprendizaje_ia"]["modo_agente"], "aprendizaje")
        self.assertTrue(len(est["aprendizaje_ia"]["traza_razonamiento"]) >= 2)

    def test_turno_automatico_cuando_robot_es_mano(self):
        partida = PartidaTruco(robot=None, meta_puntos=15, historial_path=self.tmp_hist)
        partida.es_mano_humano = False
        est = partida.nueva_mano(primera_vez=True)
        # El robot al ser mano o bien cantó (canto_pendiente) o bien ya tiró su 1ra carta a la mesa
        self.assertTrue(len(est["mesa"]) == 1 or est["canto_pendiente"] is not None)

    def test_voz_truco(self):
        partida = PartidaTruco(robot=None, meta_puntos=15, historial_path=self.tmp_hist)
        partida.es_mano_humano = True
        partida.nueva_mano(primera_vez=True)
        partida.cartas_humano = [
            {"valor": 1, "palo": "Espada"},
            {"valor": 3, "palo": "Oro"},
            {"valor": 2, "palo": "Copa"},
        ]
        partida.cartas_humano_disponibles = list(partida.cartas_humano)

        res = partida.procesar_voz_truco("juego el ancho de espada")
        self.assertNotIn("error", res)
        self.assertGreaterEqual(len(res["estado"]["mesa"]), 1)

    def test_vision_detector_y_matcheo_mano(self):
        det = DetectorCartas()
        if not det.disponible:
            self.skipTest("OpenCV no instalado")
        import cv2
        import numpy as np

        # Crear imagen sintética con fondo oscuro y carta blanca con círculo dorado (Oro)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (180, 80), (460, 400), (255, 255, 255), -1)
        # Círculo amarillo/dorado en BGR: (0, 215, 255)
        cv2.circle(img, (320, 240), 45, (0, 215, 255), -1)
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf).decode("ascii")

        mano = [{"valor": 1, "palo": "Oro"}, {"valor": 4, "palo": "Espada"}]
        res = det.procesar_frame_base64(b64, cartas_mano=mano)
        self.assertIsNotNone(res)
        self.assertEqual(res["palo"], "Oro")
        self.assertEqual(res["valor"], 1)

    def test_control_voz_bilingue_y_seguridad(self):
        agente = AgenteRobot(robot=None)
        r1 = agente.procesar("avanzá 2 metros")
        self.assertEqual(r1["tipo"], "MOVER")
        self.assertTrue(r1["ejecutar"])

        r2 = agente.procesar("festejá")
        self.assertEqual(r2["tipo"], "FESTEJO")
        self.assertTrue(r2["ejecutar"])

        r3 = agente.procesar("repartí las cartas")
        self.assertEqual(r3["tipo"], "REPARTIR")
        self.assertTrue(r3["ejecutar"])

        r4 = agente.procesar("sentate en la mesa")
        self.assertEqual(r4["tipo"], "SENTARSE")
        self.assertTrue(r4["ejecutar"])

        r_peligro = agente.procesar("salta desde la mesa")
        self.assertTrue(r_peligro["bloqueado"])
        self.assertFalse(r_peligro["ejecutar"])


if __name__ == "__main__":
    unittest.main()
