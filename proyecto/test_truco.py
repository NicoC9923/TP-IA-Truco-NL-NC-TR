"""Tests integrales del sistema de Truco Argentino, Visión, IA (474k manos) y Control por Voz."""

from __future__ import annotations

import base64
import unittest
import truco_reglas as tr
from truco_ia import AgenteTrucoIA
from partida_truco import PartidaTruco
from truco_vision import DetectorCartas
from control_voz import AgenteRobot


class TestTrucoReglasYMotor(unittest.TestCase):
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
        ia = AgenteTrucoIA(prob_farol=0.0)
        self.assertEqual(ia.stats["total_manos"], 474000)

        mano_fuerte = [{"valor": 7, "palo": "Espada"}, {"valor": 6, "palo": "Espada"}, {"valor": 1, "palo": "Espada"}]
        tantos, prob = ia.prob_envido(mano_fuerte, es_mano=True)
        self.assertEqual(tantos, 33)
        self.assertEqual(prob, 1.0)
        self.assertGreater(ia.percentil_fuerza_truco(mano_fuerte), 0.98)

    def test_partida_flujo_completo_y_turnos(self):
        partida = PartidaTruco(robot=None, meta_puntos=15)
        partida.ia.prob_farol = 0.0

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

    def test_turno_automatico_cuando_robot_es_mano(self):
        partida = PartidaTruco(robot=None, meta_puntos=15)
        partida.es_mano_humano = False
        est = partida.nueva_mano(primera_vez=True)
        # El robot al ser mano o bien cantó (canto_pendiente) o bien ya tiró su 1ra carta a la mesa
        self.assertTrue(len(est["mesa"]) == 1 or est["canto_pendiente"] is not None)

    def test_voz_truco(self):
        partida = PartidaTruco(robot=None, meta_puntos=15)
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

        r_peligro = agente.procesar("salta desde la mesa")
        self.assertTrue(r_peligro["bloqueado"])
        self.assertFalse(r_peligro["ejecutar"])


if __name__ == "__main__":
    unittest.main()
