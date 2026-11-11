"""ADR 0068: envido only before the caller's own first-baza card (game layer)."""

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from partida_truco import PartidaTruco


def _partida(tmp, human_mano=True, seed=3):
    p = PartidaTruco(robot=None, historial_path=str(Path(tmp) / "h.json"), rng=random.Random(seed))
    if not human_mano:
        p.es_mano_humano = True
        p.mano_terminada = True
        with patch.object(PartidaTruco, "_avanzar_turno_robot_si_corresponde"):
            p.nueva_mano()
    return p


def _robot_played(p):
    carta = p.cartas_robot_disponibles.pop(0)
    p.mesa.append({"quien": "robot", "carta": carta, "nombre": "x", "ronda": 1})
    p.turno_carta = "humano"


class EnvidoSoloAntesDeTirar(unittest.TestCase):
    def test_human_cannot_call_after_playing_first_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _partida(tmp)
            self.assertTrue(p.puede_cantar_envido())
            with patch.object(PartidaTruco, "_avanzar_turno_robot_si_corresponde"):
                p.jugar_carta_humano(p.cartas_humano_disponibles[0])
            self.assertFalse(p.puede_cantar_envido())
            self.assertFalse(p.estado_publico()["envido_disponible"])
            self.assertIn("error", p.cantar_envido_humano("ENVIDO"))
            self.assertIn("error", p.procesar_voz_truco("envido"))
            self.assertFalse(p.envido_resuelto)

    def test_human_cannot_call_envido_in_response_to_truco_after_playing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _partida(tmp)
            with patch.object(PartidaTruco, "_avanzar_turno_robot_si_corresponde"):
                p.jugar_carta_humano(p.cartas_humano_disponibles[0])
            p._robot_canta_truco("TRUCO", "")
            opciones = p.canto_pendiente["opciones"]
            self.assertFalse({"ENVIDO", "REAL_ENVIDO", "FALTA_ENVIDO"} & set(opciones))
            self.assertIn("error", p.responder_canto_humano("ENVIDO"))
            self.assertIn("error", p.cantar_envido_humano("ENVIDO"))
            self.assertIn("error", p.procesar_voz_truco("envido"))
            self.assertEqual(p.canto_pendiente["tipo"], "TRUCO")

    def test_human_can_still_respond_envido_to_truco_before_playing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _partida(tmp, human_mano=False)
            _robot_played(p)
            p._robot_canta_truco("TRUCO", "")
            self.assertIn("ENVIDO", p.canto_pendiente["opciones"])

    def test_pie_can_call_after_mano_played(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _partida(tmp, human_mano=False)
            _robot_played(p)
            self.assertTrue(p.puede_cantar_envido())
            self.assertTrue(p.estado_publico()["envido_disponible"])
            self.assertNotIn("error", p.cantar_envido_humano("ENVIDO"))
            self.assertFalse(p.puede_cantar_envido("robot"))

    def test_robot_never_calls_after_playing_first_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(30):
                p = _partida(tmp, human_mano=False, seed=seed)
                _robot_played(p)
                self.assertFalse(p.puede_cantar_envido("robot"))
                with patch.object(PartidaTruco, "_robot_canta_envido") as canta:
                    p.cantar_truco_humano()
                    self.assertFalse(canta.called)

    def test_network_mask_excludes_envido_after_robot_played(self):
        for played in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                p = _partida(tmp, human_mano=False)
                if played:
                    _robot_played(p)
                p.ia.modo_agente = "refuerzo"
                seen = []

                def fake(actions, pending=None):
                    seen.append(list(actions))
                    return "quiero"

                with patch.object(p, "_network_decide", side_effect=fake),                         patch.object(PartidaTruco, "_avanzar_turno_robot_si_corresponde"):
                    p._responder_truco_robot("TRUCO", 2, 1, None)
                has_envido = any("envido" in a for a in seen[0])
                self.assertEqual(has_envido, not played)

    def test_network_mask_on_robot_turn_before_playing_offers_envido(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _partida(tmp, human_mano=False)
            p.ia.modo_agente = "refuerzo"
            seen = []

            def fake(actions, pending=None):
                seen.append(list(actions))
                return next(a for a in actions if a.startswith("card_"))

            with patch.object(p, "_network_decide", side_effect=fake):
                p._avanzar_turno_robot_si_corresponde()
            self.assertIn("envido", seen[0])


if __name__ == "__main__":
    unittest.main()
