"""Tests for the human-like pacing config (ritmo.py)."""

from __future__ import annotations

import random
import unittest

import robot  # noqa: F401  (puts entorno/ on sys.path)
from sim import cartas as C

import ritmo as R
from ritmo import Ritmo


class TestRitmo(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(1)

    def test_pensar_cae_en_1_5_a_3_segundos(self):
        r = Ritmo.humano()
        muestras = [r.retraso(R.PENSAR, self.rng) for _ in range(300)]
        self.assertGreaterEqual(min(muestras), 1.5)
        self.assertLessEqual(max(muestras), 3.0)
        self.assertGreater(max(muestras) - min(muestras), 0.5, "must be randomized")

    def test_las_decisiones_grandes_tardan_mas(self):
        r = Ritmo.humano()
        normal = [r.retraso(R.PENSAR, self.rng) for _ in range(200)]
        grande = [r.retraso(R.PENSAR_GRANDE, self.rng) for _ in range(200)]
        self.assertGreater(sum(grande) / 200, sum(normal) / 200 + 0.8)
        self.assertGreaterEqual(min(grande), 2.5)

    def test_cuales_son_grandes(self):
        for accion in ("truco", "retruco", "vale_cuatro", "falta_envido", "real_envido"):
            self.assertTrue(R.es_decision_grande(accion), accion)
        for accion in ("envido", "quiero", "no_quiero", "carta", "mazo"):
            self.assertFalse(R.es_decision_grande(accion), accion)
        self.assertTrue(R.es_decision_grande("quiero", pendiente="vale_cuatro"))
        self.assertFalse(R.es_decision_grande("quiero", pendiente="envido"))

    def test_retraso_pensar_elige_el_rango_por_accion(self):
        r = Ritmo.humano()
        for _ in range(50):
            self.assertLessEqual(r.retraso_pensar(["carta"], rng=self.rng), 3.0)
            self.assertGreaterEqual(r.retraso_pensar(["truco"], rng=self.rng), 2.5)
            self.assertGreaterEqual(r.retraso_pensar(["quiero"], pendiente="retruco", rng=self.rng), 2.5)

    def test_seguir_una_cadena_es_corto(self):
        r = Ritmo.humano()
        muestras = [r.retraso(R.SEGUIR, self.rng) for _ in range(200)]
        self.assertGreaterEqual(min(muestras), 0.5)
        self.assertLessEqual(max(muestras), 1.0)
        self.assertLess(max(muestras), min(r.retraso(R.PENSAR, self.rng) for _ in range(50)))
        self.assertIn(R.SEGUIR, R.EVENTOS)
        self.assertAlmostEqual(Ritmo.humano(2.0).retraso(R.SEGUIR, random.Random(3)),
                               Ritmo.humano(1.0).retraso(R.SEGUIR, random.Random(3)) / 2)
        self.assertEqual(Ritmo.instantaneo().retraso(R.SEGUIR), 0.0)

    def test_la_carta_sigue_la_animacion_3d(self):
        sim = C.Ritmo()
        self.assertEqual(Ritmo().alcanzar, sim.alcanzar)
        self.assertEqual(Ritmo().colocar, sim.llevar)
        self.assertAlmostEqual(Ritmo().retraso(R.CARTA), sim.alcanzar + sim.llevar)

    def test_pausas_cortas(self):
        r = Ritmo.humano()
        self.assertTrue(0 < r.retraso(R.BAZA) < 2)
        self.assertTrue(0 < r.retraso(R.MANO) < 3)

    def test_multiplicador_de_velocidad(self):
        base = Ritmo.humano(1.0)
        rapido = Ritmo.humano(2.0)
        lento = Ritmo.humano(0.5)
        for evento in (R.CARTA, R.BAZA, R.MANO):
            self.assertAlmostEqual(rapido.retraso(evento), base.retraso(evento) / 2)
            self.assertAlmostEqual(lento.retraso(evento), base.retraso(evento) * 2)
        muestras = [rapido.retraso(R.PENSAR, self.rng) for _ in range(100)]
        self.assertTrue(0.75 <= min(muestras) and max(muestras) <= 1.5)
        self.assertEqual(rapido.rango_pensar, (0.75, 1.5))

    def test_velocidad_invalida(self):
        with self.assertRaises(ValueError):
            Ritmo.humano(3)

    def test_modo_instantaneo_es_cero_en_todo(self):
        r = Ritmo.instantaneo()
        self.assertTrue(r.es_instantaneo)
        for evento in R.EVENTOS:
            self.assertEqual(r.retraso(evento, self.rng), 0.0)
        self.assertEqual(r.retraso_pensar(["truco"], rng=self.rng), 0.0)

    def test_evento_desconocido(self):
        with self.assertRaises(ValueError):
            Ritmo().retraso("dormir")

    def test_desde_entorno(self):
        self.assertTrue(R.desde_entorno("0").es_instantaneo)
        self.assertTrue(R.desde_entorno("sin").es_instantaneo)
        self.assertEqual(R.desde_entorno("2").velocidad, 2.0)
        self.assertEqual(R.desde_entorno("x0.5").velocidad, 0.5)
        self.assertEqual(R.desde_entorno(None).velocidad, 1.0)
        self.assertFalse(R.desde_entorno("").es_instantaneo)


if __name__ == "__main__":
    unittest.main()
