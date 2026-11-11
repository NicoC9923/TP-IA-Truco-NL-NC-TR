"""Tests for what the robots say out loud (ADR 0066)."""

from __future__ import annotations

import inspect
import random
import re
import unittest

import truco_reglas as tr
import voz_jugador as vj

PROHIBIDAS = [
    r"\bfarol", r"\d", r"\btengo\b", r"\bmentira", r"\bespada\b", r"\bbasto\b", r"\boro\b",
    r"\bcopa\b", r"\bancho\b",
]
NUMEROS = {vj.numero_en_palabras(n) for n in range(34)}


class TestFrases(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(3)

    def test_cada_accion_tiene_frase(self):
        for accion in vj.ACCIONES:
            if accion == vj.CARTA:
                continue
            self.assertTrue(vj.frase(accion, self.rng), accion)

    def test_frases_de_los_cantos_son_las_del_jugador_real(self):
        esperadas = {
            vj.TRUCO: "¡Truco!", vj.RETRUCO: "¡Quiero retruco!", vj.VALE_CUATRO: "¡Quiero vale cuatro!",
            vj.ENVIDO: "¡Envido!", vj.REAL_ENVIDO: "¡Real envido!", vj.FALTA_ENVIDO: "¡Falta envido!",
            vj.ENVIDO_ENVIDO: "¡Envido, envido!", vj.QUIERO: "¡Quiero!", vj.NO_QUIERO: "No quiero.",
            vj.MAZO: "Me voy al mazo.",
        }
        for accion, texto in esperadas.items():
            self.assertEqual(vj.frase(accion, self.rng), texto)

    def test_carta_varia_y_a_veces_calla(self):
        salidas = {vj.frase(vj.CARTA, self.rng) for _ in range(200)}
        self.assertIn(None, salidas)
        self.assertGreaterEqual(len(salidas - {None}), 3)

    def test_nada_prohibido_en_ninguna_frase(self):
        for texto in vj.todas_las_frases():
            baja = texto.lower()
            if baja.strip(". ") in NUMEROS:
                continue  # a tanto announcement is a number in words
            for patron in PROHIBIDAS:
                self.assertIsNone(re.search(patron, baja), (texto, patron))

    def test_las_frases_de_carta_no_dependen_de_la_carta(self):
        self.assertEqual(list(inspect.signature(vj.frase).parameters), ["accion", "rng"])
        for texto in (f for f in vj._FRASES_CARTA if f):
            self.assertLess(len(texto.split()), 3)

    def test_numeros_en_palabras(self):
        casos = {0: "cero", 7: "siete", 20: "veinte", 27: "veintisiete", 30: "treinta",
                 31: "treinta y uno", 33: "treinta y tres", 22: "veintidós", 16: "dieciséis"}
        for n, palabras in casos.items():
            self.assertEqual(vj.numero_en_palabras(n), palabras)
        for malo in (-1, 34, "7", 7.0):
            with self.assertRaises(ValueError):
                vj.numero_en_palabras(malo)

    def test_tanto_segun_las_reglas(self):
        # The mano always says its number first.
        self.assertEqual(vj.frase_tanto(33, 20, robot_es_mano=True), "Treinta y tres.")
        self.assertEqual(vj.frase_tanto(5, 30, robot_es_mano=True), "Cinco.")
        # The pie says its number only if it beats the mano; a tie goes to the mano.
        self.assertEqual(vj.frase_tanto(31, 27, robot_es_mano=False), "Treinta y uno.")
        self.assertEqual(vj.frase_tanto(27, 31, robot_es_mano=False), "Son buenas.")
        self.assertEqual(vj.frase_tanto(27, 27, robot_es_mano=False), "Son buenas.")

    def test_frase_de_tanto_con_cartas_reales(self):
        for _ in range(100):
            mano = self.rng.sample([(v, p) for p in tr.PALOS for v in tr.VALORES], 3)
            cartas = [{"valor": v, "palo": p} for v, p in mano]
            texto = vj.frase_tanto(tr.calcular_envido(cartas), 20, robot_es_mano=True)
            self.assertIn(texto.strip(".").lower(), NUMEROS)


class TestGestosYPerfiles(unittest.TestCase):
    def test_gestos(self):
        for accion in (vj.TRUCO, vj.RETRUCO, vj.VALE_CUATRO, vj.ENVIDO, vj.REAL_ENVIDO, vj.FALTA_ENVIDO):
            self.assertEqual(vj.gesto(accion), "desafiar")
        self.assertEqual(vj.gesto(vj.MAZO), "gesto_mazo")
        self.assertIsNone(vj.gesto(vj.CARTA))
        self.assertIsNone(vj.gesto(vj.QUIERO))

    def test_los_gestos_existen_en_el_robot(self):
        import robot  # noqa: F401
        from sim.robot import Robot
        for accion in vj.ACCIONES:
            g = vj.gesto(accion)
            if g:
                self.assertTrue(callable(getattr(Robot, g, None)), g)

    def test_dos_robots_suenan_distinto(self):
        a, b = vj.perfil_voz(1), vj.perfil_voz(2)
        self.assertNotEqual(a["tono"], b["tono"])
        self.assertNotEqual(a["preferencias"], b["preferencias"])
        self.assertNotEqual(a["nombre"], b["nombre"])
        for p in (a, b):
            self.assertEqual(p["idioma"], "es-AR")
            self.assertTrue(0.5 <= p["tono"] <= 2 and 0.5 <= p["velocidad"] <= 2)
        a["preferencias"].append("x")
        self.assertNotIn("x", vj.perfil_voz(1)["preferencias"])
        with self.assertRaises(ValueError):
            vj.perfil_voz(3)

    def test_preferencias_de_voz_cubren_las_voces_comunes(self):
        uno = [p.lower() for p in vj.perfil_voz(1)["preferencias"]]
        dos = [p.lower() for p in vj.perfil_voz(2)["preferencias"]]
        for nombre in ("tomás", "pablo", "raúl", "álvaro", "jorge"):
            self.assertIn(nombre, uno)
        self.assertNotIn("female", uno)
        self.assertEqual(set(uno) & set(dos), set())  # each seat aims at a different voice

    def test_accion_de_canto(self):
        self.assertEqual(vj.accion_de_canto("REAL_ENVIDO"), vj.REAL_ENVIDO)
        self.assertEqual(vj.accion_de_canto("fold"), vj.MAZO)
        self.assertEqual(vj.accion_de_canto("VALE_CUATRO"), vj.VALE_CUATRO)
        self.assertIsNone(vj.accion_de_canto("otra"))


if __name__ == "__main__":
    unittest.main()
