"""Spectator mode, step 3: pacing over `ritmo`, the voice of each AI and the
gestures of both robots (ADR 0064, 0066, 0070)."""

from __future__ import annotations

import importlib.util
import random
import re
import threading
import time
import unittest
import urllib.request
from unittest import mock

import robot  # noqa: F401  (puts entorno/ on sys.path, as the server does)
import espectador as E
import ritmo as R
import voz_jugador as vj
from sim.mundo import Mundo

HAY_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _ev(kind, seat=0, **extra):
    return {'kind': kind, 'seat': seat, 'action': extra.pop('action', None), 'label': extra.pop('label', ''),
            'hand': 0, **extra}


class TestRitmoEspectador(unittest.TestCase):
    def setUp(self):
        self.base = R.Ritmo.humano(1.0)
        self.r = E.RitmoEspectador(rng=random.Random(4))

    def test_pensar_normal_y_grande(self):
        lo, hi = self.base.pensar
        glo, ghi = self.base.pensar_grande
        for _ in range(20):
            d = self.r.demora(_ev('think', 0, legales=4, decision={'accion': 'card_0', 'pendiente': None}))
            self.assertTrue(lo <= d <= hi, d)
            d = self.r.demora(_ev('think', 0, legales=4, decision={'accion': 'truco', 'pendiente': None}))
            self.assertTrue(glo <= d <= ghi, d)
            # Accepting a retruco is a big decision even though "quiero" is not.
            d = self.r.demora(_ev('think', 0, legales=4, decision={'accion': 'quiero', 'pendiente': 'retruco'}))
            self.assertTrue(glo <= d <= ghi, d)

    def test_una_cadena_solo_sigue(self):
        slo, shi = self.base.seguir
        self.r.demora(_ev('respond', 1, action='quiero'))
        d = self.r.demora(_ev('think', 1, legales=3, decision={'accion': 'card_1', 'pendiente': None}))
        self.assertTrue(slo <= d <= shi, d)
        # The other AI thinking is a new decision; so is the next one after a card.
        self.assertGreater(self.r.demora(_ev('think', 0, legales=3, decision={'accion': 'card_0'})), shi)
        self.r.demora(_ev('play', 1, carta={'valor': 1, 'palo': 'Oro'}))
        self.assertGreater(self.r.demora(_ev('think', 1, legales=3, decision={'accion': 'card_0'})), shi)
        # Nothing to decide: a short pause.
        self.assertLessEqual(self.r.demora(_ev('think', 0, legales=1)), shi)

    def test_carta_baza_y_mano(self):
        self.r.demora(_ev('deal', None))
        carta = self.base.retraso(R.CARTA)
        self.assertAlmostEqual(self.r.demora(_ev('play', 0)), carta)
        self.assertAlmostEqual(self.r.demora(_ev('play', 1)), carta + self.base.retraso(R.BAZA))
        self.assertAlmostEqual(self.r.demora(_ev('hand_end', None)), self.base.retraso(R.MANO))
        self.assertEqual(self.r.demora(_ev('match_end', 0)), 0.0)
        # The 3D card timing is the same one the human match and the simulator use.
        self.assertAlmostEqual(carta, self.base.alcanzar + self.base.colocar)

    def test_el_reloj_ve_la_decision_y_la_pagina_no(self):
        p = E.PartidaEspectador(('reactivo', 'utilidad'), seed=3)
        p.iniciar()
        while True:
            ev = p.siguiente_evento()
            if ev['kind'] == 'think':
                break
        visto = p.para_ritmo(ev)
        self.assertIn('accion', visto['decision'])
        self.assertNotIn('decision', ev)
        self.assertNotIn('decision', p.eventos[-1])
        p.cancelar()

    def test_es_el_ritmo_por_defecto_y_la_velocidad_lo_escala(self):
        r = E.Reproductor(oyentes=[])
        try:
            self.assertIsInstance(r.ritmo, E.RitmoEspectador)
            self.assertEqual(E.VELOCIDADES, R.VELOCIDADES)
        finally:
            r.cerrar()


class TestVozEspectador(unittest.TestCase):
    def setUp(self):
        self.v = E.VozEspectador(rng=random.Random(2))

    def test_frases_por_evento(self):
        f = self.v.frase
        self.assertEqual(f(_ev('call', 0, action='truco', label='Truco')), '¡Truco!')
        self.assertEqual(f(_ev('respond', 1, action='retruco', responde_a='truco')), '¡Quiero retruco!')
        self.assertEqual(f(_ev('respond', 1, action='quiero', responde_a='envido')), '¡Quiero!')
        self.assertEqual(f(_ev('respond', 1, action='no_quiero', responde_a='truco')), 'No quiero.')
        self.assertEqual(f(_ev('respond', 1, action='envido', responde_a='envido')), '¡Envido, envido!')
        self.assertEqual(f(_ev('respond', 1, action='envido', responde_a='truco')), '¡Envido!')  # el envido está primero
        self.assertEqual(f(_ev('fold', 0, action='fold')), 'Me voy al mazo.')
        self.assertEqual(f(_ev('tanto', 0, tanto=33, es_mano=True)), 'Treinta y tres.')
        self.assertEqual(f(_ev('tanto', 1, tanto='son buenas', es_mano=False)), 'Son buenas.')
        self.assertIsNone(f(_ev('think', 0)))
        self.assertIsNone(f(_ev('hand_end', None)))

    def test_el_tanto_sigue_la_regla_del_anuncio(self):
        """Over many matches: the mano says its number, the pie only when it beats it."""
        for seed in range(6):
            p = E.PartidaEspectador(('reactivo', 'utilidad'), seed=seed)
            p.iniciar()
            p.jugar_hasta_el_final()
            eventos = p.eventos
            for i, ev in enumerate(eventos):
                if ev['kind'] != 'tanto' or not ev['es_mano']:
                    continue
                pie = eventos[i + 1]
                self.assertEqual(pie['kind'], 'tanto')
                if isinstance(pie['tanto'], int):
                    self.assertGreater(pie['tanto'], ev['tanto'])
                    esperado = vj.frase_tanto(pie['tanto'], ev['tanto'], robot_es_mano=False)
                else:
                    esperado = 'Son buenas.'
                self.assertEqual(self.v.frase(pie), esperado)

    def test_una_partida_entera_solo_dice_frases_de_jugador(self):
        voz = E.VozEspectador(rng=random.Random(1))
        voz.MAXIMO = 100000                               # keep the whole match
        r = E.Reproductor(ritmo=E.RitmoSimple(escala=0), oyentes=[voz])
        try:
            r.empezar(('reactivo', 'utilidad'), seed=7)
            fin = time.monotonic() + 30
            while r.estado != 'terminada' and time.monotonic() < fin:
                time.sleep(0.01)
            vistas = {f['id']: f for f in r.snapshot()['voz']}
        finally:
            r.cerrar()
        self.assertTrue(vistas)
        catalogo = set(vj.todas_las_frases())
        for f in vistas.values():
            self.assertIn(f['frase'], catalogo)
            self.assertIn(f['asiento'], (1, 2))
            self.assertEqual(f['perfil'], vj.perfil_voz(f['asiento']))
            self.assertNotRegex(f['frase'].lower(), r'farol|mentira|tengo|\d')
        ids = sorted(vistas)
        self.assertEqual(ids, list(range(ids[0], ids[-1] + 1)))       # one id per phrase, no gaps

    def test_lo_que_se_dice_nunca_es_el_razonamiento(self):
        self.v(_ev('call', 0, action='truco', razonamiento={'tipo': 'red', 'nota': 'Tengo 33 y el ancho'}))
        self.assertEqual([f['frase'] for f in self.v.snapshot()['voz']], ['¡Truco!'])


class RobotConGestos:
    def __init__(self):
        self.llamadas = []
        self._lock = threading.Lock()

    def _rec(self, *a):
        with self._lock:
            self.llamadas.append(a)

    def desafiar(self, esperar=True, asiento=1): self._rec('desafiar', esperar, asiento)
    def gesto_mazo(self, esperar=True, asiento=1): self._rec('gesto_mazo', esperar, asiento)
    def tirar_carta(self, carta, asiento=1): self._rec('tirar_carta', asiento)


class TestGestos(unittest.TestCase):
    def _esperar(self, cond):
        fin = time.monotonic() + 3
        while not cond() and time.monotonic() < fin:
            time.sleep(0.01)
        self.assertTrue(cond())

    def test_los_dos_robots_hacen_gestos_al_cantar(self):
        robot = RobotConGestos()
        mesa = E.MesaRemota(robot)
        g = E.GestosEspectador(mesa)
        g(_ev('call', 0, action='truco'))
        g(_ev('call', 1, action='real_envido'))
        g(_ev('respond', 1, action='no_quiero', responde_a='truco'))
        g(_ev('fold', 0, action='fold'))
        g(_ev('respond', 0, action='quiero'))          # no gesture
        g(_ev('play', 1, carta={'valor': 1, 'palo': 'Oro'}))
        self._esperar(lambda: len(robot.llamadas) == 4)
        self.assertEqual(robot.llamadas, [('desafiar', False, 1), ('desafiar', False, 2),
                                          ('gesto_mazo', False, 2), ('gesto_mazo', False, 1)])

    def test_el_robot_manda_el_gesto_del_asiento_2_solo_al_simulador_local(self):
        from sim.robot import Robot
        r = Robot.__new__(Robot)
        cliente = mock.MagicMock(GESTO_POR_ASIENTO=True)
        r._cliente = cliente
        r.verificar_estado = lambda: None
        r.desafiar(esperar=False, asiento=2)
        cliente.Gesto.assert_called_once_with('desafiando', duracion=1.8, asiento=2)
        cliente = mock.MagicMock(GESTO_POR_ASIENTO=False)      # DDS / real robot: no seat 2
        r._cliente = cliente
        r.gesto_mazo(esperar=False, asiento=2)
        cliente.Gesto.assert_not_called()
        r.desafiar(esperar=False)
        cliente.Gesto.assert_called_once_with('desafiando', duracion=1.8)

    def test_el_mundo_guarda_el_gesto_del_rival(self):
        m = Mundo()
        m.gesto_rival('desafiando', 0.2)
        self.assertEqual(m.leer()['accion_rival'], 'desafiando')
        self.assertEqual(m.leer()['accion'], 'quieto')            # Ottoman does not move
        m.gesto_rival('sentarse', 1.0)                            # not a gesture for the seated rival
        self.assertEqual(m.leer()['accion_rival'], 'desafiando')
        time.sleep(0.25)
        m.avanzar()
        self.assertEqual(m.leer()['accion_rival'], 'quieto')

    def test_el_simulador_local_dirige_el_gesto_al_asiento(self):
        from sim.local import ServidorLocal
        srv = ServidorLocal.__new__(ServidorLocal)
        srv.mundo = Mundo()
        srv._orden_gesto({'orden': 'gesto', 'nombre': 'mazo', 'duracion': 5, 'asiento': 2})
        self.assertEqual(srv.mundo.leer()['accion_rival'], 'mazo')
        self.assertEqual(srv.mundo.leer()['accion'], 'quieto')
        srv._orden_gesto({'orden': 'gesto', 'nombre': 'desafiando', 'duracion': 5})
        self.assertEqual(srv.mundo.leer()['accion'], 'desafiando')


@unittest.skipUnless(HAY_MUJOCO, "MuJoCo no instalado")
class TestGestoDelRobot2EnLaMesa(unittest.TestCase):
    def test_con_cartas_levanta_los_dos_brazos_con_el_abanico(self):
        import numpy as np
        from sim import cartas as C
        from sim.capturas import Fotografo
        from sim.escena_cartas import ELEVACION_GESTO
        from sim.robots import G1_L_SHOULDER_P, G1_R_SHOULDER_P
        f = Fotografo(320, 240)
        f.mesa.poner_modo(C.MODO_ROBOTS)
        f.mesa.repartir({1: [C.Carta(1, "Espada"), C.Carta(7, "Oro"), C.Carta(12, "Copa")],
                         2: [C.Carta(3, "Basto"), C.Carta(1, "Basto"), C.Carta(10, "Espada")]})
        f.correr(1.5)
        rb = f.sim.cartas.robots[2]
        q = f.sim.data.qpos
        quieto = (q[rb.qpos_art + G1_L_SHOULDER_P], q[rb.qpos_art + G1_R_SHOULDER_P])
        ottoman = q[f.sim.cartas.robots[1].qpos_art + G1_R_SHOULDER_P]
        f.sim.mundo.gesto_rival("desafiando", 30.0)
        f.correr(1.0)
        izq, der = q[rb.qpos_art + G1_L_SHOULDER_P], q[rb.qpos_art + G1_R_SHOULDER_P]
        self.assertAlmostEqual(izq, rb.pose_sostener[G1_L_SHOULDER_P] - ELEVACION_GESTO, places=2)
        self.assertAlmostEqual(der, rb.pose_sostener_der[G1_R_SHOULDER_P] - ELEVACION_GESTO, places=2)
        self.assertLess(izq, quieto[0] - 0.3)
        self.assertLess(der, quieto[1] - 0.3)
        self.assertAlmostEqual(q[f.sim.cartas.robots[1].qpos_art + G1_R_SHOULDER_P], ottoman, places=3)
        # The fan went up with the hands: still between the palms.
        d, e = f.sim.data, f.sim.cartas
        centros = [d.mocap_pos[e.mocap[c.cuerpo]] for c in f.mesa.estado() if c.asiento == 2 and c.lugar == "mano"]
        centro = np.mean(centros, axis=0)
        self.assertLess(np.linalg.norm(d.site_xpos[rb.sitio_izq] - centro), 0.12)
        self.assertLess(np.linalg.norm(d.site_xpos[rb.sitio_der] - centro), 0.12)


class TestPaginaEspectadorHabla(unittest.TestCase):
    def test_la_pagina_usa_el_script_de_voz_compartido(self):
        import servidor_truco
        from pathlib import Path
        texto = servidor_truco.PAGINA_ESPECTADOR.read_text(encoding='utf-8')
        self.assertIn('/voz_robots.js', texto)
        self.assertIn('voz.procesar(d.voz)', texto)
        self.assertIn('/voz_robots.js', servidor_truco.PAGINA.read_text(encoding='utf-8'))
        js = servidor_truco.SCRIPT_VOZ.read_text(encoding='utf-8')
        self.assertIn('window.VozRobots', js)
        self.assertNotRegex(js.lower(), r'farol')
        self.assertTrue(Path(servidor_truco.SCRIPT_VOZ).is_file())


if __name__ == "__main__":
    unittest.main()
