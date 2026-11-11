"""Spectator mode (ADR 0064, 0069): driver, engine length hooks, endpoints,
the human-mode length selector and the two-robot 3D scene."""

from __future__ import annotations

import importlib.util
import json
import random
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

import robot  # noqa: F401  (puts entorno/ on sys.path, as the server does)
import espectador as E
from engine_policies import RandomPolicy, ReactivePolicy
from symmetric_engine import play_match

HAY_TORCH = importlib.util.find_spec("torch") is not None
HAY_MUJOCO = importlib.util.find_spec("mujoco") is not None
REPO = Path(__file__).resolve().parent.parent


def _partida(ias, largo=E.Largo(), seed=1, mesa=None):
    p = E.PartidaEspectador(ias, largo, seed, mesa)
    p.iniciar()
    return p


class TestMotorLargo(unittest.TestCase):
    """The engine's new optional hooks never change a match played without them."""

    def test_sin_ganchos_la_partida_es_identica(self):
        a = play_match(ReactivePolicy(1), RandomPolicy(2), seed=11)
        eventos = []
        b = play_match(ReactivePolicy(1), RandomPolicy(2), seed=11, on_event=eventos.append)
        self.assertEqual((a.points, a.deals), (b.points, b.deals))
        self.assertEqual([d['action'] for d in a.decisions], [d['action'] for d in b.decisions])
        self.assertEqual([e['kind'] for e in eventos].count('deal'), len(b.deals))
        self.assertEqual([e['kind'] for e in eventos].count('hand_end'), len(b.deals))
        self.assertEqual(eventos[-1]['score'], b.points)

    def test_n_manos_y_desempate(self):
        for seed in range(25):
            r = play_match(RandomPolicy(seed), RandomPolicy(seed + 1), seed=seed, target=30, max_hands=2)
            self.assertNotEqual(r.points[0], r.points[1])
            if len(r.deals) > 2:
                # Extra hands only while level after the second one.
                self.assertTrue(max(r.points) < 30)
            self.assertGreaterEqual(len(r.deals), 2 if max(r.points) < 30 else 1)

    def test_max_hands_invalido(self):
        for malo in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                play_match(RandomPolicy(1), RandomPolicy(2), max_hands=malo)


class TestReglaEnvido(unittest.TestCase):
    OBS = {'player': 0, 'tricks': (), 'table': ((0, (4, 'Copa')),), 'pending': None}
    LEGAL = ('card_1', 'card_2', 'fold', 'envido', 'real_envido', 'falta_envido', 'truco')

    def test_quien_ya_tiro_no_canta_envido(self):
        self.assertEqual(E.sin_envido_tras_primera_carta(self.OBS, self.LEGAL),
                         ('card_1', 'card_2', 'fold', 'truco'))

    def test_tampoco_el_envido_va_primero(self):
        obs = dict(self.OBS, pending={'kind': 'truco', 'caller': 1, 'level': 2})
        legal = ('quiero', 'no_quiero', 'fold', 'retruco', 'envido', 'real_envido', 'falta_envido')
        self.assertNotIn('envido', E.sin_envido_tras_primera_carta(obs, legal))

    def test_el_pie_que_no_tiro_si_puede_y_se_puede_subir_un_envido(self):
        pie = dict(self.OBS, player=1)
        self.assertEqual(E.sin_envido_tras_primera_carta(pie, self.LEGAL), self.LEGAL)
        subir = dict(self.OBS, pending={'kind': 'envido', 'caller': 1, 'call': 'envido'})
        legal = ('quiero', 'no_quiero', 'fold', 'envido', 'real_envido', 'falta_envido')
        self.assertEqual(E.sin_envido_tras_primera_carta(subir, legal), legal)

    def test_en_partidas_nadie_canta_envido_despues_de_tirar(self):
        for seed in range(12):
            p = _partida(('reactivo', 'utilidad'), seed=seed)
            p.jugar_hasta_el_final()
            tiro = set()
            pendiente_envido = False
            for ev in p.eventos:
                if ev['kind'] == 'deal':
                    tiro, pendiente_envido = set(), False
                elif ev['kind'] == 'play' and len(tiro) < 2:
                    tiro.add(ev['seat'])
                elif ev['action'] in E.ENVIDOS:
                    if not pendiente_envido:
                        self.assertNotIn(ev['seat'], tiro, f'semilla {seed}')
                    pendiente_envido = True
                elif ev['kind'] in ('respond',) and ev['action'] in ('quiero', 'no_quiero'):
                    pendiente_envido = False


class TestPartidaEspectador(unittest.TestCase):
    def test_secuencia_legal_de_eventos(self):
        p = _partida(('reactivo', 'utilidad'), seed=3)
        p.jugar_hasta_el_final()
        ids = [e['id'] for e in p.eventos]
        self.assertEqual(ids, sorted(ids))
        for e in p.eventos:
            self.assertIn(e['kind'], E.KINDS)
            json.dumps(e)                                   # JSON-ready
        self.assertEqual(p.eventos[0]['kind'], 'deal')
        self.assertEqual(p.eventos[-1]['kind'], 'match_end')
        # Each action follows the thinking of the same seat.
        for previo, ev in zip(p.eventos, p.eventos[1:]):
            if ev['kind'] in E.ACCIONES_DE_JUEGO:
                self.assertEqual((previo['kind'], previo['seat']), ('think', ev['seat']))
        self.assertTrue(p.terminada)
        self.assertGreaterEqual(max(p.puntos), 15)
        self.assertEqual(p.ganador, p.puntos.index(max(p.puntos)))
        self.assertIsNone(p.siguiente_evento())

    def test_a_30(self):
        p = _partida(('reactivo', 'reactivo'), E.Largo.de_dict({'tipo': 'puntos', 'meta': 30}), seed=5)
        p.jugar_hasta_el_final()
        self.assertGreaterEqual(max(p.puntos), 30)
        self.assertLess(min(p.puntos), 30)

    def test_n_manos_con_desempate(self):
        vistos_desempate = False
        for seed in range(30):
            largo = E.Largo.de_dict({'tipo': 'manos', 'manos': 1})
            p = _partida(('reactivo', 'reactivo'), largo, seed=seed)
            p.jugar_hasta_el_final()
            self.assertNotEqual(p.puntos[0], p.puntos[1])
            manos = [e for e in p.eventos if e['kind'] == 'deal']
            self.assertEqual(manos[0]['desempate'], False)
            if len(manos) > 1:
                vistos_desempate = True
                self.assertTrue(all(m['desempate'] for m in manos[1:]))
                self.assertTrue(p.estado()['desempate'])
        self.assertTrue(vistos_desempate, 'some 1-hand match should need a tie-break')

    def test_largo_invalido(self):
        for malo in ({'tipo': 'manos', 'manos': 0}, {'tipo': 'manos', 'manos': 11},
                     {'tipo': 'puntos', 'meta': 20}, {'tipo': 'manos', 'manos': '3'}, None):
            with self.assertRaises(ValueError):
                E.Largo.de_dict(malo)

    def test_avanzar_una_jugada(self):
        p = _partida(('reactivo', 'utilidad'), seed=8)
        primeros = p.avanzar_jugada()
        self.assertEqual([e['kind'] for e in primeros][:2], ['deal', 'think'])
        self.assertEqual(primeros[-1]['kind'], 'think')
        siguientes = p.avanzar_jugada()
        acciones = [e for e in siguientes if e['kind'] in E.ACCIONES_DE_JUEGO]
        self.assertEqual(len(acciones), 1)
        self.assertIn(siguientes[-1]['kind'], ('think', 'match_end'))

    def test_razonamiento_de_cada_tipo(self):
        p = _partida(('utilidad', 'reactivo'), seed=2)
        p.jugar_hasta_el_final()
        tipos = {}
        for e in p.eventos:
            if e['kind'] == 'think':
                tipos.setdefault(e['seat'], e['razonamiento'])
        self.assertEqual(tipos[0]['tipo'], 'umbral')
        self.assertTrue(tipos[0]['metrica'] and tipos[0]['umbral'])
        self.assertEqual(tipos[1]['tipo'], 'regla')
        self.assertTrue(tipos[1]['regla'])

    def test_tantos_anunciados(self):
        for seed in range(20):
            p = _partida(('utilidad', 'utilidad'), seed=seed)
            p.jugar_hasta_el_final()
            for i, e in enumerate(p.eventos):
                if e['kind'] == 'tanto':
                    self.assertTrue(e['label'] == 'Son buenas' or e['label'].isdigit())
                    return
        self.fail('no envido was accepted in 20 matches')

    def test_la_mesa_3d_recibe_reparto_y_jugadas(self):
        robot_falso = MagicMock()
        mesa = E.MesaRemota(robot_falso)
        p = _partida(('reactivo', 'reactivo'), E.Largo('manos', 30, 1), seed=4, mesa=mesa)
        p.jugar_hasta_el_final()
        limite = time.monotonic() + 5
        jugadas = sum(1 for e in p.eventos if e['kind'] == 'play')
        while robot_falso.tirar_carta.call_count < jugadas and time.monotonic() < limite:
            time.sleep(0.02)
        robot_falso.modo_mesa.assert_any_call('robots')
        mano_1, mano_2 = robot_falso.repartir_cartas.call_args_list[0][0]
        self.assertEqual(len(mano_1), 3)
        self.assertEqual(len(mano_2), 3)
        asientos = [c[0][1] for c in robot_falso.tirar_carta.call_args_list]
        self.assertEqual(len(asientos), jugadas)
        self.assertTrue(set(asientos) <= {1, 2})


class TestModelos(unittest.TestCase):
    def test_nombres_publicados(self):
        sidecar = json.loads((REPO / 'champions/phase2/reigning_champion.json').read_text(encoding='utf-8'))
        self.assertEqual((sidecar['run'], sidecar['generation'], sidecar['selection']), ('run3', 73, 'greedy'))
        filas = [json.loads(l) for l in (REPO / 'champions/phase1/generations.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        self.assertEqual(filas[-1]['reigning_champion'], 48)
        self.assertEqual([m.nombre for m in E.MODELOS.values()],
                         ['Refuerzo corrida 3 · gen 73', 'Utilidad adaptativa', 'Genético publicado (gen 48)', 'Reactivo'])

    @unittest.skipUnless(HAY_TORCH, "PyTorch no instalado")
    def test_los_cuatro_juegan_una_partida_entera(self):
        for ias in (('refuerzo', 'genetico'), ('utilidad', 'reactivo'), ('refuerzo', 'refuerzo')):
            p = _partida(ias, seed=6)
            p.jugar_hasta_el_final()
            self.assertTrue(p.terminada, ias)
            self.assertIsNone(p.error)
            self.assertGreaterEqual(max(p.puntos), 15)
        red = next(e['razonamiento'] for e in p.eventos if e['kind'] == 'think')
        self.assertEqual(red['tipo'], 'red')
        self.assertLessEqual(len(red['opciones']), 3)
        self.assertIn('valor', red['opciones'][0])

    def test_genetico_da_probabilidades(self):
        p = _partida(('genetico', 'reactivo'), seed=2)
        ev = p.avanzar_jugada()[-1]
        self.assertEqual(ev['seat'], 0)
        probs = [o['probabilidad'] for o in ev['razonamiento']['opciones']]
        self.assertEqual(probs, sorted(probs, reverse=True))
        p.cancelar()


class RitmoLento:
    def demora(self, evento):
        return 100.0


class TestReproductor(unittest.TestCase):
    def setUp(self):
        self.r = E.Reproductor(ritmo=RitmoLento())

    def tearDown(self):
        self.r.cerrar()

    def _esperar(self, condicion):
        limite = time.monotonic() + 5
        while not condicion() and time.monotonic() < limite:
            time.sleep(0.01)
        self.assertTrue(condicion())

    def test_pausa_paso_velocidad(self):
        self.r.empezar(('reactivo', 'utilidad'), seed=1)
        self._esperar(lambda: self.r.partida.eventos)
        self.assertEqual(len(self.r.partida.eventos), 1)          # the deal, then a long wait
        self.r.pausar()
        self.assertEqual(self.r.estado, 'pausa')
        antes = len(self.r.partida.eventos)
        estado = self.r.paso()
        nuevos = self.r.partida.eventos[antes:]
        self.assertEqual(nuevos[-1]['kind'], 'think')
        antes = len(self.r.partida.eventos)
        estado = self.r.paso()
        acciones = [e for e in self.r.partida.eventos[antes:] if e['kind'] in E.ACCIONES_DE_JUEGO]
        self.assertEqual(len(acciones), 1)
        self.assertEqual(estado['estado'], 'pausa')
        self.assertEqual(self.r.poner_velocidad(2)['velocidad'], 2.0)
        with self.assertRaises(ValueError):
            self.r.poner_velocidad(3)
        self.assertEqual(self.r.seguir()['estado'], 'jugando')

    def test_hasta_el_final_revancha_y_cambiar(self):
        r = E.Reproductor(ritmo=E.RitmoSimple(escala=0))
        try:
            oidos = []
            r.oyentes.append(oidos.append)
            r.empezar(('reactivo', 'reactivo'), E.Largo('manos', 30, 1), seed=2)
            self._esperar(lambda: r.estado == 'terminada')
            snap = r.snapshot()
            self.assertTrue(snap['partida']['terminada'])
            self.assertEqual(oidos[-1]['kind'], 'match_end')
            semilla = r.partida.seed
            r.revancha()
            self.assertEqual(r.partida.ias, ('reactivo', 'reactivo'))
            self.assertNotEqual(r.partida.seed, semilla)
            self.assertEqual(r.detener()['estado'], 'sin_partida')
            self.assertIsNone(r.snapshot()['partida'])
        finally:
            r.cerrar()


class TestEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import servidor_truco
        from partida_truco import PartidaTruco
        cls.tmp = tempfile.TemporaryDirectory()
        partida = PartidaTruco(robot=None, historial_path=str(Path(cls.tmp.name) / 'h.json'), rng=random.Random(1))
        cls.srv = servidor_truco.crear_servidor(partida, 0, sin_robot=True)
        cls.hilo = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.hilo.start()
        cls.base = f'http://127.0.0.1:{cls.srv.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.srv.espectador.cerrar()
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.tmp.cleanup()

    def _get(self, ruta):
        with urllib.request.urlopen(self.base + ruta) as r:
            return r.status, r.read()

    def _post(self, ruta, datos):
        pedido = urllib.request.Request(self.base + ruta, json.dumps(datos).encode(), {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(pedido) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_pagina_y_modelos(self):
        estado, cuerpo = self._get('/espectador')
        self.assertEqual(estado, 200)
        self.assertIn('Modo espectador', cuerpo.decode('utf-8'))
        self.assertNotIn(b'farol', cuerpo.lower())
        modelos = json.loads(self._get('/espectador/modelos')[1])['modelos']
        self.assertEqual([m['clave'] for m in modelos], ['refuerzo', 'utilidad', 'genetico', 'reactivo'])
        self.assertIn(b'/espectador', self._get('/')[1])                # linked from the main page

    def test_validacion(self):
        for datos in ({'ias': ['reactivo']}, {'ias': ['reactivo', 'nada']},
                      {'ias': ['reactivo', 'reactivo'], 'largo': {'tipo': 'manos', 'manos': 12}}):
            self.assertEqual(self._post('/espectador/empezar', datos)[0], 400)
        self.assertEqual(self._post('/espectador/velocidad', {'velocidad': 5})[0], 400)
        self.assertEqual(self._post('/espectador/camara', {'nombre': 'cenital'})[0], 400)
        self.assertEqual(self._post('/espectador/nada', {})[0], 404)

    def test_una_partida_por_la_web(self):
        estado, d = self._post('/espectador/empezar', {'ias': ['reactivo', 'utilidad'],
                                                        'largo': {'tipo': 'manos', 'manos': 1}})
        self.assertEqual(estado, 200)
        self.assertEqual(d['partida']['largo_texto'], '1 mano')
        self.assertEqual(self._post('/espectador/pausa', {})[1]['estado'], 'pausa')
        d = self._post('/espectador/paso', {})[1]
        self.assertTrue(d['eventos'])
        self.assertEqual(self._post('/espectador/velocidad', {'velocidad': 0.5})[1]['velocidad'], 0.5)
        ultimo = d['eventos'][-1]['id']
        nuevos = json.loads(self._get(f'/espectador/estado?desde={ultimo}')[1])
        self.assertTrue(all(e['id'] > ultimo for e in nuevos['eventos']))
        self.assertEqual(self._post('/espectador/revancha', {})[0], 200)
        self.assertEqual(self._post('/espectador/detener', {})[1]['estado'], 'sin_partida')

    def test_partida_corta_contra_el_humano(self):
        estado, d = self._post('/truco/reiniciar', {'manos': 3})
        self.assertEqual(estado, 200)
        self.assertEqual(d['estado']['tanteador']['manos'], 3)
        self.assertEqual(self._post('/truco/reiniciar', {'manos': 11})[0], 400)
        d = self._post('/truco/reiniciar', {'meta_puntos': 15})[1]
        self.assertIsNone(d['estado']['tanteador']['manos'])


class TestPartidaCortaHumana(unittest.TestCase):
    def _partida(self, tmp):
        from partida_truco import PartidaTruco
        return PartidaTruco(robot=None, historial_path=str(Path(tmp) / 'h.json'), rng=random.Random(3))

    def test_termina_tras_n_manos_quien_suma_mas(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._partida(tmp)
            p.reiniciar_partido(manos=1)
            self.assertEqual((p.manos_partida, p.meta_puntos), (1, 30))
            p._adjudicar_mano('humano', 2)
            self.assertTrue(p.partido_terminado)
            self.assertEqual(p.ganador_partido, 'humano')

    def test_empate_juega_otra_mano(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._partida(tmp)
            p.reiniciar_partido(manos=1)
            p.puntos_robot = 1                       # e.g. the robot won the envido
            p._adjudicar_mano('humano', 1)
            self.assertFalse(p.partido_terminado)    # 1 a 1: tie-break hand
            p.nueva_mano()
            p._adjudicar_mano('robot', 2)
            self.assertTrue(p.partido_terminado)
            self.assertEqual(p.ganador_partido, 'robot')

    def test_volver_a_puntos(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._partida(tmp)
            p.reiniciar_partido(manos=4)
            p.reiniciar_partido(meta_puntos=15)
            self.assertEqual((p.manos_partida, p.meta_puntos), (None, 15))
            p._adjudicar_mano('humano', 2)
            self.assertFalse(p.partido_terminado)


class TestEscenaDosRobots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sim.robots import G1
        cls.ruta = Path(G1.ruta_escena())
        cls.raiz = ET.parse(cls.ruta).getroot()

    def test_el_xml_trae_al_segundo_g1(self):
        from sim.robots import PREFIJO_RIVAL
        modelos = {m.get('name'): m.get('file') for m in self.raiz.iter('model')}
        self.assertIn('rival', modelos)
        attach = next(self.raiz.iter('attach'))
        self.assertEqual((attach.get('model'), attach.get('body'), attach.get('prefix')),
                         ('rival', 'pelvis', PREFIJO_RIVAL))
        rival = ET.parse(self.ruta.parent / modelos['rival']).getroot()
        self.assertIsNone(rival.find('actuator'))       # kinematic: no motors
        colores = {g.get('rgba') for g in rival.iter('geom')}
        self.assertIn('0.42 0.68 0.92 1', colores)       # light-blue poncho
        self.assertNotIn('0.46 0.11 0.09 1', colores)    # no red vest

    @unittest.skipUnless(HAY_MUJOCO, "MuJoCo no instalado")
    def test_carga_con_dos_robots_y_los_motores_de_uno(self):
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(self.ruta))
        self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'r2_floating_base_joint'), 0)
        for n in ('r2_mano_izq', 'r2_mano_der', 'mano_izq'):
            self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n), 0)
        self.assertEqual(m.nu, 29)


@unittest.skipUnless(HAY_MUJOCO, "MuJoCo no instalado")
class TestEscenaDosRobotsMuJoCo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sim.capturas import Fotografo
        cls.f = Fotografo(160, 120)

    def tearDown(self):
        from sim import cartas as C
        self.f.mesa.poner_modo(C.MODO_HUMANO)

    def test_el_rival_se_sienta_sostiene_y_juega_sus_cartas(self):
        import numpy as np
        from sim import cartas as C
        f = self.f
        e = f.sim.cartas
        rival = e.robots[2]
        f.mesa.poner_modo(C.MODO_ROBOTS)
        mano = [C.Carta(3, 'Basto'), C.Carta(1, 'Basto'), C.Carta(10, 'Espada')]
        f.mesa.repartir({1: [C.Carta(1, 'Espada'), C.Carta(7, 'Oro'), C.Carta(12, 'Copa')], 2: mano})
        f.correr(0.6)
        base = f.sim.data.qpos[rival.qpos_base:rival.qpos_base + 3]
        self.assertAlmostEqual(base[0], C.ASIENTO_X[2], places=3)
        self.assertGreater(base[2], 0.3)
        palma = f.sim.data.site_xpos[rival.sitio_izq]
        for c in f.mesa.estado():
            if c.asiento == 2:
                self.assertEqual(c.lugar, 'mano')
                self.assertLess(np.linalg.norm(f.sim.data.mocap_pos[e.mocap[c.cuerpo]] - palma), 0.12)
                # Robot vs robot: faces are shown.
                self.assertEqual(f.sim.model.geom_matid[e.geom_frente[c.cuerpo]], e.material[c.carta])
        f.mesa.jugar(2, mano[0])
        r = f.mesa.ritmo
        f.correr(r.alcanzar + r.llevar / 2)
        der = f.sim.data.site_xpos[rival.sitio_der].copy()
        viaje = next(c for c in f.mesa.estado() if c.carta == mano[0])
        self.assertEqual(viaje.lugar, 'viaje')
        self.assertLess(np.linalg.norm(f.sim.data.mocap_pos[e.mocap[viaje.cuerpo]] - der), 0.15)
        f.correr(r.llevar / 2 + r.volver + 0.2)
        pos, _ = C.slot_mesa(2, 0)
        np.testing.assert_allclose(f.sim.data.mocap_pos[e.mocap[viaje.cuerpo]], pos, atol=1e-6)

    def test_contra_un_humano_el_rival_no_esta_en_la_mesa(self):
        from sim import cartas as C
        f = self.f
        f.mesa.poner_modo(C.MODO_HUMANO)
        f.mesa.repartir({1: [C.Carta(1, 'Espada')], 2: [C.Carta(3, 'Basto')]})
        f.correr(0.2)
        rival = f.sim.cartas.robots[2]
        self.assertLess(f.sim.data.qpos[rival.qpos_base + 2], -1.0)
        self.assertFalse(any(c.asiento == 2 and c.lugar == 'mano' for c in f.mesa.estado()))
        self.assertEqual(f.mesa.elegir_camara('mesa'), 'general')


if __name__ == '__main__':
    unittest.main()
