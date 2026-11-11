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
from unittest import mock
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


class TestFaltaEnvidoEnPartidaCorta(unittest.TestCase):
    """ADR 0069 (user decision Q26 = A): with the internal target of 30, an
    accepted falta envido ends a short match and its winner goes to 30."""

    class _Canta:
        def __call__(self, obs, legal):
            return 'falta_envido' if 'falta_envido' in legal else legal[0]

    class _Quiere:
        def __call__(self, obs, legal):
            return 'quiero' if 'quiero' in legal else legal[0]

    def test_motor(self):
        for manos in (1, 5, 10):
            r = play_match(self._Canta(), self._Quiere(), seed=manos, target=30, max_hands=manos)
            self.assertEqual(len(r.deals), 1, manos)
            self.assertEqual(max(r.points), 30)
            self.assertLess(min(r.points), 30)

    def test_regla_de_la_falta(self):
        import truco_reglas as rules
        for lider, otro in ((0, 0), (3, 1), (14, 9)):
            for ganador, propios in (('humano', lider), ('robot', otro)):
                puntos, por_el_partido = rules.puntos_falta_envido(lider, otro, 30, ganador)
                self.assertTrue(por_el_partido)
                self.assertEqual(propios + puntos, 30)

    def test_en_el_espectador(self):
        with mock.patch.object(E, 'crear_politica', side_effect=[self._Canta(), self._Quiere()]):
            p = _partida(('reactivo', 'reactivo'), E.Largo('manos', 30, 5), seed=1)
        p.jugar_hasta_el_final()
        self.assertEqual([e['kind'] for e in p.eventos].count('deal'), 1)
        self.assertEqual(max(p.puntos), 30)


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
            if e['kind'] in E.ACCIONES_DE_JUEGO:
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
        red = next(e['razonamiento'] for e in p.eventos if e['kind'] in E.ACCIONES_DE_JUEGO)
        self.assertEqual(red['tipo'], 'red')
        self.assertLessEqual(len(red['opciones']), 4)
        self.assertIn('valor', red['opciones'][0])

    def test_genetico_da_probabilidades(self):
        p = _partida(('genetico', 'reactivo'), seed=2)
        self.assertEqual(p.avanzar_jugada()[-1]['seat'], 0)
        ev = next(e for e in p.avanzar_jugada() if e['kind'] in E.ACCIONES_DE_JUEGO)
        self.assertEqual(ev['seat'], 0)
        opciones = ev['razonamiento']['opciones']
        probs = [o['probabilidad'] for o in opciones[:3]]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertIn(ev['action'], [o['accion'] for o in opciones])    # the chosen one is always shown
        p.cancelar()


class TestPanelNoAdelantaLaJugada(unittest.TestCase):
    """While an AI is thinking, nothing the page receives tells what it chose."""

    def test_el_razonamiento_llega_con_la_jugada(self):
        p = _partida(('utilidad', 'reactivo'), seed=3)
        vistos = 0
        while True:
            ev = p.siguiente_evento()
            if ev is None:
                break
            if ev['kind'] == 'think':
                self.assertNotIn('razonamiento', ev)
                self.assertNotIn('razonamiento', p.ultimo_evento)
                panel = p.estado()['asientos'][ev['seat']]['razonamiento']
                self.assertEqual((panel['tipo'], panel['decision']), ('pensando', None))
                self.assertEqual(set(panel) - {'tipo', 'decision', 'envido_bloqueado'}, set())
                vistos += 1
            elif ev['kind'] in E.ACCIONES_DE_JUEGO:
                self.assertIn(ev['razonamiento']['tipo'], ('umbral', 'regla'))
                panel = p.estado()['asientos'][ev['seat']]['razonamiento']
                self.assertEqual(panel['decision'], ev['label'])
        self.assertGreater(vistos, 5)

    def test_el_snapshot_durante_la_pausa_tampoco(self):
        r = E.Reproductor(ritmo=RitmoLento())
        try:
            r.empezar(('reactivo', 'utilidad'), seed=4)
            r.pausar()
            snap = r.snapshot()
            while not snap['eventos'] or snap['eventos'][-1]['kind'] != 'think':
                snap = r.paso()
            self.assertNotIn('razonamiento', snap['ultimo_evento'])
            self.assertTrue(all('razonamiento' not in e for e in snap['eventos'] if e['kind'] == 'think'))
            seat = snap['ultimo_evento']['seat']
            self.assertEqual(snap['partida']['asientos'][seat]['razonamiento']['tipo'], 'pensando')
        finally:
            r.cerrar()


class TestExplicacionCoherente(unittest.TestCase):
    class _Politica:
        def __init__(self, traza):
            self.agent = MagicMock(traza_razonamiento=traza)

    def test_no_quiero_no_dice_que_se_va_al_mazo(self):
        traza = [{'decision': 'NO_QUIERO', 'fase': 'Respuesta a TRUCO', 'metrica': 'Fuerza', 'umbral': 'Umbral 44%',
                  'explicacion': 'Fuerza (20%) inferior al umbral (44.0%): se va al mazo cediendo 1 pto.'}]
        r = E._explicar_agente(self._Politica(traza), 'umbral', 'no_quiero')
        self.assertNotIn('mazo', r['explicacion'])
        self.assertIn('no quiere', r['explicacion'])
        regla = [{'decision': 'NO_QUIERO', 'explicacion': 'Regla reactiva: se va al mazo porque 5 < 8.'}]
        self.assertNotIn('mazo', E._explicar_agente(self._Politica(regla), 'regla', 'no_quiero')['regla'])
        # Going to the deck for real keeps its words.
        self.assertIn('mazo', E._texto_coherente('se va al mazo', 'fold'))

    def test_usa_el_paso_que_eligio_la_accion(self):
        traza = [{'decision': 'Jugar 4 de Copa', 'explicacion': 'carta'},
                 {'decision': 'NO CANTAR', 'explicacion': 'no canta truco'}]
        self.assertEqual(E._explicar_agente(self._Politica(traza), 'umbral', 'card_1')['explicacion'], 'carta')
        self.assertEqual(E._explicar_agente(self._Politica(traza), 'umbral', 'truco')['explicacion'], 'no canta truco')

    @unittest.skipUnless(HAY_TORCH, "PyTorch no instalado")
    def test_refuerzo_explica_sin_segunda_pasada(self):
        import phase2_torch
        original = phase2_torch._q_values
        llamadas = []

        def contar(*args):
            llamadas.append(1)
            return original(*args)

        with mock.patch.object(phase2_torch, '_q_values', contar):
            p = _partida(('refuerzo', 'reactivo'), E.Largo('manos', 30, 1), seed=5)
            p.jugar_hasta_el_final()
        decisiones = sum(1 for e in p.eventos if e['kind'] == 'think' and e['seat'] == 0)
        self.assertEqual(len(llamadas), decisiones)
        self.assertIsNone(p.error)


class TestModeloQueNoCarga(unittest.TestCase):
    def test_torch_que_falla_al_importar(self):
        importar = __import__

        def falso(nombre, *args, **kwargs):
            if nombre == 'torch':
                raise OSError('[WinError 126] No se puede encontrar el módulo especificado')
            return importar(nombre, *args, **kwargs)

        with mock.patch('builtins.__import__', falso):
            self.assertFalse(E.modelo_disponible('refuerzo'))
            modelos = {m['clave']: m['disponible'] for m in E.lista_modelos()}
        self.assertFalse(modelos['refuerzo'])
        self.assertTrue(modelos['reactivo'] and modelos['utilidad'])

    def test_el_error_nombra_la_ia(self):
        crear = E.crear_politica

        def falla(clave, *args):
            if clave == 'genetico':
                raise OSError('[WinError 126]')
            return crear(clave, *args)

        with mock.patch.object(E, 'crear_politica', falla):
            with self.assertRaises(E.ModeloNoCargado) as ctx:
                E.PartidaEspectador(('reactivo', 'genetico'), seed=1)
        self.assertEqual(ctx.exception.clave, 'genetico')


class TestMotorQueSeCae(unittest.TestCase):
    """The engine thread dying can never leave the driver waiting under its lock."""

    def _en_hilo(self, funcion, limite=5):
        resultado = []
        hilo = threading.Thread(target=lambda: resultado.append(funcion()), daemon=True)
        hilo.start()
        hilo.join(limite)
        self.assertFalse(hilo.is_alive(), f'{funcion} hung')
        return resultado[0]

    def test_excepcion_base_en_el_motor(self):
        def explota(*args, **kwargs):
            raise SystemExit('motor caído')

        with mock.patch.object(E, 'play_match', explota):
            p = _partida(('reactivo', 'reactivo'))
            self._en_hilo(p.jugar_hasta_el_final)
        self.assertTrue(p.terminada)
        self.assertIn('motor caído', p.error)
        self.assertEqual(p.eventos[-1]['kind'], 'match_end')

    def test_hilo_que_muere_sin_avisar(self):
        class Muda(E.PartidaEspectador):
            ESPERA_MOTOR = 0.05

            def _correr(self):
                pass

        p = Muda(('reactivo', 'reactivo'), seed=1)
        self._en_hilo(p.jugar_hasta_el_final)
        self.assertTrue(p.terminada)
        self.assertIsNotNone(p.error)

    def test_el_reproductor_sigue_respondiendo(self):
        def explota(*args, **kwargs):
            raise KeyboardInterrupt()

        r = E.Reproductor(ritmo=E.RitmoSimple(escala=0))
        try:
            with mock.patch.object(E, 'play_match', explota):
                self._en_hilo(lambda: r.empezar(('reactivo', 'reactivo'), seed=1))
                self._en_hilo(r.paso)
                self.assertTrue(self._en_hilo(r.snapshot)['partida']['terminada'])
                self._en_hilo(r.detener)
            self.assertEqual(self._en_hilo(lambda: r.empezar(('reactivo', 'reactivo'), seed=2))['estado'],
                             'jugando')
        finally:
            r.cerrar()


class RobotGrabador:
    """Records the 3D calls in the order they land; one call can be made slow."""

    def __init__(self, lento=None):
        self.llamadas, self.lento = [], lento
        self._lock = threading.Lock()

    def _llamar(self, *args):
        if args == self.lento:
            time.sleep(0.3)
        with self._lock:
            self.llamadas.append(args)

    def conectar(self): pass
    def modo_mesa(self, modo): self._llamar('modo_mesa', modo)
    def sentarse(self): self._llamar('sentarse')
    def elegir_camara(self, nombre): self._llamar('elegir_camara', nombre)
    def repartir_cartas(self, *manos): self._llamar('repartir_cartas')
    def tirar_carta(self, carta, asiento=1): self._llamar('tirar_carta', asiento)


class TestMesaRemota(unittest.TestCase):
    def _servidor(self, sin_robot=True):
        import servidor_truco
        from partida_truco import PartidaTruco
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        partida = PartidaTruco(robot=None, historial_path=str(Path(tmp.name) / 'h.json'), rng=random.Random(1))
        srv = servidor_truco.crear_servidor(partida, 0, sin_robot=sin_robot)
        self.addCleanup(srv.server_close)
        self.addCleanup(lambda: srv._espectador is not None and srv._espectador.cerrar())
        return srv

    def _esperar(self, condicion, limite=5):
        fin = time.monotonic() + limite
        while not condicion() and time.monotonic() < fin:
            time.sleep(0.01)
        self.assertTrue(condicion())

    def test_soltar_vacia_la_cola_y_espera_lo_que_esta_en_vuelo(self):
        robot = RobotGrabador(lento=('modo_mesa', 'robots'))
        mesa = E.MesaRemota(robot)
        mesa.empezar()                          # modo_mesa('robots') goes in flight, slowly
        mesa.camara('detras_robot_1')
        time.sleep(0.05)
        self.assertTrue(mesa.soltar())
        self.assertEqual(robot.llamadas, [('modo_mesa', 'robots')])   # landed before soltar returned
        mesa.camara('detras_robot_2')
        time.sleep(0.1)
        self.assertEqual(robot.llamadas, [('modo_mesa', 'robots')])

    def test_lo_del_espectador_no_cae_en_la_mesa_del_humano(self):
        srv = self._servidor()
        robot = RobotGrabador(lento=('modo_mesa', 'robots'))
        srv.partida.robot = robot
        srv.mesa_en_espectador = True
        mesa = srv.espectador.mesa
        mesa.robot = robot
        mesa.empezar()
        mesa.camara('detras_robot_1')
        time.sleep(0.05)
        srv.mesa_para_humano()
        self._esperar(lambda: ('modo_mesa', 'humano') in robot.llamadas)
        time.sleep(0.5)
        humano = robot.llamadas.index(('modo_mesa', 'humano'))
        self.assertNotIn(('elegir_camara', 'detras_robot_1'), robot.llamadas)
        self.assertTrue(all(v not in ('robots', 'detras_robot_1') for c in robot.llamadas[humano:] for v in c[1:2]))

    def test_el_hilo_unico_mantiene_el_orden(self):
        robot = RobotGrabador()
        mesa = E.MesaRemota(robot)
        hilos = [threading.Thread(target=lambda i=i: [mesa.jugar(i % 2, (1, 'Oro')) for _ in range(50)])
                 for i in range(4)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()
        self._esperar(lambda: len(robot.llamadas) == 200)

    def test_tras_reiniciar_el_simulador_vuelve_el_3d(self):
        import servidor_truco
        srv = self._servidor(sin_robot=False)
        caido = MagicMock()
        caido.modo_mesa.side_effect = OSError('conexión perdida')
        nuevo = RobotGrabador()
        with mock.patch.object(servidor_truco, 'Robot', side_effect=[caido, nuevo]):
            srv.espectador                      # created before the table switches
            srv.mesa_para_espectador()          # connects `caido`; its first move fails
            self._esperar(lambda: srv.robot is None)
            self.assertIsNone(srv.espectador.mesa.robot)
            srv.asegurar_robot()                # what the next /espectador/estado does
        self.assertIs(srv.espectador.mesa.robot, nuevo)
        self._esperar(lambda: ('modo_mesa', 'robots') in nuevo.llamadas)


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

    def test_tras_pausa_no_sale_ningun_evento(self):
        r = E.Reproductor(ritmo=E.RitmoSimple(escala=0))
        try:
            r.empezar(('reactivo', 'reactivo'), seed=3)
            self._esperar(lambda: len(r.partida.eventos) > 3)
            r.pausar()
            with r._lock_partida:               # an event already being shown finishes
                antes = len(r.partida.eventos)
            time.sleep(0.2)
            self.assertEqual(len(r.partida.eventos), antes)
        finally:
            r.cerrar()

    def test_paso_avisa_a_los_oyentes_sin_el_lock(self):
        bloqueados = []

        def oyente(evento):
            hilo = threading.Thread(target=self.r.snapshot, daemon=True)
            hilo.start()
            hilo.join(1)
            bloqueados.append(hilo.is_alive())

        self.r.oyentes.append(oyente)
        self.r.empezar(('reactivo', 'utilidad'), seed=1)
        self.r.pausar()
        self.r.paso()
        self.assertTrue(bloqueados)
        self.assertFalse(any(bloqueados))

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

    def test_una_ia_que_no_carga(self):
        crear = E.crear_politica

        def falla(clave, *args):
            if clave == 'genetico':
                raise OSError('[WinError 126]')
            return crear(clave, *args)

        self.srv.mesa_para_humano()
        antes = self.srv.espectador.partida
        with mock.patch.object(E, 'crear_politica', falla), mock.patch.object(E, 'modelo_disponible', lambda *a: True):
            estado, d = self._post('/espectador/empezar', {'ias': ['genetico', 'reactivo']})
        self.assertEqual(estado, 409)
        self.assertEqual(d, {'error': 'No se pudo cargar esa IA', 'ia': 'genetico'})
        self.assertFalse(self.srv.mesa_en_espectador)          # the table stays with the human
        self.assertIs(self.srv.espectador.partida, antes)

    def test_partida_corta_contra_el_humano(self):
        estado, d = self._post('/truco/reiniciar', {'manos': 3})
        self.assertEqual(estado, 200)
        self.assertEqual(d['estado']['tanteador']['manos'], 3)
        self.assertEqual(self._post('/truco/reiniciar', {'manos': 11})[0], 400)
        d = self._post('/truco/reiniciar', {'meta_puntos': 15})[1]
        self.assertIsNone(d['estado']['tanteador']['manos'])
        pagina = self._get('/')[1].decode('utf-8')
        # Short match: hands counted ("Mano 2 de 5" / "Desempate"), no bar towards 30.
        self.assertIn('id="manos-texto"', pagina)
        self.assertIn("'Desempate'", pagina)


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
