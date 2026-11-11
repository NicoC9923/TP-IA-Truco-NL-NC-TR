"""Spectator mode: two AIs play a match on the symmetric engine, shown at human pace.

ADR 0064. The rules come from `symmetric_engine.play_match`, unchanged: the engine
runs in a worker thread and each policy call blocks until the driver releases it.
Everything the viewer sees comes out of the driver as a stream of events, one at a
time, so pause / step / speed only decide *when* the next event is released.

Layers
------
`PartidaEspectador`   one match, stepped one event at a time, no clock.
`Reproductor`         the clock: plays a `PartidaEspectador` at the pace given by a
                      pacing object, with pause, step, speed, rematch and new AIs.

Pacing seam (step 3 plugs in here)
----------------------------------
Every event is a JSON-ready dict::

    {"id": 12, "t": 4.31, "seat": 0 | 1 | None, "kind": KIND, "action": str | None,
     "label": "Truco", "hand": 2, ...kind-specific fields}

KIND is one of ``deal``, ``think``, ``call``, ``respond``, ``play``, ``fold``,
``tanto``, ``hand_end``, ``match_end`` (see `KINDS`). ``t`` is seconds since the
match started. ``label`` is the short Spanish text a player would say or the
table would show; the reasoning (``think`` events) is on-screen only (ADR 0066).

A pacing object needs one method, ``demora(evento) -> float``: the seconds to wait
after showing that event, at speed x1. The driver divides by the chosen speed
(x0.5 / x1 / x2). Listeners (``oyentes``) are called with each event as it is
shown, from the driver's thread: that is where voice and gestures hook in.
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import truco_reglas as rules
from symmetric_engine import play_match

KINDS = ('deal', 'think', 'call', 'respond', 'play', 'fold', 'tanto', 'hand_end', 'match_end')
ACCIONES_DE_JUEGO = ('call', 'respond', 'play', 'fold')
VELOCIDADES = (0.5, 1.0, 2.0)
ENVIDOS = ('envido', 'real_envido', 'falta_envido')
ETIQUETAS = {'envido': 'Envido', 'real_envido': 'Real envido', 'falta_envido': 'Falta envido',
             'truco': 'Truco', 'retruco': 'Quiero retruco', 'vale_cuatro': 'Quiero vale cuatro',
             'quiero': 'Quiero', 'no_quiero': 'No quiero', 'fold': 'Me voy al mazo'}
NIVEL_TRUCO = {1: 'Sin truco', 2: 'Truco', 3: 'Retruco', 4: 'Vale cuatro'}
CHAMPIONS = Path(__file__).resolve().parent.parent / 'champions'


# --------------------------------------------------------------------- models

@dataclass(frozen=True)
class Modelo:
    clave: str
    nombre: str
    tipo: str                      # 'red_refuerzo' | 'red_genetica' | 'umbral' | 'regla'


MODELOS = {m.clave: m for m in (
    Modelo('refuerzo', 'Refuerzo corrida 3 · gen 73', 'red_refuerzo'),
    Modelo('utilidad', 'Utilidad adaptativa', 'umbral'),
    Modelo('genetico', 'Genético publicado (gen 48)', 'red_genetica'),
    Modelo('reactivo', 'Reactivo', 'regla'),
)}


def modelo_disponible(clave, champions_dir=CHAMPIONS):
    champions_dir = Path(champions_dir)
    if clave == 'refuerzo':
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return (champions_dir / 'phase2/reigning_champion.pt').is_file()
    if clave == 'genetico':
        return (champions_dir / 'phase1/reigning_champion.npy').is_file()
    return clave in MODELOS


def lista_modelos(champions_dir=CHAMPIONS):
    return [{'clave': m.clave, 'nombre': m.nombre, 'disponible': modelo_disponible(m.clave, champions_dir)}
            for m in MODELOS.values()]


def nombre_accion(accion, mano=None):
    if accion.startswith('card_') and mano is not None:
        valor, palo = mano[int(accion[-1])]
        return rules.nombre_carta(valor, palo)
    return ETIQUETAS.get(accion, accion)


def _top(valores, mano, n=3):
    orden = sorted(valores.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(nombre_accion(a, mano), a, v) for a, v in orden]


class _Red:
    """Explains a network decision: its top actions with what the network gave them."""

    def explicar(self, obs, legal, accion):
        raise NotImplementedError


def _crear_refuerzo(seed, champions_dir):
    import torch
    from phase2_game_policy import GamePhase2Policy
    from phase2_torch import _q_values
    from phase1_network import ACTIONS

    class RedRefuerzo(GamePhase2Policy):
        def record_decision(self, inputs, action):
            with torch.no_grad():
                self.last_values = _q_values(self.model, torch.from_numpy(inputs)).numpy()

        def explicar(self, obs, legal, accion):
            valores = {a: float(self.last_values[ACTIONS.index(a)]) for a in legal}
            opciones = [{'accion': a, 'etiqueta': e, 'valor': round(v, 3)} for e, a, v in _top(valores, obs['hand'])]
            nota = ('Elige siempre la jugada de mayor valor.' if self.greedy
                    else 'Sortea entre las jugadas según su valor.')
            return {'tipo': 'red', 'opciones': opciones, 'nota': nota}

    return RedRefuerzo.load(Path(champions_dir) / 'phase2/reigning_champion.pt', seed=seed)


def _crear_genetico(seed, champions_dir):
    from phase1_network import NetworkPolicy

    class RedGenetica(NetworkPolicy):
        def explicar(self, obs, legal, accion):
            opciones = [{'accion': a, 'etiqueta': e, 'probabilidad': round(p, 4)}
                        for e, a, p in _top(self.last_probabilities, obs['hand'])]
            return {'tipo': 'red', 'opciones': opciones, 'nota': 'Sortea la jugada con estas probabilidades.'}

    return RedGenetica.load(Path(champions_dir) / 'phase1/reigning_champion.npy', seed=seed, explain=True)


def _explicar_agente(policy, tipo):
    """Utilidad adaptativa / Reactivo: the last step of their own reasoning trace."""
    traza = policy.agent.traza_razonamiento
    paso = traza[-1] if traza else None
    if paso is None:
        return {'tipo': tipo, 'explicacion': 'Juega la carta según su regla de cartas.'}
    if tipo == 'regla':
        return {'tipo': 'regla', 'regla': paso.get('explicacion') or paso.get('umbral', ''),
                'fase': paso.get('fase', '')}
    return {'tipo': 'umbral', 'fase': paso.get('fase', ''), 'metrica': paso.get('metrica', ''),
            'umbral': paso.get('umbral', ''), 'explicacion': paso.get('explicacion', '')}


def crear_politica(clave, seed, champions_dir=CHAMPIONS):
    """A fresh policy for `clave`, with an `explicar(obs, legal, action)` method."""
    from engine_policies import AdaptiveUtilityPolicy, ReactivePolicy
    if clave == 'refuerzo':
        return _crear_refuerzo(seed, champions_dir)
    if clave == 'genetico':
        return _crear_genetico(seed, champions_dir)
    if clave in ('utilidad', 'reactivo'):
        base = AdaptiveUtilityPolicy if clave == 'utilidad' else ReactivePolicy
        tipo = 'umbral' if clave == 'utilidad' else 'regla'

        class Agente(base):
            def __call__(self, obs, legal):
                self.agent.limpiar_traza_mano()
                return super().__call__(obs, legal)

            def explicar(self, obs, legal, accion):
                return _explicar_agente(self, tipo)

        return Agente(seed)
    raise ValueError(f'Modelo desconocido: {clave!r}')


# ------------------------------------------------------------- envido rule

def sin_envido_tras_primera_carta(obs, legal):
    """ADR 0068: who already played their first-baza card may not call envido,
    not even as "el envido va primero" against a truco. Raising inside an envido
    exchange already under way stays allowed: the rule limits starting it.
    Returns the legal set without those calls (the training engine allows them)."""
    pending = obs['pending']
    if pending is not None and pending['kind'] == 'envido':
        return tuple(legal)
    jugo = not obs['tricks'] and any(seat == obs['player'] for seat, _ in obs['table'])
    if not jugo:
        return tuple(legal)
    return tuple(a for a in legal if a not in ENVIDOS)


# -------------------------------------------------------------- match length

@dataclass(frozen=True)
class Largo:
    """15 or 30 points, or a short match of 1 to 10 hands (ADR 0069)."""
    tipo: str = 'puntos'           # 'puntos' | 'manos'
    meta: int = 15
    manos: int | None = None

    # A short match still needs a points target for falta envido and for the
    # networks' inputs; 30 is the one they know, and reaching it also ends it.
    META_CORTA = 30

    @classmethod
    def de_dict(cls, datos):
        if not isinstance(datos, dict):
            raise ValueError('largo inválido')
        if datos.get('tipo') == 'manos':
            manos = datos.get('manos')
            if type(manos) is not int or not 1 <= manos <= 10:
                raise ValueError('manos debe ser de 1 a 10')
            return cls('manos', cls.META_CORTA, manos)
        meta = datos.get('meta', 15)
        if type(meta) is not int or meta not in (15, 30):
            raise ValueError('meta debe ser 15 o 30')
        return cls('puntos', meta, None)

    def como_dict(self):
        return {'tipo': self.tipo, 'meta': self.meta, 'manos': self.manos}

    def texto(self):
        return f'{self.manos} {"mano" if self.manos == 1 else "manos"}' if self.tipo == 'manos' else f'A {self.meta}'


# ---------------------------------------------------------------- the match

class _Cancelado(Exception):
    pass


class _Jugador:
    """Wraps a policy for the engine thread: masks envido, explains, waits for release."""

    def __init__(self, seat, policy, partida):
        self.seat, self.policy, self.partida = seat, policy, partida

    def __call__(self, obs, legal):
        permitidas = sin_envido_tras_primera_carta(obs, legal)
        accion = self.policy(obs, permitidas)
        try:
            razon = self.policy.explicar(obs, permitidas, accion)
        except Exception:                                   # noqa: BLE001  (never stop the match)
            razon = {'tipo': 'nada'}
        if len(permitidas) < len(legal):
            razon['envido_bloqueado'] = 'No puede cantar envido: ya jugó su primera carta.'
        self.partida._salida.put(('decision', self.seat, obs, permitidas, accion, razon))
        if self.partida._entrada.get() == 'cancelar':
            raise _Cancelado()
        return accion

    def observe_hand(self, decisions, opponent):
        observer = getattr(self.policy, 'observe_hand', None)
        if observer is not None:
            observer(decisions, opponent)


class PartidaEspectador:
    """One AI-vs-AI match that advances one event at a time (no clock)."""

    def __init__(self, ias, largo=Largo(), seed=None, mesa=None, champions_dir=CHAMPIONS):
        if len(ias) != 2 or any(clave not in MODELOS for clave in ias):
            raise ValueError('Hacen falta dos modelos válidos')
        self.ias = tuple(ias)
        self.largo = largo
        self.seed = random.randrange(2**31) if seed is None else seed
        self.mesa = mesa                         # optional 3D table (MesaRemota)
        rng = random.Random(self.seed)
        self.politicas = [crear_politica(clave, rng.randrange(2**31), champions_dir) for clave in self.ias]
        self._salida: queue.Queue = queue.Queue()
        self._entrada: queue.Queue = queue.Queue()
        self._cola: deque = deque()
        self._pendiente = None
        self._fin_motor = False
        self._t0 = time.monotonic()
        self._mano_motor = 0
        self._manos_motor = ()
        self._puntos_al_repartir = (0, 0)
        self._siguiente_id = 1
        self.eventos: list[dict] = []
        self.terminada = False
        self.error = None
        # What the viewer sees, updated event by event.
        self.puntos = [0, 0]
        self.mano_numero = 0
        self.mano = 0
        self.cartas = [[], []]
        self.jugadas = [set(), set()]
        self.mesa_cartas: list = []             # [(seat, (valor, palo))]
        self.bazas: list = []
        self.truco = 1
        self.pendiente = None
        self.tantos = [None, None]
        self.turno = None
        self.razonamiento = [None, None]
        self.ultimo_evento = None
        self.ganador = None
        self._hilo = threading.Thread(target=self._correr, daemon=True, name='espectador-motor')
        self._hilo.start()

    # ---- engine thread ----
    def _correr(self):
        jugadores = [_Jugador(seat, p, self) for seat, p in enumerate(self.politicas)]
        try:
            result = play_match(jugadores[0], jugadores[1], seed=self.seed, target=self.largo.meta,
                                max_hands=self.largo.manos, on_event=self._salida.put)
            self._salida.put(('fin', result))
        except _Cancelado:
            pass
        except Exception as error:                           # noqa: BLE001
            self._salida.put(('error', error))

    def cancelar(self):
        if not self._fin_motor:
            self._fin_motor = True
            self._entrada.put('cancelar')

    # ---- events ----
    def _evento(self, kind, seat=None, accion=None, label='', **extra):
        assert kind in KINDS
        return {'kind': kind, 'seat': seat, 'action': accion, 'label': label, 'hand': self._mano_motor, **extra}

    def _leer_motor(self):
        """Turns engine output into queued events, up to the next decision or the end."""
        while True:
            item = self._salida.get()
            if isinstance(item, dict):                  # on_event from the engine
                if item['kind'] == 'deal':
                    self._mano_motor = item['hand_index']
                    self._puntos_al_repartir = item['score']
                    manos = [[list(c) for c in h] for h in item['hands']]
                    self._manos_motor = item['hands']
                    numero = item['hand_index'] + 1
                    desempate = self.largo.manos is not None and numero > self.largo.manos
                    self._cola.append(self._evento(
                        'deal', label=f'Mano {numero}' + (' (desempate)' if desempate else ''),
                        mano=item['mano'], cartas=manos, score=list(item['score']), desempate=desempate))
                else:
                    antes = self._puntos_al_repartir
                    ganados = [item['score'][i] - antes[i] for i in (0, 1)]
                    self._cola.append(self._evento('hand_end', label='Fin de la mano', score=list(item['score']),
                                                   ganados=ganados))
                continue
            tipo = item[0]
            if tipo == 'decision':
                _, seat, obs, legal, accion, razon = item
                self._pendiente = (seat, obs, legal, accion)
                self._cola.append(self._evento('think', seat, label='Pensando…', razonamiento=razon,
                                               legales=len(legal), _obs=obs))
                return
            self._fin_motor = True
            if tipo == 'fin':
                result = item[1]
                self._cola.append(self._evento('match_end', result.winner, label='Fin de la partida',
                                               score=list(result.points), ganador=result.winner))
            else:
                self.error = repr(item[1])
                self._cola.append(self._evento('match_end', None, label='La partida se cortó por un error',
                                               score=list(self.puntos), ganador=None, error=self.error))
            return

    def _revelar_accion(self):
        seat, obs, _legal, accion = self._pendiente
        self._pendiente = None
        pending = obs['pending']
        if accion.startswith('card_'):
            carta = obs['hand'][int(accion[-1])]
            self._cola.append(self._evento('play', seat, accion, rules.nombre_carta(*carta),
                                           carta={'valor': carta[0], 'palo': carta[1]}))
        elif accion == 'fold':
            self._cola.append(self._evento('fold', seat, accion, ETIQUETAS['fold']))
        else:
            kind = 'respond' if pending else 'call'
            label = ETIQUETAS[accion]
            if accion == 'envido' and pending and pending.get('call') == 'envido':
                label = 'Envido'
            self._cola.append(self._evento(kind, seat, accion, label))
            if accion == 'quiero' and pending and pending['kind'] == 'envido':
                hands = self._manos_motor
                values = [rules.calcular_envido([{'valor': v, 'palo': p} for v, p in h]) for h in hands]
                mano = obs['mano']
                pie = 1 - mano
                self._cola.append(self._evento('tanto', mano, label=str(values[mano]), tanto=values[mano]))
                if values[pie] > values[mano]:
                    self._cola.append(self._evento('tanto', pie, label=str(values[pie]), tanto=values[pie]))
                else:
                    self._cola.append(self._evento('tanto', pie, label='Son buenas', tanto='son buenas'))
        self._entrada.put('seguir')

    def iniciar(self):
        """Prepares the 3D table; the first event comes from `siguiente_evento`."""
        self._mano_motor = 0
        if self.mesa is not None:
            self.mesa.empezar()

    def siguiente_evento(self):
        """Shows the next event and returns it, or None once the match is over."""
        if not self._cola:
            if self.terminada:
                return None
            if self._pendiente is not None:
                self._revelar_accion()
            if not self._fin_motor or not self._salida.empty():
                self._leer_motor()
            if not self._cola:
                return None
        evento = self._cola.popleft()
        obs = evento.pop('_obs', None)
        evento['id'] = self._siguiente_id
        evento['t'] = round(time.monotonic() - self._t0, 3)
        self._siguiente_id += 1
        self._aplicar(evento, obs)
        self.eventos.append(evento)
        if len(self.eventos) > 400:
            del self.eventos[:100]
        self.ultimo_evento = evento
        return evento

    def avanzar_jugada(self):
        """Step: reveals the pending action (if an AI is thinking) and shows events
        up to and including the next AI thinking, or the end."""
        pensando = self.ultimo_evento is not None and self.ultimo_evento['kind'] == 'think'
        mostrados = []
        while True:
            evento = self.siguiente_evento()
            if evento is None:
                return mostrados
            mostrados.append(evento)
            if evento['kind'] == 'match_end':
                return mostrados
            if evento['kind'] == 'think' and (not pensando or any(e['kind'] in ACCIONES_DE_JUEGO for e in mostrados)):
                return mostrados

    def jugar_hasta_el_final(self, limite=100000):
        for _ in range(limite):
            if self.siguiente_evento() is None:
                return
        raise RuntimeError('La partida no terminó')

    # ---- visible state ----
    def _aplicar(self, ev, obs):
        kind, seat = ev['kind'], ev['seat']
        if kind == 'deal':
            self.mano_numero = ev['hand'] + 1
            self.mano = ev['mano']
            self.cartas = [[tuple(c) for c in h] for h in ev['cartas']]
            self.jugadas = [set(), set()]
            self.mesa_cartas, self.bazas = [], []
            self.truco, self.pendiente = 1, None
            self.tantos = [None, None]
            self.razonamiento = [None, None]
            self.puntos = list(ev['score'])
            if self.mesa is not None:
                self.mesa.repartir(self.cartas)
        elif kind == 'think':
            self.turno = seat
            self.razonamiento[seat] = {'decision': None, **ev['razonamiento']}
            self.puntos = list(obs['score'])
            self.truco = obs['truco']
            self.pendiente = obs['pending']
            self.mesa_cartas = list(obs['table'])
            self.bazas = list(obs['tricks'])
        elif kind in ACCIONES_DE_JUEGO:
            if self.razonamiento[seat] is not None:
                self.razonamiento[seat]['decision'] = ev['label']
            self.turno = None
            if kind == 'play':
                carta = (ev['carta']['valor'], ev['carta']['palo'])
                self.jugadas[seat].add(carta)
                self.mesa_cartas.append((seat, carta))
                if len(self.mesa_cartas) % 2 == 0:
                    (a, ca), (b, cb) = self.mesa_cartas[-2:]
                    pa, pb = rules.valor_carta_truco(*ca), rules.valor_carta_truco(*cb)
                    self.bazas.append(None if pa == pb else (a if pa > pb else b))
                if self.mesa is not None:
                    self.mesa.jugar(seat, carta)
            elif ev['action'] in ('truco', 'retruco', 'vale_cuatro'):
                level = {'truco': 2, 'retruco': 3, 'vale_cuatro': 4}[ev['action']]
                self.pendiente = {'kind': 'truco', 'caller': seat, 'level': level}
            elif ev['action'] in ENVIDOS:
                self.pendiente = {'kind': 'envido', 'caller': seat, 'call': ev['action']}
            else:
                if ev['action'] == 'quiero' and self.pendiente and self.pendiente['kind'] == 'truco':
                    self.truco = self.pendiente['level']
                self.pendiente = None
        elif kind == 'tanto':
            self.tantos[seat] = ev['tanto']
        elif kind == 'hand_end':
            self.puntos = list(ev['score'])
            self.turno = None
        elif kind == 'match_end':
            self.puntos = list(ev['score'])
            self.terminada = True
            self.ganador = ev['ganador']
            self.turno = None

    def estado(self):
        def asiento(i):
            return {'ia': self.ias[i], 'nombre': MODELOS[self.ias[i]].nombre, 'puntos': self.puntos[i],
                    'es_mano': self.mano == i, 'pensando': self.turno == i,
                    'cartas': [{'valor': v, 'palo': p, 'nombre': rules.nombre_carta(v, p),
                                'jugada': (v, p) in self.jugadas[i]} for v, p in self.cartas[i]],
                    'tanto': self.tantos[i], 'razonamiento': self.razonamiento[i]}
        pend = self.pendiente
        return {
            'asientos': [asiento(0), asiento(1)],
            'largo': self.largo.como_dict(), 'largo_texto': self.largo.texto(),
            'mano_numero': self.mano_numero,
            'desempate': self.largo.manos is not None and self.mano_numero > self.largo.manos,
            'mesa': [{'asiento': s, 'valor': c[0], 'palo': c[1], 'nombre': rules.nombre_carta(*c)}
                     for s, c in self.mesa_cartas],
            'bazas': self.bazas,
            'truco': NIVEL_TRUCO.get(self.truco, 'Truco'), 'puntos_truco': self.truco,
            'pendiente': None if pend is None else {
                'tipo': pend['kind'], 'quien': pend['caller'],
                'canto': (NIVEL_TRUCO.get(pend.get('level'), 'Truco') if pend['kind'] == 'truco'
                          else ETIQUETAS.get(str(pend.get('call', 'envido')).replace('envido_envido', 'envido'), 'Envido'))},
            'terminada': self.terminada, 'ganador': self.ganador, 'error': self.error,
            'semilla': self.seed,
        }


# ------------------------------------------------------------------ pacing

class RitmoSimple:
    """Default pacing until step 3's `ritmo` plugs in: fixed delays at speed x1.

    `escala` multiplies every delay (0 = no waiting at all, for tests)."""

    DEMORAS = {'deal': 1.4, 'think': 1.6, 'call': 1.3, 'respond': 1.2, 'play': 1.9,
               'fold': 1.5, 'tanto': 1.0, 'hand_end': 2.4, 'match_end': 0.0}

    def __init__(self, escala=1.0):
        self.escala = escala

    def demora(self, evento):
        base = self.DEMORAS.get(evento['kind'], 1.0)
        if evento['kind'] == 'think' and evento.get('legales', 2) <= 1:
            base = 0.6                       # nothing to decide: short pause
        return base * self.escala


# ------------------------------------------------------------- 3D table

class MesaRemota:
    """Feeds the MuJoCo table like PartidaTruco does, in order, never blocking.

    `robot` is a `sim.robot.Robot` (or None). Seat 0 of the engine sits in
    3D seat 1, seat 1 in seat 2."""

    def __init__(self, robot=None):
        self.robot = robot
        self._cola: queue.Queue = queue.Queue()
        self._hilo = None

    def _enviar(self, metodo, *args):
        if self.robot is None or not hasattr(self.robot, metodo):
            return
        self._cola.put((metodo, args))
        if self._hilo is None or not self._hilo.is_alive():
            self._hilo = threading.Thread(target=self._trabajar, daemon=True, name='espectador-mesa')
            self._hilo.start()

    def _trabajar(self):
        while True:
            metodo, args = self._cola.get()
            robot = self.robot
            if robot is None:
                continue
            try:
                getattr(robot, metodo)(*args)
            except Exception as error:                       # noqa: BLE001
                if isinstance(error, OSError) or type(error).__name__ in ('ErrorTransporte', 'NoHaySimulador'):
                    self.robot = None

    def empezar(self):
        self._enviar('modo_mesa', 'robots')
        self._enviar('sentarse')
        self._enviar('elegir_camara', 'mesa')

    def repartir(self, cartas):
        manos = [[{'valor': v, 'palo': p} for v, p in mano] for mano in cartas]
        self._enviar('repartir_cartas', manos[0], manos[1])

    def jugar(self, seat, carta):
        self._enviar('tirar_carta', {'valor': carta[0], 'palo': carta[1]}, seat + 1)

    def camara(self, nombre):
        self._enviar('elegir_camara', nombre)


# ---------------------------------------------------------------- the clock

class Reproductor:
    """Plays spectator matches at human pace. Thread-safe; one per server."""

    def __init__(self, ritmo=None, oyentes=(), champions_dir=CHAMPIONS, robot=None):
        self.ritmo = ritmo or RitmoSimple()
        self.oyentes = list(oyentes)
        self.champions_dir = champions_dir
        self.mesa = MesaRemota(robot)
        self.partida: PartidaEspectador | None = None
        self.estado = 'sin_partida'          # sin_partida | jugando | pausa | terminada
        self.velocidad = 1.0
        self._cond = threading.Condition()
        self._lock_partida = threading.RLock()
        self._generacion = 0
        self._hilo = threading.Thread(target=self._bucle, daemon=True, name='espectador-reloj')
        self._hilo.start()

    # ---- controls ----
    def empezar(self, ias, largo=Largo(), seed=None):
        partida = PartidaEspectador(ias, largo, seed, self.mesa, self.champions_dir)
        with self._lock_partida:
            if self.partida is not None:
                self.partida.cancelar()
            self.partida = partida
            partida.iniciar()
        with self._cond:
            self._generacion += 1
            self.estado = 'jugando'
            self._cond.notify_all()
        return self.snapshot()

    def revancha(self):
        if self.partida is None:
            raise ValueError('No hay partida')
        return self.empezar(self.partida.ias, self.partida.largo)

    def detener(self):
        """Back to choosing the AIs."""
        with self._lock_partida:
            if self.partida is not None:
                self.partida.cancelar()
            self.partida = None
        with self._cond:
            self._generacion += 1
            self.estado = 'sin_partida'
            self._cond.notify_all()
        return self.snapshot()

    def pausar(self):
        with self._cond:
            if self.estado == 'jugando':
                self.estado = 'pausa'
            self._cond.notify_all()
        return self.snapshot()

    def seguir(self):
        with self._cond:
            if self.estado == 'pausa':
                self.estado = 'jugando'
            self._cond.notify_all()
        return self.snapshot()

    def paso(self):
        """One action: pauses if playing, then shows up to the next AI thinking."""
        with self._cond:
            if self.estado == 'jugando':
                self.estado = 'pausa'
        with self._lock_partida:
            if self.partida is None:
                return self.snapshot()
            for evento in self.partida.avanzar_jugada():
                self._avisar(evento)
            if self.partida.terminada:
                with self._cond:
                    self.estado = 'terminada'
        return self.snapshot()

    def poner_velocidad(self, velocidad):
        if velocidad not in VELOCIDADES:
            raise ValueError('velocidad inválida')
        with self._cond:
            self.velocidad = float(velocidad)
            self._cond.notify_all()
        return self.snapshot()

    def resincronizar_mesa(self, robot):
        """The simulator (re)connected: rebuild the current hand on the 3D table."""
        with self._lock_partida:
            self.mesa.robot = robot
            self.mesa.empezar()
            partida = self.partida
            if partida is not None and partida.cartas[0]:
                self.mesa.repartir(partida.cartas)
                for seat, carta in partida.mesa_cartas:
                    self.mesa.jugar(seat, carta)

    def cerrar(self):
        self.detener()
        with self._cond:
            self._generacion = -1
            self._cond.notify_all()

    # ---- the loop ----
    def _avisar(self, evento):
        for oyente in self.oyentes:
            try:
                oyente(evento)
            except Exception:                                # noqa: BLE001
                pass

    def _bucle(self):
        while True:
            with self._cond:
                while self.estado != 'jugando' and self._generacion >= 0:
                    self._cond.wait()
                if self._generacion < 0:
                    return
                generacion = self._generacion
            with self._lock_partida:
                partida = self.partida
                evento = partida.siguiente_evento() if partida is not None else None
            if evento is None:
                with self._cond:
                    if self._generacion == generacion and self.estado == 'jugando':
                        self.estado = 'terminada'
                continue
            self._avisar(evento)
            if evento['kind'] == 'match_end':
                with self._cond:
                    if self._generacion == generacion:
                        self.estado = 'terminada'
                continue
            self._esperar(self.ritmo.demora(evento), generacion)

    def _esperar(self, segundos, generacion):
        """Waits `segundos` of x1 time: paused time does not count, speed scales it."""
        restante = segundos
        while restante > 1e-6:
            with self._cond:
                if self._generacion != generacion:
                    return
                if self.estado != 'jugando':
                    self._cond.wait()
                    continue
                velocidad = self.velocidad
                inicio = time.monotonic()
                self._cond.wait(timeout=restante / velocidad)
                restante -= (time.monotonic() - inicio) * velocidad

    # ---- state for the page ----
    def snapshot(self, desde=0):
        with self._lock_partida:
            partida = self.partida
            datos = {'estado': self.estado, 'velocidad': self.velocidad,
                     'robot_conectado': self.mesa.robot is not None, 'partida': None, 'eventos': []}
            if partida is not None:
                datos['partida'] = partida.estado()
                datos['eventos'] = [e for e in partida.eventos if e['id'] > desde][-60:]
                datos['ultimo_evento'] = partida.ultimo_evento
            return json.loads(json.dumps(datos, default=list))
