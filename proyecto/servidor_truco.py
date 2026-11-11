"""Servidor local para jugar al Truco contra Ottoman (Unitree G1) y controlarlo por voz.

Integra:
1. Juego completo de Truco Argentino (/truco/estado, /truco/jugar, /truco/envido,
   /truco/truco, /truco/responder, /truco/mazo, /truco/voz, /truco/nuevo,
   /truco/reiniciar, /truco/detectar).
2. Modo espectador: dos IAs juegan entre sí (/espectador, /espectador/*; ADR 0064).
3. Control libre por voz bilingüe (/comando).
4. Conexión automática (y reconexión en caliente) con el simulador MuJoCo.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from robot import Robot, aviso_sin_mujoco
from control_voz import AgenteRobot
from partida_truco import PartidaTruco
from truco_ia import AgenteTrucoIA
from truco_vision import DetectorCartas, MENSAJE_SIN_OPENCV

PUERTO_DEFAULT = 8080
DIR_PROYECTO = Path(__file__).resolve().parent
PAGINA = DIR_PROYECTO / "voz.html"
PAGINA_ESPECTADOR = DIR_PROYECTO / "espectador.html"
DIR_CARTAS = DIR_PROYECTO / "cartas"
DIR_FUENTES = DIR_PROYECTO / "fuentes"

class ServidorTruco(ThreadingHTTPServer):
    def __init__(self, direccion, partida, sin_robot, evolution_root=None):
        super().__init__(direccion, Handler)
        self.evolution_root = Path(evolution_root) if evolution_root is not None else DIR_PROYECTO.parent
        self.lock = threading.RLock()
        self.partida = partida
        self.robot = partida.robot
        self.sin_robot = sin_robot
        self.agente = AgenteRobot(self.robot)
        self.detector = DetectorCartas()
        self.error_conexion = None
        partida.al_perder_robot = self.perder_robot
        self._espectador = None
        # Which match the 3D table shows: the human one, or a spectator match.
        self.mesa_en_espectador = False

    @property
    def espectador(self):
        """The spectator-mode driver, created on first use (it loads no model until a match)."""
        with self.lock:
            if self._espectador is None:
                from espectador import Reproductor
                self._espectador = Reproductor(champions_dir=self.evolution_root / "champions",
                                               al_perder_robot=self.perder_robot)
            return self._espectador

    def mesa_para_espectador(self):
        """A spectator match starts: the 3D table follows it."""
        with self.lock:
            self.mesa_en_espectador = True
            ya_conectado = self.robot
            robot = self.asegurar_robot()
            # A fresh connection was already handed over (and resynced) by asegurar_robot;
            # setting it again could undo a loss the table worker just reported.
            if robot is None or robot is ya_conectado:
                self.espectador.mesa.robot = robot

    def mesa_para_humano(self):
        """The human plays again: pause the spectator match and give the table back."""
        with self.lock:
            if not self.mesa_en_espectador:
                return
            self.mesa_en_espectador = False
            if self._espectador is not None:
                self._espectador.pausar()
                # Drop the spectator's queued moves and wait for one in flight, so a
                # late modo_mesa('robots') or camera change never lands on the human's table.
                self._espectador.mesa.soltar()
            self.partida.sincronizar_mesa_3d()

    def perder_robot(self):
        """The simulator connection died: drop it so the next request reconnects."""
        with self.lock:
            self.robot = None
            self.partida.robot = None
            self.agente.robot = None
            self.agente.ejecutor.robot = None
            self.error_conexion = None
            if self._espectador is not None:
                self._espectador.mesa.robot = None

    def asegurar_robot(self):
        with self.lock:
            if self.sin_robot or self.robot is not None:
                return self.robot
            try:
                robot = Robot()
                robot.conectar()
                self.error_conexion = None
                self.robot = robot
                self.agente.robot = robot
                self.agente.ejecutor.robot = robot
                self.partida.robot = robot
                if self.mesa_en_espectador and self._espectador is not None:
                    self._espectador.resincronizar_mesa(robot)
                else:
                    self.partida.sincronizar_mesa_3d()
                print("  [OK] Conectado dinámicamente al simulador del Robot.")
                return robot
            except Exception as error:
                tipo = type(error).__name__
                if tipo != self.error_conexion:
                    print(f"  [AVISO] Error al conectar el simulador: {tipo}")
                    self.error_conexion = tipo
                return None


def _leer_jsonl(path):
    records = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
    except (OSError, UnicodeDecodeError):
        pass
    return records


EXAM_OPPONENTS = ['adaptive_utility', 'reactive', 'phase1', 'held_out_1', 'held_out_2']


def _retired_run1(root):
    """Directory with the retired run 1 generation 28 weights, or the shipped folder before the move."""
    retired = Path(root) / 'champions/phase2/retired/run1-g28'
    return retired if (retired / 'reigning_champion.pt').is_file() else Path(root) / 'champions/phase2'


def _red_publicada(root):
    """The shipped phase 2 network and its recorded final exam; numbers are read, never recomputed."""
    root = Path(root)
    try:
        sidecar = json.loads((root / 'champions/phase2/reigning_champion.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(sidecar, dict):
        return None
    shipped = {k: sidecar.get(k) for k in ('run', 'generation', 'selection', 'temperature', 'profile_prior')}
    shipped['selection'] = shipped['selection'] or 'sampled'
    if shipped['run'] == 'run3':
        _agregar_examen_corrida3(root, shipped)
        return shipped
    try:
        exam = json.loads((root / 'experiments/phase2-final-exam/part-e/results.json').read_text(encoding='utf-8'))
        part_t = json.loads((root / 'experiments/phase2-final-exam/part-t/results.json').read_text(encoding='utf-8'))
        rates = exam['analysis']['win_rates']
        candidate = exam['analysis']['decision']
        shipped['exam'] = {'opponents': EXAM_OPPONENTS,
                           'shipped': [r['mean'] for r in rates[candidate]],
                           'retired': [r['mean'] for r in rates['B']],
                           'temperature_mean': {k: part_t['analysis']['m'][k] for k in ('0.05', 'greedy')}}
    except (OSError, ValueError, UnicodeDecodeError, KeyError, TypeError):
        pass
    return shipped


def _agregar_examen_corrida3(root, shipped):
    """Run 3 generation 73 (ADR 0063): its exam (C1 against the retired run 2 baseline B), the direct duel
    from the all-vs-all, and the one guardrail it failed. Numbers are read from the recorded results."""
    shipped['retired'] = {'run': 'run2', 'generation': 38}
    try:
        analysis = json.loads((root / 'experiments/phase2-run3-exam/part-e/results.json').read_text(encoding='utf-8'))['analysis']
        rates = analysis['win_rates']
        shipped['exam'] = {'opponents': EXAM_OPPONENTS,
                           'shipped': [r['mean'] for r in rates['C1']],
                           'retired': [r['mean'] for r in rates['B']]}
        failed = [g for g in analysis['candidates']['C1']['guardrails'] if not g['pass']]
        if failed:
            g = failed[0]
            shipped['failed_guardrail'] = {'opponent': EXAM_OPPONENTS[g['k']], 'gain': g['gain'],
                                           'lower': g['lower'], 'upper': g['upper']}
    except (OSError, ValueError, UnicodeDecodeError, KeyError, TypeError, IndexError):
        pass
    try:
        pairs = json.loads((root / 'experiments/all-vs-all/output/results.json').read_text(encoding='utf-8'))['analysis']['pairs']
        duel = next(p for p in pairs if p['row'] == 'phase2_run3_g73' and p['col'] == 'phase2_run2_g38')
        shipped['duel'] = {'opponent_generation': 38, 'win_rate': duel['mean'], 'lower': duel['lower'],
                           'upper': duel['upper'], 'pairs': duel['n']}
    except (OSError, ValueError, UnicodeDecodeError, KeyError, TypeError, StopIteration):
        pass


def _comparacion_publicada(root):
    """Only the shipped run's recorded measurements; never replay a match."""
    root = Path(root)
    p1 = root / 'champions/phase1'
    p2 = root / 'champions/phase2'
    payload = {'phase1_reference': _leer_jsonl(p1 / 'reference_evaluation.jsonl'),
               'phase1_champions': [{'generation': r.get('generation'), 'champion': r.get('reigning_champion')}
                                    for r in _leer_jsonl(p1 / 'generations.jsonl')],
               'phase2_champions': [{'generation': r.get('generation'), 'champion': r.get('champion')}
                                    for r in _leer_jsonl(p2 / 'generations.jsonl')],
               'phase2_reference': [], 'probe': None, 'shipped': _red_publicada(root)}
    try:
        archive = json.loads((root / 'experiments/phase2-run1/reference-eval/results.json').read_text(encoding='utf-8'))
        for generation, opponents in archive['summaries'].items():
            if not generation.startswith('gen') or not generation[3:].isdigit():
                continue
            for opponent, values in opponents.items():
                if opponent not in ('reactive', 'adaptive_utility'):
                    continue
                for adaptation in ('on', 'off'):
                    value = values[adaptation]
                    payload['phase2_reference'].append({'generation': int(generation[3:]), 'reference_opponent': opponent,
                        'adaptation': adaptation, 'win_rate': value['mean'], 'lower': value['lower'], 'upper': value['upper']})
    except (OSError, ValueError, UnicodeDecodeError, KeyError, TypeError):
        pass
    try:
        probe = json.loads((p2 / 'card-probe-report.json').read_text(encoding='utf-8'))
        if probe.get('champion_unchanged') and probe.get('champion_sha256'):
            import hashlib
            if hashlib.sha256((_retired_run1(root) / 'reigning_champion.pt').read_bytes()).hexdigest() == probe['champion_sha256']:
                payload['probe'] = {k: probe.get(k) for k in ('status', 'counts', 'test_scored', 'predictors', 'examples')}
    except (OSError, ValueError, UnicodeDecodeError, TypeError):
        pass
    return payload


class Handler(BaseHTTPRequestHandler):
    def log_message(self, formato, *args):
        pass

    def do_GET(self):
        partida_truco = self.server.partida
        ruta_limpia = self.path.split("?", 1)[0]
        if ruta_limpia in ("/evolucion", "/evolucion.html"):
            cuerpo = (DIR_PROYECTO / "evolucion.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)
        elif ruta_limpia == "/evolucion/comparison":
            self._responder(200, _comparacion_publicada(self.server.evolution_root))
        elif ruta_limpia in ("/evolucion/phase1/generations", "/evolucion/phase1/reference"):
            filename = "generations.jsonl" if ruta_limpia.endswith("generations") else "reference_evaluation.jsonl"
            self._responder(200, _leer_jsonl(self._origen_evolucion("phase1") / filename))
        elif ruta_limpia == "/evolucion/phase1/source":
            origen = self._origen_evolucion("phase1")
            vivo = origen.parent.name == "training_output"
            self._responder(200, {"source": "live" if vivo else ("committed" if (origen / "generations.jsonl").is_file() else "empty")})
        elif ruta_limpia == "/evolucion/phase2/runs":
            root = self.server.evolution_root
            if (root / "training_output/phase2-run3").is_dir():
                default = "run3"
            elif (root / "training_output/phase2-run2").is_dir() or (root / "champions/phase2-run2").is_dir():
                default = "run2"
            else:
                default = "run1"
            self._responder(200, {"default": default})
        elif ruta_limpia.startswith("/evolucion/phase2/") and "run" in parse_qs(urlsplit(self.path).query, keep_blank_values=True):
            run = parse_qs(urlsplit(self.path).query, keep_blank_values=True)["run"]
            if len(run) != 1 or run[0] not in ("run1", "run2", "run3"):
                self._responder(400, {"error": "invalid run"})
                return
            root = self.server.evolution_root
            if run[0] == "run1":
                directory = root / "champions/phase2"
            else:
                directory = self._origen_run(root, run[0])
            endpoint = ruta_limpia.rsplit("/", 1)[-1]
            if endpoint in ("generations", "reference"):
                name = "generations.jsonl" if endpoint == "generations" else "reference_evaluation.jsonl"
                rows = _leer_jsonl(directory / name)
                if endpoint == "reference" and run[0] == "run1" and not rows:
                    try:
                        archive = json.loads((root / "experiments/phase2-run1/reference-eval/results.json").read_text(encoding="utf-8"))
                        summaries = archive["summaries"]["gen28"]
                        rows = [{"generation": 28, "reference_opponent": opponent, "win_rate": value["on"]["mean"]}
                                for opponent, value in summaries.items() if opponent != "equal_weight"]
                    except (OSError, ValueError, UnicodeDecodeError, KeyError, TypeError):
                        pass
                self._responder(200, rows)
            elif endpoint in ("progress", "acceptance"):
                path = directory / "progress.json" if endpoint == "progress" else root / "experiments/phase2-run2/acceptance/results.json"
                payload = {"status": "empty"} if endpoint == "progress" else {"status": "pending"}
                if endpoint != "acceptance" or run[0] == "run2":  # run 3 has no final exam yet: pending
                    try:
                        value = json.loads(path.read_text(encoding="utf-8"))
                        if isinstance(value, dict):
                            payload = value
                    except (OSError, ValueError, UnicodeDecodeError):
                        pass
                self._responder(200, payload)
            elif endpoint == "example-hand":
                # Only the published run 1 model is stable; never load a live checkpoint.
                if run[0] == "run1":
                    from phase2_example_hand import example_hand_payload
                    self._responder(200, example_hand_payload(root, directory=_retired_run1(root)))
                else:
                    self._responder(200, {"available": False, "reason": "awaiting_stable_example"})
            else:
                self._responder(404, {"error": "unknown endpoint"})
        elif ruta_limpia in ("/evolucion/phase2/generations", "/evolucion/phase2/reference"):
            filename = "generations.jsonl" if ruta_limpia.endswith("generations") else "reference_evaluation.jsonl"
            self._responder(200, _leer_jsonl(self._origen_evolucion("phase2") / filename))
        elif ruta_limpia == "/evolucion/phase2/progress":
            source = self._origen_evolucion("phase2") / "progress.json"
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {"status": "empty"}
            self._responder(200, payload)
        elif ruta_limpia == "/evolucion/phase2/example-hand":
            from phase2_example_hand import example_hand_payload
            self._responder(200, example_hand_payload(self.server.evolution_root))
        elif ruta_limpia in ("/espectador", "/espectador.html"):
            cuerpo = PAGINA_ESPECTADOR.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)
        elif ruta_limpia == "/espectador/modelos":
            from espectador import lista_modelos
            self._responder(200, {"modelos": lista_modelos(self.server.evolution_root / "champions")})
        elif ruta_limpia == "/espectador/estado":
            desde = parse_qs(urlsplit(self.path).query).get("desde", ["0"])[0]
            if self.server.mesa_en_espectador:
                with self.server.lock:
                    self.server.asegurar_robot()
            self._responder(200, self.server.espectador.snapshot(int(desde) if desde.isdecimal() else 0))
        elif ruta_limpia in ("/", "/voz.html"):
            cuerpo = PAGINA.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)
        elif ruta_limpia.startswith("/cartas/"):
            nombre = Path(ruta_limpia).name
            archivo = DIR_CARTAS / nombre
            if archivo.is_file() and archivo.suffix.lower() in (".png", ".jpg", ".jpeg"):
                cuerpo = archivo.read_bytes()
                mime = "image/png" if archivo.suffix.lower() == ".png" else "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)
            else:
                self.send_response(404)
                self.end_headers()
        elif ruta_limpia.startswith("/fuentes/"):
            archivo = DIR_FUENTES / Path(ruta_limpia).name
            if archivo.is_file() and archivo.suffix.lower() == ".woff2":
                cuerpo = archivo.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)
            else:
                self.send_response(404)
                self.end_headers()
        elif ruta_limpia == "/truco/estado":
            with self.server.lock:
                self.server.asegurar_robot()
                est = copy.deepcopy(partida_truco.estado_publico())
                est["robot_conectado"] = self.server.robot is not None
            self._responder(200, est)
        elif ruta_limpia == "/truco/historial":
            with self.server.lock:
                resumen = copy.deepcopy(partida_truco.estado_publico()["aprendizaje_ia"])
            self._responder(200, resumen)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        longitud = self.headers.get("Content-Length")
        if longitud is None or not longitud.isascii() or not longitud.isdecimal():
            self._responder(400, {"error": "Content-Length inválido"})
            return
        if len(longitud.lstrip("0")) > 7:
            self._responder(413, {"error": "Cuerpo demasiado grande"})
            return
        largo = int(longitud.lstrip("0") or "0")
        if largo > 1024 * 1024:
            self._responder(413, {"error": "Cuerpo demasiado grande"})
            return
        crudo = self.rfile.read(largo) if largo else b"{}"
        try:
            datos = json.loads(crudo)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            self._responder(400, {"error": "JSON inválido"})
            return

        if not isinstance(datos, dict):
            self._responder(400, {"error": "El JSON debe ser un objeto"})
            return

        listas = {
            "/truco/envido": ("canto", "ENVIDO", {"ENVIDO", "ENVIDO_ENVIDO", "REAL_ENVIDO", "FALTA_ENVIDO"}),
            "/truco/truco": ("nivel", None, {None, "TRUCO", "RETRUCO", "VALE_CUATRO"}),
            "/truco/responder": ("respuesta", "QUIERO", {"QUIERO", "NO_QUIERO", "ENVIDO", "ENVIDO_ENVIDO", "REAL_ENVIDO", "FALTA_ENVIDO", "TRUCO", "RETRUCO", "VALE_CUATRO"}),
            "/truco/modo_ia": ("modo", "utilidad_adaptativa", set(AgenteTrucoIA.MODOS_DISPONIBLES)),
        }
        if self.path in listas:
            campo, defecto, permitidos = listas[self.path]
            valor = datos.get(campo, defecto)
            if isinstance(valor, str) and self.path in ("/truco/envido", "/truco/responder"):
                valor = valor.upper()
            elif isinstance(valor, str) and self.path == "/truco/modo_ia":
                valor = valor.strip().lower()
            if not (valor is None or isinstance(valor, str)) or valor not in permitidos:
                self._responder(400, {"error": f"{campo} inválido"})
                return
        if self.path.startswith("/espectador/"):
            codigo, respuesta = self._procesar_espectador(datos)
            self._responder(codigo, respuesta)
            return
        if self.path == "/truco/reiniciar":
            meta = datos.get("meta_puntos", 15)
            manos = datos.get("manos")
            if manos is not None and (type(manos) is not int or not 1 <= manos <= 10):
                self._responder(400, {"error": "Cantidad de manos inválida"})
                return
            if manos is None and (type(meta) is not int or meta not in (15, 30)):
                self._responder(400, {"error": "Meta inválida"})
                return
        if self.path == "/truco/adaptacion" and type(datos.get('activa')) is not bool:
            self._responder(400, {"error": "Estado de adaptación inválido"})
            return
        if self.path == "/truco/jugar":
            carta = datos.get("carta")
            if (not isinstance(carta, dict) or type(carta.get("valor")) is not int
                or carta["valor"] not in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)
                or carta.get("palo") not in ("Espada", "Basto", "Oro", "Copa")):
                self._responder(400, {"error": "Carta inválida"})
                return

        with self.server.lock:
            codigo, respuesta = self._procesar_post(datos)
            if respuesta is not None:
                respuesta = copy.deepcopy(respuesta)
        if respuesta is None:
            self.send_response(codigo)
            self.end_headers()
        else:
            self._responder(codigo, respuesta)

    def _procesar_post(self, datos):
        partida_truco = self.server.partida
        agente = self.server.agente
        detector_vision = self.server.detector
        if self.path.startswith("/truco/"):
            self.server.mesa_para_humano()
        self.server.asegurar_robot()
        if self.path == "/comando":
            texto = str(datos.get("texto", "")).strip()
            idioma = str(datos.get("idioma", "es"))
            if not texto:
                return 400, {"error": "texto vacío"}
            resultado = agente.procesar(texto)
            resultado["idioma"] = idioma
            print(f"  [VOZ-{idioma}] \"{texto}\" -> {resultado['tipo']} | {resultado.get('mensaje', '')}")
            return 200, resultado

        elif self.path == "/truco/detectar":
            if not getattr(detector_vision, "disponible", True):
                return 200, {"palo": "Desconocido", "valor": 0, "confianza": 0.0,
                             "disponible": False, "error": MENSAJE_SIN_OPENCV}
            b64_img = datos.get("imagen", "")
            res = detector_vision.procesar_frame_base64(b64_img, cartas_mano=partida_truco.cartas_humano_disponibles if datos.get("usar_mano", True) else None)
            if res:
                return 200, res
            else:
                return 200, {"palo": "Desconocido", "valor": 0, "confianza": 0.0}

        elif self.path == "/truco/jugar":
            carta = datos.get("carta")
            resultado = partida_truco.jugar_carta_humano(carta)
            return 200, resultado

        elif self.path == "/truco/envido":
            canto = str(datos.get("canto", "ENVIDO")).upper()
            resultado = partida_truco.cantar_envido_humano(canto)
            return 200, resultado

        elif self.path == "/truco/truco":
            nivel = datos.get("nivel")
            resultado = partida_truco.cantar_truco_humano(nivel)
            return 200, resultado

        elif self.path == "/truco/responder":
            respuesta = str(datos.get("respuesta", "QUIERO")).upper()
            resultado = partida_truco.responder_canto_humano(respuesta)
            return 200, resultado

        elif self.path == "/truco/mazo":
            resultado = partida_truco.irse_al_mazo_humano()
            return 200, resultado

        elif self.path == "/truco/voz":
            texto = str(datos.get("texto", "")).strip()
            resultado = partida_truco.procesar_voz_truco(texto)
            return 200, resultado

        elif self.path == "/truco/nuevo":
            est = partida_truco.nueva_mano()
            return 200, est if "error" in est else {"mensaje": "Nueva mano repartida", "estado": est}

        elif self.path == "/truco/reiniciar":
            manos = datos.get("manos")
            if manos is not None:
                est = partida_truco.reiniciar_partido(manos=manos)
                return 200, {"mensaje": f"Partida corta de {manos} manos", "estado": est}
            meta = int(datos.get("meta_puntos", 15))
            est = partida_truco.reiniciar_partido(meta_puntos=meta)
            return 200, {"mensaje": f"Partido reiniciado a {meta} puntos", "estado": est}

        elif self.path == "/truco/modo_ia":
            modo = str(datos.get("modo", "utilidad_adaptativa"))
            if not partida_truco.ia.mode_available(modo.strip().lower()):
                return 409, {"error": "Modo no disponible"}
            est = partida_truco.cambiar_modo_ia(modo)
            return 200, {"mensaje": f"Modo de IA actualizado a {modo}", "estado": est}

        elif self.path == "/truco/adaptacion":
            partida_truco.adapt_to_opponent = datos['activa']
            return 200, {"estado": partida_truco.estado_publico()}

        elif self.path == "/truco/perfil/reset":
            partida_truco.reset_profile()
            return 200, {"estado": partida_truco.estado_publico()}

        else:
            return 404, None

    def _procesar_espectador(self, datos):
        """POST /espectador/*: the spectator match controls (ADR 0064)."""
        from espectador import Largo, MODELOS, ModeloNoCargado, VELOCIDADES, modelo_disponible
        from sim.cartas import CAMARAS
        reproductor = self.server.espectador
        accion = self.path[len("/espectador/"):]
        if accion == "empezar":
            ias = datos.get("ias")
            if (not isinstance(ias, list) or len(ias) != 2
                    or any(not isinstance(c, str) or c not in MODELOS for c in ias)):
                return 400, {"error": "Elegí dos IAs válidas"}
            root = self.server.evolution_root / "champions"
            if not all(modelo_disponible(c, root) for c in ias):
                return 409, {"error": "Esa IA no está disponible en esta instalación"}
            try:
                largo = Largo.de_dict(datos.get("largo", {"tipo": "puntos", "meta": 15}))
            except ValueError as error:
                return 400, {"error": str(error)}
            # The table switches to the spectator match only once both AIs loaded.
            try:
                return 200, reproductor.empezar(ias, largo, al_cargar=self.server.mesa_para_espectador)
            except ModeloNoCargado as error:
                print(f"  [AVISO] {error}")
                return 409, {"error": "No se pudo cargar esa IA", "ia": error.clave}
        if accion == "velocidad":
            velocidad = datos.get("velocidad")
            if isinstance(velocidad, bool) or velocidad not in VELOCIDADES:
                return 400, {"error": "Velocidad inválida"}
            return 200, reproductor.poner_velocidad(velocidad)
        if accion == "camara":
            nombre = datos.get("nombre")
            if nombre not in CAMARAS:
                return 400, {"error": "Cámara inválida"}
            reproductor.mesa.camara(nombre)
            return 200, reproductor.snapshot()
        if accion == "revancha":
            if reproductor.partida is None:
                return 409, {"error": "No hay partida para revancha"}
            try:
                return 200, reproductor.revancha(al_cargar=self.server.mesa_para_espectador)
            except ModeloNoCargado as error:
                print(f"  [AVISO] {error}")
                return 409, {"error": "No se pudo cargar esa IA", "ia": error.clave}
        acciones = {"pausa": reproductor.pausar, "seguir": reproductor.seguir,
                    "paso": reproductor.paso, "detener": reproductor.detener}
        if accion not in acciones:
            return 404, {"error": "Acción desconocida"}
        return 200, acciones[accion]()

    @staticmethod
    def _origen_run(root, run):
        """Live run directory when it has data, else the committed copy (run 2: next to the shipped champion)."""
        name = "phase2-" + run
        live = root / "training_output" / name
        if (live / "generations.jsonl").is_file() or (live / "progress.json").is_file():
            return live
        return root / "champions" / name

    def _origen_evolucion(self, fase):
        """Live run directory when it has data, else the committed copies."""
        live = self.server.evolution_root / "training_output" / fase
        if (live / "generations.jsonl").is_file() or (live / "progress.json").is_file():
            return live
        return self.server.evolution_root / "champions" / fase

    def _responder(self, codigo: int, data: dict) -> None:
        cuerpo = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)


def crear_servidor(partida, puerto=PUERTO_DEFAULT, sin_robot=False, evolution_root=None):
    return ServidorTruco(("127.0.0.1", puerto), partida, sin_robot, evolution_root)


def main() -> None:
    sin_robot_flag = "--sin-robot" in sys.argv
    abrir_evolution = "--abrir-evolucion" in sys.argv
    abrir_nav = "--abrir" in sys.argv or abrir_evolution
    puerto = PUERTO_DEFAULT
    for arg in sys.argv:
        if arg.startswith("--puerto="):
            puerto = int(arg.split("=", 1)[1])

    partida_truco = PartidaTruco(None)
    server = crear_servidor(partida_truco, puerto, sin_robot_flag)
    if not sin_robot_flag and server.asegurar_robot() is None:
        print("  [AVISO] Simulador aún no detectado. Se conectará automáticamente en cuanto abras sim.bat.")
    if not server.detector.disponible:
        print(f"  [AVISO] {MENSAJE_SIN_OPENCV}")
    aviso_3d = aviso_sin_mujoco()
    if aviso_3d and not sin_robot_flag:
        print(f"  [AVISO] {aviso_3d}")

    # The deck is committed; redraw it only if someone deleted the PNGs.
    if not (DIR_CARTAS / "1_Espada.png").is_file():
        try:
            from generar_baraja import generar_baraja_40
            generar_baraja_40(str(DIR_CARTAS))
        except ImportError:
            print("  [AVISO] Faltan las cartas y Pillow para dibujarlas: ejecutá instalar.bat.")

    url = f"http://127.0.0.1:{server.server_port}"
    if abrir_evolution:
        url += "/evolucion"
    print("\n  =========================================================")
    print("  [OK] TRUCO & VOZ CON OTTOMAN (Unitree G1) LISTO")
    print(f"  [WEB] Abri en Chrome o Edge: {url}")
    print("  =========================================================\n")

    if abrir_nav:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server._espectador is not None:
            server._espectador.cerrar()
        server.server_close()
        if server.robot is not None:
            try:
                server.robot.detenerse()
                server.robot.desconectar()
            except Exception:
                pass


if __name__ == "__main__":
    main()
