"""Servidor local para jugar al Truco contra Ottoman (Unitree G1) y controlarlo por voz.

Integra:
1. Juego completo de Truco Argentino (/truco/estado, /truco/jugar, /truco/envido,
   /truco/truco, /truco/responder, /truco/mazo, /truco/voz, /truco/nuevo,
   /truco/reiniciar, /truco/detectar).
2. Control libre por voz bilingüe (/comando).
3. Conexión automática (y reconexión en caliente) con el simulador MuJoCo.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from robot import Robot
from control_voz import AgenteRobot
from partida_truco import PartidaTruco
from truco_vision import DetectorCartas, MENSAJE_SIN_OPENCV

PUERTO_DEFAULT = 8080
DIR_PROYECTO = Path(__file__).resolve().parent
PAGINA = DIR_PROYECTO / "voz.html"
DIR_CARTAS = DIR_PROYECTO / "cartas"
DIR_FUENTES = DIR_PROYECTO / "fuentes"

class ServidorTruco(ThreadingHTTPServer):
    def __init__(self, direccion, partida, sin_robot):
        super().__init__(direccion, Handler)
        self.lock = threading.RLock()
        self.partida = partida
        self.robot = partida.robot
        self.sin_robot = sin_robot
        self.agente = AgenteRobot(self.robot)
        self.detector = DetectorCartas()
        self.error_conexion = None

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
                print("  [OK] Conectado dinámicamente al simulador del Robot.")
                return robot
            except Exception as error:
                tipo = type(error).__name__
                if tipo != self.error_conexion:
                    print(f"  [AVISO] Error al conectar el simulador: {tipo}")
                    self.error_conexion = tipo
                return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, formato, *args):
        pass

    def do_GET(self):
        partida_truco = self.server.partida
        ruta_limpia = self.path.split("?", 1)[0]
        if ruta_limpia in ("/", "/voz.html"):
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
            "/truco/modo_ia": ("modo", "utilidad_adaptativa", {"reactivo", "utilidad_adaptativa", "neuroevolucion", "refuerzo"}),
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
        if self.path == "/truco/reiniciar":
            meta = datos.get("meta_puntos", 15)
            if type(meta) is not int or meta not in (15, 30):
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
            partida_truco.opponent_profile.reset()
            partida_truco._human_decisions.clear()
            return 200, {"estado": partida_truco.estado_publico()}

        else:
            return 404, None

    def _responder(self, codigo: int, data: dict) -> None:
        cuerpo = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)


def crear_servidor(partida, puerto=PUERTO_DEFAULT, sin_robot=False):
    return ServidorTruco(("127.0.0.1", puerto), partida, sin_robot)


def main() -> None:
    sin_robot_flag = "--sin-robot" in sys.argv
    abrir_nav = "--abrir" in sys.argv
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

    # Asegurar que las 40 imágenes de la Baraja Española estén generadas
    if not (DIR_CARTAS / "1_Espada.png").is_file() and (DIR_CARTAS / "template_naipes.jpg").is_file():
        from generar_baraja import generar_baraja_40
        generar_baraja_40(str(DIR_CARTAS / "template_naipes.jpg"), str(DIR_CARTAS))

    url = f"http://127.0.0.1:{puerto}"
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
        server.server_close()
        if server.robot is not None:
            try:
                server.robot.detenerse()
                server.robot.desconectar()
            except Exception:
                pass


if __name__ == "__main__":
    main()
