"""Servidor local para jugar al Truco contra Ottoman (Unitree G1) y controlarlo por voz.

Integra:
1. Juego completo de Truco Argentino (/truco/estado, /truco/jugar, /truco/envido,
   /truco/truco, /truco/responder, /truco/mazo, /truco/voz, /truco/nuevo,
   /truco/reiniciar, /truco/detectar).
2. Control libre por voz bilingüe (/comando).
3. Conexión automática (y reconexión en caliente) con el simulador MuJoCo.
"""

from __future__ import annotations

import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from robot import Robot
from control_voz import AgenteRobot
from partida_truco import PartidaTruco
from truco_vision import DetectorCartas

PUERTO_DEFAULT = 8080
DIR_PROYECTO = Path(__file__).resolve().parent
PAGINA = DIR_PROYECTO / "voz.html"
DIR_CARTAS = DIR_PROYECTO / "cartas"

robot_instancia = None
sin_robot_flag = False
agente = None
partida_truco = None
detector_vision = None


def asegurar_robot():
    """Intenta conectar al simulador si aún no estaba conectado y se abrió después."""
    global robot_instancia
    if sin_robot_flag:
        return None
    if robot_instancia is not None:
        return robot_instancia
    try:
        r = Robot()
        r.conectar()
        robot_instancia = r
        if agente is not None:
            agente.robot = r
            agente.ejecutor.robot = r
        if partida_truco is not None:
            partida_truco.robot = r
        print("  [OK] Conectado dinámicamente al simulador del Robot.")
        return r
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, formato, *args):
        pass

    def do_GET(self):
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
        elif ruta_limpia == "/truco/estado":
            asegurar_robot()
            est = partida_truco.estado_publico()
            est["robot_conectado"] = robot_instancia is not None
            self._responder(200, est)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        asegurar_robot()
        largo = int(self.headers.get("Content-Length", 0))
        crudo = self.rfile.read(largo) if largo else b"{}"
        try:
            datos = json.loads(crudo)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            self._responder(400, {"error": "JSON inválido"})
            return

        if self.path == "/comando":
            texto = str(datos.get("texto", "")).strip()
            idioma = str(datos.get("idioma", "es"))
            if not texto:
                self._responder(400, {"error": "texto vacío"})
                return
            resultado = agente.procesar(texto)
            resultado["idioma"] = idioma
            print(f"  [VOZ-{idioma}] \"{texto}\" -> {resultado['tipo']} | {resultado.get('mensaje', '')}")
            self._responder(200, resultado)

        elif self.path == "/truco/detectar":
            b64_img = datos.get("imagen", "")
            usar_mano = bool(datos.get("usar_mano", True))
            cartas_disp = partida_truco.cartas_humano_disponibles if usar_mano else None
            res = detector_vision.procesar_frame_base64(b64_img, cartas_mano=cartas_disp)
            if res:
                self._responder(200, res)
            else:
                self._responder(200, {"palo": "Desconocido", "valor": 0, "confianza": 0.0})

        elif self.path == "/truco/jugar":
            carta = datos.get("carta")
            if not carta:
                self._responder(400, {"error": "Falta parámetro carta"})
                return
            resultado = partida_truco.jugar_carta_humano(carta)
            self._responder(200, resultado)

        elif self.path == "/truco/envido":
            canto = str(datos.get("canto", "ENVIDO")).upper()
            resultado = partida_truco.cantar_envido_humano(canto)
            self._responder(200, resultado)

        elif self.path == "/truco/truco":
            nivel = datos.get("nivel")
            resultado = partida_truco.cantar_truco_humano(nivel)
            self._responder(200, resultado)

        elif self.path == "/truco/responder":
            respuesta = str(datos.get("respuesta", "QUIERO")).upper()
            resultado = partida_truco.responder_canto_humano(respuesta)
            self._responder(200, resultado)

        elif self.path == "/truco/mazo":
            resultado = partida_truco.irse_al_mazo_humano()
            self._responder(200, resultado)

        elif self.path == "/truco/voz":
            texto = str(datos.get("texto", "")).strip()
            resultado = partida_truco.procesar_voz_truco(texto)
            self._responder(200, resultado)

        elif self.path == "/truco/nuevo":
            est = partida_truco.nueva_mano()
            self._responder(200, {"mensaje": "Nueva mano repartida", "estado": est})

        elif self.path == "/truco/reiniciar":
            meta = int(datos.get("meta_puntos", 15))
            est = partida_truco.reiniciar_partido(meta_puntos=meta)
            self._responder(200, {"mensaje": f"Partido reiniciado a {meta} puntos", "estado": est})

        else:
            self.send_response(404)
            self.end_headers()

    def _responder(self, codigo: int, data: dict) -> None:
        cuerpo = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)


def main() -> None:
    global sin_robot_flag, agente, partida_truco, detector_vision, robot_instancia

    sin_robot_flag = "--sin-robot" in sys.argv
    abrir_nav = "--abrir" in sys.argv
    puerto = PUERTO_DEFAULT
    for arg in sys.argv:
        if arg.startswith("--puerto="):
            puerto = int(arg.split("=", 1)[1])

    if not sin_robot_flag:
        asegurar_robot()
        if robot_instancia is None:
            print("  [AVISO] Simulador aún no detectado. Se conectará automáticamente en cuanto abras sim.bat.")

    agente = AgenteRobot(robot_instancia)
    partida_truco = PartidaTruco(robot_instancia)
    detector_vision = DetectorCartas()

    # Asegurar que las 40 imágenes de la Baraja Española estén generadas
    if not (DIR_CARTAS / "1_Espada.png").is_file() and (DIR_CARTAS / "template_naipes.jpg").is_file():
        from generar_baraja import generar_baraja_40
        generar_baraja_40(str(DIR_CARTAS / "template_naipes.jpg"), str(DIR_CARTAS))

    server = ThreadingHTTPServer(("127.0.0.1", puerto), Handler)
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
        if robot_instancia is not None:
            try:
                robot_instancia.detenerse()
                robot_instancia.desconectar()
            except Exception:
                pass


if __name__ == "__main__":
    main()
