"""Agente de Control Libre por Voz (Español + Inglés) para el Robot Unitree G1 / Go2.

Interpreta órdenes en lenguaje natural ("avanzá 2 metros", "girá 90 grados a la derecha",
"saludá", "festejá", "dame la mano", "detente") con validación de seguridad independiente.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Optional


class ClasificadorIntencion:
    """Decide qué acción general solicita el usuario (ES + EN)."""

    TIPOS = (
        "MOVER",
        "GIRAR",
        "DETENERSE",
        "SALUDO",
        "FESTEJO",
        "DAR_LA_MANO",
        "CONSULTAR_ESTADO",
        "DESCONOCIDO",
    )

    def clasificar(self, texto: str) -> str:
        if not texto or not isinstance(texto, str):
            return "DESCONOCIDO"

        t = texto.strip().lower()

        if re.search(
            r"\b(?:c[oó]mo\s+te\s+llam[aá]s|qui[eé]n\s+sos|qu[eé]\s+hora|qu[eé]\s+onda|clima|chiste"
            r"|what.?s\s+your\s+name|who\s+are\s+you|what\s+time|tell\s+me\s+a\s+joke)\b",
            t,
        ):
            return "DESCONOCIDO"

        # 1. DETENERSE
        if re.search(r"\bno\s+(?:avances?|camines?|te\s+muevas?|sigas?)\b", t):
            return "DETENERSE"
        if re.search(r"\b(?:don'?t|do\s+not)\s+(?:move|walk|go|continue)\b", t):
            return "DETENERSE"
        if re.search(
            r"\b(?:deten(?:te|ete|erse|er)|par[aá](?:r|te|lo|todo)?|fren[aá](?:r|te)?|quieto|alto|stop|basta"
            r"|halt|freeze|hold\s+still)\b",
            t,
        ):
            return "DETENERSE"

        # 2. CONSULTAR_ESTADO
        if re.search(r"\b(?:bater[ií]a|telemetr[ií]a|estado|posici[oó]n|status|telemetry|position)\b", t):
            return "CONSULTAR_ESTADO"

        # 3. FESTEJO
        if re.search(r"\b(?:festej[aá](?:r)?|celebr[aá](?:r)?|victoria|celebrate|victory|cheer)\b", t):
            return "FESTEJO"

        # 4. DAR_LA_MANO
        if re.search(r"\b(?:dame\s+la\s+mano|dar\s+la\s+mano|estrech[aá]\s+la\s+mano|shake\s+hands?)\b", t):
            return "DAR_LA_MANO"

        # 5. SALUDO
        if re.search(
            r"\b(?:salud[aá](?:r|te|nos)?|saludo|hac[eé](?:le)?\s+(?:un\s+)?saludo|hacele\s+hola"
            r"|greet|wave|say\s+hi|say\s+hello|hello|hi"
            r"|levant[aá](?:r|te)?\s+(?:el\s+)?(?:brazo|mano)|sub[ií](?:r|te)?\s+(?:el\s+)?brazo"
            r"|move\s+(?:your\s+)?arm\s+up|raise\s+(?:your\s+)?arm|lift\s+(?:your\s+)?arm)\b",
            t,
        ):
            return "SALUDO"

        # 6. GIRAR
        if re.search(
            r"\b(?:gir[aá](?:r|te)?|rot[aá](?:r)?|dobl[aá](?:r)?|vir[aá](?:r)?|media\s+vuelta|vuelta\s+completa"
            r"|turn|rotate|spin|pivot)\b",
            t,
        ):
            return "GIRAR"

        # 7. MOVER
        if re.search(
            r"\b(?:avanz[aá](?:r)?|camin[aá](?:r)?|mu[eé]vete|movete|mover(?:se)?|desplaz[aá](?:r)?|retroced[eé](?:r)?|"
            r"and[aá](?:r)?\s+(?:para|hacia)?|and[aá]\b|march[aá](?:r)?|"
            r"move|walk|advance|go\s+forward|step\s+forward|back\s+up|march)\b",
            t,
        ):
            return "MOVER"

        return "DESCONOCIDO"


class ExtractorParametros:
    """Extrae distancia, velocidad, ángulo y dirección en unidades humanas."""

    def extraer(self, texto: str, tipo: str) -> Dict:
        if not texto or not isinstance(texto, str):
            return {}

        params: Dict = {}
        t = texto.strip().lower()

        m_dist = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:metros?|meters?|m\b)(?!\s*/\s*s)", t)
        if m_dist:
            params["distancia_m"] = float(m_dist.group(1).replace(",", "."))

        m_vel = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:m/s|ms\b|metros?\s*(?:por|/)\s*seg(?:undo)?s?|meters?\s*(?:per|/)\s*sec(?:ond)?s?)",
            t,
        )
        if m_vel:
            params["velocidad_ms"] = float(m_vel.group(1).replace(",", "."))
        elif re.search(r"\b(?:despacio|lento|despacito|slow|slowly)\b", t):
            params["velocidad_ms"] = 0.2
        elif re.search(r"\b(?:r[aá]pido|veloz|r[aá]pidamente|fast|quick|quickly)\b", t):
            params["velocidad_ms"] = 0.5

        m_ang = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:grados?|°|deg\b|degrees?)", t)
        if m_ang:
            val = float(m_ang.group(1).replace(",", "."))
            params["angulo_deg"] = int(val) if val.is_integer() else val
        elif "media vuelta" in t or "half turn" in t:
            params["angulo_deg"] = 180

        if re.search(r"\b(?:derecha|right)\b", t):
            params["direccion"] = "derecha"
        elif re.search(r"\b(?:izquierda|left)\b", t):
            params["direccion"] = "izquierda"
        elif re.search(r"\b(?:atr[aá]s|retroced[eé](?:r)?|back(?:wards?)?)\b", t):
            params["direccion"] = "atras"
        elif re.search(r"\b(?:adelante|forward)\b", t):
            params["direccion"] = "adelante"

        return params


class ValidadorSeguridad:
    """Validador independiente de acciones peligrosas o fuera de rango seguro."""

    PALABRAS_PELIGROSAS = (
        "salta", "saltá", "salto", "saltar", "brinca", "brincar",
        "corre", "corré", "correr", "sprint",
        "empuja", "empujá", "empujar",
        "golpea", "golpeá", "golpear", "pega", "pegá", "pegar",
        "rompe", "rompé", "romper",
        "cae", "caé", "caer",
        "choca", "chocá", "chocar",
        "jump", "jumps", "jumping",
        "run", "runs", "running",
        "push", "pushes", "pushing",
        "hit", "hits", "hitting", "punch",
        "break", "crash", "fall",
    )

    MAX_VELOCIDAD = 1.0
    MAX_DISTANCIA = 10.0
    MAX_ANGULO = 360.0

    def validar(self, texto: str, tipo: str, parametros: Dict):
        texto_norm = (texto or "").lower().strip()
        parametros = parametros or {}

        for palabra in self.PALABRAS_PELIGROSAS:
            if re.search(r"\b" + re.escape(palabra) + r"\b", texto_norm):
                return False, f"palabra peligrosa ('{palabra}')"

        if "velocidad_ms" in parametros:
            vel = abs(float(parametros["velocidad_ms"]))
            if vel > self.MAX_VELOCIDAD:
                return False, f"velocidad excesiva ({vel} m/s > {self.MAX_VELOCIDAD} m/s)"

        if "distancia_m" in parametros:
            dist = abs(float(parametros["distancia_m"]))
            if dist > self.MAX_DISTANCIA:
                return False, f"distancia excesiva ({dist} m > {self.MAX_DISTANCIA} m)"

        if "angulo_deg" in parametros:
            ang = abs(float(parametros["angulo_deg"]))
            if ang > self.MAX_ANGULO:
                return False, f"ángulo excesivo ({ang}° > {self.MAX_ANGULO}°)"

        return True, ""


class EjecutorRobot:
    """Traduce intenciones validadas a llamadas de la API `Robot`."""

    def __init__(self, robot):
        self.robot = robot

    def ejecutar(self, tipo: str, parametros: Dict) -> str:
        p = parametros or {}
        if self.robot is None:
            return f"Simulado (sin robot conectado): {tipo} {p}"

        if tipo == "DETENERSE":
            self.robot.detenerse()
            return "Robot detenido."
        if tipo == "SALUDO":
            self.robot.saludar()
            return "Gesto de saludo realizado."
        if tipo == "FESTEJO":
            if hasattr(self.robot, "festejar"):
                self.robot.festejar()
            else:
                self.robot.saludar()
            return "Gesto de festejo realizado."
        if tipo == "DAR_LA_MANO":
            if hasattr(self.robot, "dar_la_mano"):
                self.robot.dar_la_mano()
            else:
                self.robot.saludar()
            return "Gesto de dar la mano realizado."
        if tipo == "CONSULTAR_ESTADO":
            e = self.robot.verificar_estado()
            return f"Estado actual: {e}"
        if tipo == "MOVER":
            distancia = float(p.get("distancia_m", 0.5))
            velocidad = float(p.get("velocidad_ms", 0.4))
            signo = -1.0 if p.get("direccion") == "atras" else 1.0
            tiempo = abs(distancia) / max(0.05, abs(velocidad))
            self.robot.avanzar(velocidad=signo * abs(velocidad), tiempo=tiempo)
            sentido = "atrás" if signo < 0 else "adelante"
            return f"Se desplazó {distancia:g} m hacia {sentido}."
        if tipo == "GIRAR":
            grados = float(p.get("angulo_deg", 90))
            velocidad_ang = 0.8
            signo = -1.0 if p.get("direccion") == "derecha" else 1.0
            tiempo = math.radians(abs(grados)) / velocidad_ang
            self.robot.girar(velocidad=signo * velocidad_ang, tiempo=tiempo)
            lado = "derecha" if signo < 0 else "izquierda"
            return f"Giró {grados:g}° a la {lado}."

        return f"Acción desconocida: {tipo}"


class AgenteRobot:
    def __init__(self, robot=None):
        self.robot = robot
        self.clasificador = ClasificadorIntencion()
        self.extractor = ExtractorParametros()
        self.validador = ValidadorSeguridad()
        self.ejecutor = EjecutorRobot(robot)
        self.historial = []

    def procesar(self, texto: str) -> Dict:
        texto_limpio = (texto or "").strip()
        tipo = self.clasificador.clasificar(texto_limpio)
        parametros = self.extractor.extraer(texto_limpio, tipo)
        es_seguro, motivo = self.validador.validar(texto_limpio, tipo, parametros)

        if not es_seguro:
            resultado = {
                "tipo": tipo,
                "parametros": parametros,
                "ejecutar": False,
                "bloqueado": True,
                "texto_original": texto,
                "mensaje": f"BLOQUEADO: {motivo}",
            }
        elif tipo == "DESCONOCIDO":
            resultado = {
                "tipo": "DESCONOCIDO",
                "parametros": parametros,
                "ejecutar": False,
                "bloqueado": False,
                "texto_original": texto,
                "mensaje": "Comando no reconocido.",
            }
        else:
            mensaje = self.ejecutor.ejecutar(tipo, parametros)
            resultado = {
                "tipo": tipo,
                "parametros": parametros,
                "ejecutar": True,
                "bloqueado": False,
                "texto_original": texto,
                "mensaje": mensaje,
            }

        self.historial.append(resultado)
        return resultado
