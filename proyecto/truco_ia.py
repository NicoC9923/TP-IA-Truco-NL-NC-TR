"""Motor de Inteligencia Artificial para el Agente Robot Jugador de Truco (Ottoman G1).

Utiliza las probabilidades precomputadas del dataset T1K22.json (474.000 manos)
combinadas con conciencia de tanteador y una cuota controlada de farol (bluff):
- Cantar y responder Envido / Real Envido / Falta Envido
- Cantar y responder Truco / Retruco / Vale Cuatro
- Selección estratégica de carta según la baza (1ra, 2da, 3ra) y resultado previo
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import truco_reglas as tr


class AgenteTrucoIA:
    def __init__(self, stats_path: Optional[str] = None, prob_farol: float = 0.14):
        if stats_path is None:
            stats_path = str(Path(__file__).resolve().parent / "truco_stats.json")

        with open(stats_path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)

        self.prob_farol = prob_farol

    def prob_envido(self, mano: List[Dict], es_mano: bool) -> Tuple[int, float]:
        """Devuelve (tantos, probabilidad_de_ganar) según el dataset de 474.000 manos."""
        tantos = tr.calcular_envido(mano)
        pts_key = str(tantos)
        e_info = self.stats["envido"].get(
            pts_key,
            {"prob_ganar_siendo_mano": 0.5, "prob_ganar_siendo_pie": 0.5},
        )
        prob = e_info["prob_ganar_siendo_mano"] if es_mano else e_info["prob_ganar_siendo_pie"]
        return tantos, float(prob)

    def decidir_iniciar_envido(
        self,
        mano_completa: List[Dict],
        es_mano: bool,
        puntos_humano: int = 0,
        puntos_robot: int = 0,
        meta_puntos: int = 15,
    ) -> Tuple[Optional[str], str]:
        """Decide si el robot inicia el canto de Envido en primera ronda.

        Devuelve (canto_o_None, frase_criolla).
        """
        tantos, prob = self.prob_envido(mano_completa, es_mano)
        falta_humano = meta_puntos - puntos_humano
        falta_robot = meta_puntos - puntos_robot

        # Si el rival está a 1 punto de ganar el partido, no regalar el no-quiero salvo mano fuerte
        if falta_humano <= 1 and prob >= 0.65:
            return "FALTA_ENVIDO", "¡A todo o nada, Falta Envido!"

        if prob >= 0.94:  # 31, 32 o 33
            if falta_robot <= 4 or random.random() < 0.45:
                return "FALTA_ENVIDO", "¡Falta Envido te canto!"
            return "REAL_ENVIDO", "¡Real Envido!"
        if prob >= 0.85:  # 29 o 30
            if random.random() < 0.35:
                return "REAL_ENVIDO", "¡Real Envido!"
            return "ENVIDO", "¡Envido!"
        if prob >= 0.68:  # 26 a 28
            return "ENVIDO", "¡Envido!"

        # Cuota de picardía / farol ("mentir" siendo pie o mano con pocos puntos)
        if not es_mano and random.random() < self.prob_farol:
            return "ENVIDO", "¡Envido! A ver si te animás..."

        return None, ""

    def responder_envido(
        self,
        mano_completa: List[Dict],
        canto_actual: str,
        es_mano: bool,
        puntos_acumulados_no_quiero: int = 1,
        puntos_humano: int = 0,
        puntos_robot: int = 0,
        meta_puntos: int = 15,
    ) -> Tuple[str, str]:
        """Responde a un canto de envido del humano ('ENVIDO', 'ENVIDO_ENVIDO', 'REAL_ENVIDO', 'FALTA_ENVIDO').

        Devuelve (decision, frase_voz):
        Decisiones posibles: 'QUIERO', 'NO_QUIERO', 'ENVIDO_ENVIDO', 'REAL_ENVIDO', 'FALTA_ENVIDO'
        """
        tantos, prob = self.prob_envido(mano_completa, es_mano)
        falta_humano = meta_puntos - puntos_humano

        # Si diciendo "No quiero" el humano igual gana el partido, estamos obligados a querer
        if puntos_humano + puntos_acumulados_no_quiero >= meta_puntos:
            return "QUIERO", f"¡No me queda otra, quiero con {tantos}!"

        if canto_actual == "ENVIDO":
            if prob >= 0.93:
                return "FALTA_ENVIDO", "¡Para qué achicarse, Falta Envido!"
            if prob >= 0.84:
                return "REAL_ENVIDO", "¡Envido está primero... Real Envido!"
            if prob >= 0.78 and random.random() < 0.5:
                return "ENVIDO_ENVIDO", "¡Envido otra vez!"
            if prob >= 0.58:
                return "QUIERO", f"¡Quiero! Tengo {tantos}."
            if random.random() < (self.prob_farol * 0.5):
                return "REAL_ENVIDO", "¡Real Envido, si te da el cuero!"
            return "NO_QUIERO", "Son buenas, no quiero."

        if canto_actual == "ENVIDO_ENVIDO":
            if prob >= 0.94:
                return "FALTA_ENVIDO", "¡Falta Envido!"
            if prob >= 0.70:
                return "QUIERO", f"¡Quiero! Tengo {tantos}."
            return "NO_QUIERO", "No quiero, dejá nomás."

        if canto_actual == "REAL_ENVIDO":
            if prob >= 0.94:
                return "FALTA_ENVIDO", "¡Falta Envido carajo!"
            if prob >= 0.72:
                return "QUIERO", f"¡Quiero! {tantos} tengo."
            return "NO_QUIERO", "Paso, no quiero."

        if canto_actual == "FALTA_ENVIDO":
            # Si la falta vale pocos puntos (final de partido), umbral más flexible
            pts_falta = tr.puntos_falta_envido(puntos_humano, puntos_robot, meta_puntos)
            umbral = 0.75 if (pts_falta <= 3 or falta_humano <= 2) else 0.87
            if prob >= umbral:
                return "QUIERO", f"¡Quiero la falta! Tengo {tantos}."
            return "NO_QUIERO", "Demasiado riesgo, no quiero."

        return "NO_QUIERO", "No quiero."

    def percentil_fuerza_truco(self, cartas: List[Dict]) -> float:
        """Calcula el percentil de fuerza (0.0 a 1.0) de la mano en base al dataset."""
        if not cartas:
            return 0.0
        # Si ya tiró cartas, proyectar a escala de 3 cartas para consultar la tabla
        poder = tr.poder_mano_truco(cartas)
        if len(cartas) == 2:
            poder = poder * (1.9 / 1.6)
        elif len(cartas) == 1:
            poder = poder * 1.9
        poder_red = round(min(23.5, max(0.0, poder)), 1)
        p_str = str(poder_red)
        if p_str in self.stats["truco_poder_percentil"]:
            return float(self.stats["truco_poder_percentil"][p_str])
        # Buscar la clave más cercana
        claves = [float(k) for k in self.stats["truco_poder_percentil"].keys()]
        mas_cercana = min(claves, key=lambda k: abs(k - poder_red))
        return float(self.stats["truco_poder_percentil"][str(mas_cercana)])

    def evaluar_situacion_baza(
        self,
        cartas_disponibles: List[Dict],
        ganadores_ronda: List[str],
        carta_rival_mesa: Optional[Dict] = None,
    ) -> float:
        """Estima la fuerza efectiva (0.0 a 1.0) considerando las bazas ya ganadas y la carta en mesa."""
        pct_base = self.percentil_fuerza_truco(cartas_disponibles)

        # Si el rival ya tiró carta en esta baza, vemos si la matamos
        if carta_rival_mesa is not None and cartas_disponibles:
            p_rival = tr.valor_carta_truco(carta_rival_mesa["valor"], carta_rival_mesa["palo"])
            max_propia = max(tr.valor_carta_truco(c["valor"], c["palo"]) for c in cartas_disponibles)
            if max_propia > p_rival:
                pct_base = min(0.99, pct_base + 0.18)
            elif max_propia < p_rival and "robot" not in ganadores_ronda:
                pct_base = max(0.05, pct_base - 0.25)

        if ganadores_ronda:
            if ganadores_ronda[0] == "robot":
                pct_base = min(0.98, pct_base + 0.15)
            elif ganadores_ronda[0] == "humano":
                pct_base = max(0.05, pct_base - 0.10)

        return pct_base

    def decidir_iniciar_truco(
        self,
        cartas_disponibles: List[Dict],
        nivel_actual: str,
        ganadores_ronda: List[str],
        carta_rival_mesa: Optional[Dict] = None,
    ) -> Tuple[Optional[str], str]:
        """Decide si el robot inicia o sube la apuesta de Truco en su turno."""
        fuerza = self.evaluar_situacion_baza(cartas_disponibles, ganadores_ronda, carta_rival_mesa)

        if nivel_actual == "NADA":
            if fuerza >= 0.72:
                return "TRUCO", "¡Truco!"
            # Farol ocasional en 2da o 3ra ronda
            if len(ganadores_ronda) >= 1 and random.random() < self.prob_farol:
                return "TRUCO", "¡Truco te canto, a ver qué tenés!"
        elif nivel_actual == "TRUCO":
            if fuerza >= 0.86:
                return "RETRUCO", "¡Quiero Retruco!"
        elif nivel_actual == "RETRUCO":
            if fuerza >= 0.93:
                return "VALE_CUATRO", "¡Quiero Vale Cuatro!"

        return None, ""

    def responder_canto_truco(
        self,
        cartas_disponibles: List[Dict],
        canto_rival: str,
        ganadores_ronda: List[str],
        carta_rival_mesa: Optional[Dict] = None,
        puntos_humano: int = 0,
        meta_puntos: int = 15,
    ) -> Tuple[str, str]:
        """Responde cuando el humano canta TRUCO, RETRUCO o VALE_CUATRO.

        Devuelve (decision, frase_criolla):
        Decisiones: 'QUIERO', 'NO_QUIERO', 'RETRUCO', 'VALE_CUATRO'
        """
        fuerza = self.evaluar_situacion_baza(cartas_disponibles, ganadores_ronda, carta_rival_mesa)
        puntos_no_quiero = {"TRUCO": 1, "RETRUCO": 2, "VALE_CUATRO": 3}.get(canto_rival, 1)

        # Si irse al mazo hace que el humano gane el partido, jugársela
        if puntos_humano + puntos_no_quiero >= meta_puntos and fuerza >= 0.25:
            return "QUIERO", "¡No me puedo ir al mazo ahora, Quiero!"

        if canto_rival == "TRUCO":
            if fuerza >= 0.87:
                return "RETRUCO", "¡Quiero Retruco!"
            if fuerza >= 0.44:
                return "QUIERO", "¡Quiero!"
            if random.random() < (self.prob_farol * 0.6):
                return "RETRUCO", "¡Quiero Retruco, no me asustás!"
            return "NO_QUIERO", "Me voy al mazo, son buenas."

        if canto_rival == "RETRUCO":
            if fuerza >= 0.92:
                return "VALE_CUATRO", "¡Quiero Vale Cuatro!"
            if fuerza >= 0.56:
                return "QUIERO", "¡Quiero ese Retruco!"
            return "NO_QUIERO", "Hasta acá llegué, me voy al mazo."

        if canto_rival == "VALE_CUATRO":
            if fuerza >= 0.68:
                return "QUIERO", "¡Quiero el Vale Cuatro con todas las de la ley!"
            return "NO_QUIERO", "Al mazo me voy."

        return "QUIERO", "¡Quiero!"

    def elegir_carta_a_jugar(
        self,
        cartas_disponibles: List[Dict],
        ronda: int,
        carta_rival_mesa: Optional[Dict] = None,
        ganadores_ronda: Optional[List[str]] = None,
    ) -> Dict:
        """Selecciona la mejor carta a tirar de las disponibles según la baza actual."""
        if len(cartas_disponibles) == 1:
            return cartas_disponibles[0]

        ganadores_ronda = ganadores_ronda or []
        cartas_ordenadas = sorted(
            cartas_disponibles,
            key=lambda c: tr.valor_carta_truco(c["valor"], c["palo"]),
        )

        # Caso 1: El humano ya tiró su carta en esta baza y nos toca responder
        if carta_rival_mesa is not None:
            poder_rival = tr.valor_carta_truco(carta_rival_mesa["valor"], carta_rival_mesa["palo"])
            matadoras = [
                c for c in cartas_ordenadas
                if tr.valor_carta_truco(c["valor"], c["palo"]) > poder_rival
            ]
            pardas = [
                c for c in cartas_ordenadas
                if tr.valor_carta_truco(c["valor"], c["palo"]) == poder_rival
            ]

            if matadoras:
                # Tirar la carta mínima necesaria para matar la del rival
                return matadoras[0]
            if pardas:
                # Si empatar nos da la mano (ganamos 1ra) o estamos en 1ra, empardar
                if ronda == 1 or (ganadores_ronda and ganadores_ronda[0] == "robot"):
                    return pardas[0]
            # Si no llegamos a matar, descartar la más baja
            return cartas_ordenadas[0]

        # Caso 2: Nos toca abrir la baza a nosotros
        if ronda == 1:
            # En primera ronda, abrir con la intermedia o la más alta para asegurar primera
            if len(cartas_ordenadas) == 3:
                return cartas_ordenadas[1] if tr.valor_carta_truco(
                    cartas_ordenadas[1]["valor"], cartas_ordenadas[1]["palo"]
                ) >= 7 else cartas_ordenadas[2]
            return cartas_ordenadas[-1]

        if ronda == 2:
            # Si empatamos primera (parda), en segunda hay que tirar la más alta sí o sí
            if ganadores_ronda and ganadores_ronda[0] == "parda":
                return cartas_ordenadas[-1]
            # Si ganamos primera y tenemos una carta muy alta y una baja, podemos achicar con la baja
            if len(cartas_ordenadas) == 2:
                alta = tr.valor_carta_truco(cartas_ordenadas[1]["valor"], cartas_ordenadas[1]["palo"])
                if alta >= 10:
                    return cartas_ordenadas[0]
            return cartas_ordenadas[-1]

        # En 3ra ronda queda una sola carta
        return cartas_ordenadas[-1]
