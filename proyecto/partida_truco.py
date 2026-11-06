"""Módulo de Control de Partida de Truco 1 vs 1: Humano vs Robot Unitree G1 (Ottoman).

Gestiona:
- Reparto único de mazo de 40 cartas (3 para el humano, 3 para el robot).
- Turnos reales por baza (mano en 1ra, ganador de baza previa en 2da y 3ra, desempate por parda).
- Ida y vuelta completo de cantos de Envido (Envido, Envido-Envido, Real Envido, Falta Envido).
- Ida y vuelta completo de cantos de Truco (Truco, Retruco, Vale Cuatro) e Irse al mazo.
- Revelación de las 3 cartas de Ottoman al finalizar cada mano y auditoría de bluffs (mentiras).
- Estadísticas de engaños por partida e integración con el historial y aprendizaje adaptativo.
- Reacciones físicas asíncronas del robot G1 sentado en la mesa del simulador MuJoCo.
"""

from __future__ import annotations

import random
import re
import threading
from typing import Dict, List, Optional, Tuple
import truco_reglas as tr
from truco_ia import AgenteTrucoIA


class PartidaTruco:
    def __init__(self, robot=None, meta_puntos: int = 15, historial_path: Optional[str] = None,
                 rng: random.Random | None = None):
        self.robot = robot
        self.rng = rng if rng is not None else random.Random()
        self._robot_lock = threading.Lock()
        self.ia = AgenteTrucoIA(historial_path=historial_path, rng=self.rng)

        self.puntos_humano = 0
        self.puntos_robot = 0
        self.meta_puntos = meta_puntos
        self.partido_terminado = False
        self.ganador_partido: Optional[str] = None

        self.es_mano_humano = True
        self.cartas_humano: List[Dict] = []
        self.cartas_humano_disponibles: List[Dict] = []
        self.cartas_robot: List[Dict] = []
        self.cartas_robot_disponibles: List[Dict] = []

        self.ronda_actual = 1
        self.turno_carta = "humano"
        self.mesa: List[Dict] = []
        self.ganadores_ronda: List[str] = []

        # Estado de Envido
        self.envido_resuelto = False
        self.puntos_envido_acumulados = 0
        self.ultimo_canto_envido: Optional[str] = None
        self.resultado_envido_detalle: Optional[Dict] = None

        # Estado de Truco
        self.apuesta_truco = "NADA"  # 'NADA', 'TRUCO', 'RETRUCO', 'VALE_CUATRO'
        self.puntos_truco_actuales = 1
        self.quien_tiene_quiero_truco: Optional[str] = None

        # Si el robot cantó algo y espera que el humano diga Quiero / No Quiero / Subir
        self.canto_pendiente: Optional[Dict] = None

        # Seguimiento de mentiras (bluffs) en la mano y en la partida
        self.bluffs_mano_actual: List[Dict] = []
        self._ultimo_bluff_envido_mano: Optional[Dict] = None
        self._ultimo_bluff_truco_mano: Optional[Dict] = None
        self._humano_canto_envido_mano: bool = False
        self._humano_canto_truco_mano: bool = False
        self._mano_analizada: bool = False
        self.estadisticas_partida: Dict = self._crear_estadisticas_vacias()

        self.mano_terminada = False
        self.ganador_mano: Optional[str] = None
        self.ultima_frase_robot = "¡Listo para jugar unos trucos! Mostrame tu carta o cantá."
        self.log_mensajes: List[str] = []

        self.nueva_mano(primera_vez=True)

    def _crear_estadisticas_vacias(self) -> Dict:
        return {
            "manos_jugadas": 0,
            "envido": {
                "cantados_robot": 0,
                "bluffs_intentados": 0,
                "bluffs_exitosos": 0,
                "bluffs_descubiertos": 0,
            },
            "truco": {
                "cantados_robot": 0,
                "bluffs_intentados": 0,
                "bluffs_exitosos": 0,
                "bluffs_descubiertos": 0,
            },
        }

    # ---------- reacciones físicas asíncronas del G1 ----------
    def _reaccion_robot(self, evento: str) -> None:
        """Ejecuta gestos en el G1 sentado frente a la mesa sin bloquear la respuesta web."""
        if self.robot is None:
            return

        def _worker():
            if not self._robot_lock.acquire(blocking=False):
                return
            try:
                if evento == "repartir":
                    if hasattr(self.robot, "repartir"):
                        self.robot.repartir(esperar=True)
                    elif hasattr(self.robot, "hacer_gesto"):
                        self.robot.hacer_gesto("repartiendo", duracion=1.8, esperar=True)
                elif evento == "festejo":
                    if hasattr(self.robot, "festejar"):
                        self.robot.festejar(esperar=True)
                    else:
                        self.robot.saludar()
                elif evento == "desafio":
                    if hasattr(self.robot, "desafiar"):
                        self.robot.desafiar(esperar=True)
                elif evento == "mazo":
                    if hasattr(self.robot, "gesto_mazo"):
                        self.robot.gesto_mazo(esperar=True)
                elif evento == "saludo":
                    self.robot.saludar()
                elif evento == "fin_partido":
                    if hasattr(self.robot, "dar_la_mano"):
                        self.robot.dar_la_mano()
                    else:
                        self.robot.saludar()
            except Exception:
                pass
            finally:
                self._robot_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def registrar(self, mensaje: str) -> None:
        self.log_mensajes.append(mensaje)
        if len(self.log_mensajes) > 60:
            self.log_mensajes = self.log_mensajes[-60:]
        try:
            print(f"  [TRUCO] {mensaje}")
        except UnicodeEncodeError:
            limpio = mensaje.encode("ascii", errors="ignore").decode("ascii")
            print(f"  [TRUCO] {limpio}")

    def decir_robot(self, frase: str) -> None:
        if frase:
            self.ultima_frase_robot = frase
            self.registrar(f"🤖 Ottoman: \"{frase}\"")

    # ---------- ciclo de partida y mano ----------
    def cambiar_modo_ia(self, modo: str) -> Dict:
        """Cambia la arquitectura activa de la IA ('reactivo', 'utilidad', 'aprendizaje')."""
        info = self.ia.set_modo_agente(modo)
        self.registrar(f"🧠 Arquitectura IA cambiada a: {info['nombre']}")
        return self.estado_publico()

    def reiniciar_partido(self, meta_puntos: Optional[int] = None) -> Dict:
        if meta_puntos in (15, 30):
            self.meta_puntos = meta_puntos
        self.puntos_humano = 0
        self.puntos_robot = 0
        self.partido_terminado = False
        self.ganador_partido = None
        self.es_mano_humano = True
        self.estadisticas_partida = self._crear_estadisticas_vacias()
        self.nueva_mano(primera_vez=True)
        self.decir_robot(f"¡Partido nuevo a {self.meta_puntos} puntos! Vos sos mano.")
        return self.estado_publico()

    def nueva_mano(self, primera_vez: bool = False) -> Dict:
        """Mezcla y reparte 3 cartas para cada jugador y ejecuta el turno inicial si el robot es mano."""
        if self.partido_terminado:
            return self.reiniciar_partido(self.meta_puntos)

        if not primera_vez and self.mano_terminada:
            self.es_mano_humano = not self.es_mano_humano

        mazo = [{"valor": v, "palo": p, "nombre": tr.nombre_carta(v, p)} for p in tr.PALOS for v in tr.VALORES]
        self.rng.shuffle(mazo)

        self.cartas_humano = mazo[0:3]
        self.cartas_humano_disponibles = list(self.cartas_humano)
        self.cartas_robot = mazo[3:6]
        self.cartas_robot_disponibles = list(self.cartas_robot)

        self.ronda_actual = 1
        self.turno_carta = "humano" if self.es_mano_humano else "robot"
        self.mesa = []
        self.ganadores_ronda = []

        self.envido_resuelto = False
        self.puntos_envido_acumulados = 0
        self.ultimo_canto_envido = None
        self.resultado_envido_detalle = None

        self.apuesta_truco = "NADA"
        self.puntos_truco_actuales = 1
        self.quien_tiene_quiero_truco = "ambos"
        self.canto_pendiente = None

        self.bluffs_mano_actual = []
        self._ultimo_bluff_envido_mano = None
        self._ultimo_bluff_truco_mano = None
        self._humano_canto_envido_mano = False
        self._humano_canto_truco_mano = False
        self._mano_analizada = False
        self.ia.limpiar_traza_mano()

        self.mano_terminada = False
        self.ganador_mano = None

        mano_str = "HUMANO" if self.es_mano_humano else "OTTOMAN (Robot)"
        self.registrar(f"--- NUEVA MANO (Mano: {mano_str}) | Humano {self.puntos_humano} - Ottoman {self.puntos_robot} ---")

        # Animación de repartir las cartas en la mesa del simulador
        self._reaccion_robot("repartir")

        if self.es_mano_humano:
            self.decir_robot("Ya repartí. Sos mano, tirá una carta o cantá.")
        else:
            self.decir_robot("Soy mano esta vuelta, ahí reparto y juego yo...")
            self._avanzar_turno_robot_si_corresponde()

        return self.estado_publico()

    def _cerrar_aprendizaje_mano(self) -> None:
        if self._mano_analizada:
            return
        self._mano_analizada = True
        self.estadisticas_partida["manos_jugadas"] += 1
        self.ia.analizar_mano_humano(
            self.cartas_humano,
            humano_canto_envido=(self._humano_canto_envido_mano and self.resultado_envido_detalle is not None
                                 and "pts_humano" in self.resultado_envido_detalle),
            humano_canto_truco=self._humano_canto_truco_mano,
            cartas_truco_reveladas=[m["carta"] for m in self.mesa if m["quien"] == "humano"],
        )

    def _verificar_fin_partido(self) -> None:
        if self.partido_terminado:
            return
        if self.puntos_humano >= self.meta_puntos:
            self.partido_terminado = True
            self.ganador_partido = "humano"
            self.mano_terminada = True
            self._cerrar_aprendizaje_mano()
            self.ia.registrar_partida_finalizada(
                "humano",
                self.puntos_humano,
                self.puntos_robot,
                self.meta_puntos,
                self.estadisticas_partida,
            )
            self.decir_robot("¡Me ganaste el partido en buena ley! Te doy la mano, campeón.")
            self._reaccion_robot("fin_partido")
        elif self.puntos_robot >= self.meta_puntos:
            self.partido_terminado = True
            self.ganador_partido = "robot"
            self.mano_terminada = True
            self._cerrar_aprendizaje_mano()
            self.ia.registrar_partida_finalizada(
                "robot",
                self.puntos_humano,
                self.puntos_robot,
                self.meta_puntos,
                self.estadisticas_partida,
            )
            self.decir_robot("¡Gané el partido! Qué baile te pegué, hermano.")
            self._reaccion_robot("festejo")

    def puede_cantar_envido(self) -> bool:
        """El envido solo se puede cantar en 1ra ronda, antes de resolverse y sin truco ya querido."""
        return (
            not self.mano_terminada
            and self.ronda_actual == 1
            and not self.envido_resuelto
            and self.canto_pendiente is None
            and self.apuesta_truco == "NADA"
        )

    # ---------- jugadas de cartas ----------
    def jugar_carta_humano(self, carta: Dict) -> Dict:
        """El humano tira una carta a la mesa (por cámara, clic o voz)."""
        if self.mano_terminada:
            return {"error": "La mano ya terminó. Repartí una nueva mano.", "estado": self.estado_publico()}

        if self.canto_pendiente is not None:
            return {
                "error": f"Primero respondé al canto de {self.canto_pendiente['canto']} (Quiero / No Quiero).",
                "estado": self.estado_publico(),
            }

        if self.turno_carta != "humano":
            return {"error": "Todavía no es tu turno de tirar carta.", "estado": self.estado_publico()}

        if not isinstance(carta, dict):
            return {"error": "Esa carta no está en tu mano", "estado": self.estado_publico()}
        val_in = carta.get("valor")
        palo_in = carta.get("palo")
        if (type(val_in) is not int or val_in not in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)
                or palo_in not in ("Espada", "Basto", "Oro", "Copa")):
            return {"error": "Esa carta no está en tu mano", "estado": self.estado_publico()}
        carta_jugada = None
        for c in self.cartas_humano_disponibles:
            if int(c["valor"]) == val_in and str(c["palo"]).capitalize() == palo_in:
                carta_jugada = c
                break

        if carta_jugada is None:
            return {"error": "Esa carta no está en tu mano", "estado": self.estado_publico()}

        self.cartas_humano_disponibles.remove(carta_jugada)
        nombre_c = tr.nombre_carta(carta_jugada["valor"], carta_jugada["palo"])
        self.mesa.append(
            {
                "quien": "humano",
                "carta": carta_jugada,
                "nombre": nombre_c,
                "ronda": self.ronda_actual,
            }
        )
        self.registrar(f"👤 Humano jugó: {nombre_c} (Ronda {self.ronda_actual})")

        cartas_ronda = [m for m in self.mesa if m["ronda"] == self.ronda_actual]
        if len(cartas_ronda) == 2:
            self._evaluar_ronda()
        else:
            self.turno_carta = "robot"

        self._avanzar_turno_robot_si_corresponde()
        return {"carta_jugada": carta_jugada, "estado": self.estado_publico()}

    def _avanzar_turno_robot_si_corresponde(self) -> None:
        """Si es el turno del robot y no hay un canto pendiente del humano, el robot actúa."""
        if self.mano_terminada or self.canto_pendiente is not None or self.turno_carta != "robot":
            return
        if not self.cartas_robot_disponibles:
            return

        carta_humano_en_baza = next(
            (m["carta"] for m in self.mesa if m["ronda"] == self.ronda_actual and m["quien"] == "humano"),
            None,
        )

        # 1. En 1ra ronda, antes de tirar su 1ra carta, evaluar si cantar Envido
        if self.ronda_actual == 1 and not self.envido_resuelto and len(self.cartas_robot_disponibles) == 3:
            canto_e, frase_e = self.ia.decidir_iniciar_envido(
                self.cartas_robot,
                es_mano=(not self.es_mano_humano),
                puntos_humano=self.puntos_humano,
                puntos_robot=self.puntos_robot,
                meta_puntos=self.meta_puntos,
            )
            if canto_e:
                self._robot_canta_envido(
                    canto_e,
                    frase_e,
                    es_farol=self.ia.ultimo_canto_fue_farol,
                    detalle_farol=self.ia.ultimo_detalle_farol,
                )
                return

        # 2. Evaluar si cantar o subir el Truco antes de tirar la carta
        if self.quien_tiene_quiero_truco in ("ambos", "robot"):
            canto_t, frase_t = self.ia.decidir_iniciar_truco(
                self.cartas_robot_disponibles,
                self.apuesta_truco,
                self.ganadores_ronda,
                carta_rival_mesa=carta_humano_en_baza,
            )
            if canto_t:
                self._robot_canta_truco(
                    canto_t,
                    frase_t,
                    es_farol=self.ia.ultimo_canto_fue_farol,
                    detalle_farol=self.ia.ultimo_detalle_farol,
                )
                return

        # 3. Tirar la carta elegida por la IA
        self._robot_tira_carta(carta_humano_en_baza)

    def _robot_tira_carta(self, carta_humano_en_baza: Optional[Dict] = None) -> None:
        if not self.cartas_robot_disponibles or self.mano_terminada:
            return

        carta = self.ia.elegir_carta_a_jugar(
            self.cartas_robot_disponibles,
            ronda=self.ronda_actual,
            carta_rival_mesa=carta_humano_en_baza,
            ganadores_ronda=self.ganadores_ronda,
        )
        self.cartas_robot_disponibles.remove(carta)
        nombre_c = tr.nombre_carta(carta["valor"], carta["palo"])
        self.mesa.append(
            {
                "quien": "robot",
                "carta": carta,
                "nombre": nombre_c,
                "ronda": self.ronda_actual,
            }
        )
        self.decir_robot(f"Juego el {nombre_c}.")

        cartas_ronda = [m for m in self.mesa if m["ronda"] == self.ronda_actual]
        if len(cartas_ronda) == 2:
            self._evaluar_ronda()
            if not self.mano_terminada and self.turno_carta == "robot":
                self._avanzar_turno_robot_si_corresponde()
        else:
            self.turno_carta = "humano"

    def _evaluar_ronda(self) -> Optional[str]:
        """Evalúa quién ganó la baza actual y actualiza el turno de la siguiente."""
        cartas_ronda = [m for m in self.mesa if m["ronda"] == self.ronda_actual]
        if len(cartas_ronda) < 2:
            return None

        if self.ronda_actual == 1:
            self.envido_resuelto = True

        c_humano = next(m["carta"] for m in cartas_ronda if m["quien"] == "humano")
        c_robot = next(m["carta"] for m in cartas_ronda if m["quien"] == "robot")

        p_humano = tr.valor_carta_truco(c_humano["valor"], c_humano["palo"])
        p_robot = tr.valor_carta_truco(c_robot["valor"], c_robot["palo"])

        if p_humano > p_robot:
            ganador = "humano"
        elif p_robot > p_humano:
            ganador = "robot"
        else:
            ganador = "parda"

        self.ganadores_ronda.append(ganador)
        etiqueta = "EMPATE (Parda)" if ganador == "parda" else ganador.upper()
        self.registrar(f"Baza {self.ronda_actual}: {etiqueta}")

        self._verificar_ganador_mano()
        if not self.mano_terminada:
            self.ronda_actual += 1
            if ganador in ("humano", "robot"):
                self.turno_carta = ganador
            else:
                if self.ganadores_ronda[0] in ("humano", "robot"):
                    self.turno_carta = self.ganadores_ronda[0]
                else:
                    self.turno_carta = "humano" if self.es_mano_humano else "robot"

        return ganador

    def _verificar_ganador_mano(self) -> None:
        """Determina si alguno ya ganó la mano según las reglas oficiales del Truco."""
        g = self.ganadores_ronda
        ganador_final = None

        if g.count("humano") == 2:
            ganador_final = "humano"
        elif g.count("robot") == 2:
            ganador_final = "robot"
        elif len(g) >= 2:
            if g[0] == "parda" and g[1] in ("humano", "robot"):
                ganador_final = g[1]
            elif g[0] in ("humano", "robot") and g[1] == "parda":
                ganador_final = g[0]
            elif len(g) == 3:
                if g[2] in ("humano", "robot"):
                    ganador_final = g[2]
                elif g[0] in ("humano", "robot"):
                    ganador_final = g[0]
                else:
                    ganador_final = "humano" if self.es_mano_humano else "robot"

        if ganador_final is not None:
            self._adjudicar_mano(ganador_final, self.puntos_truco_actuales)

    def _adjudicar_mano(self, ganador: str, puntos: int, motivo: str = "") -> None:
        self.mano_terminada = True
        self.ganador_mano = ganador
        self.canto_pendiente = None
        sufijo = f" ({motivo})" if motivo else ""

        # Si hubo un farol de Truco en esta mano que fue querido (se jugó en mesa), registrar su desenlace
        if self._ultimo_bluff_truco_mano is not None and self._ultimo_bluff_truco_mano["resultado"] == "en_juego":
            if ganador == "robot":
                self._ultimo_bluff_truco_mano["resultado"] = "exitoso"
                self.estadisticas_partida["truco"]["bluffs_exitosos"] += 1
            else:
                self._ultimo_bluff_truco_mano["resultado"] = "descubierto"
                self.estadisticas_partida["truco"]["bluffs_descubiertos"] += 1

        if ganador == "robot":
            self.puntos_robot += puntos
            self.decir_robot(f"¡Esa mano es mía{sufijo}! Sumo {puntos} {'punto' if puntos == 1 else 'puntos'}.")
            self._reaccion_robot("festejo")
        else:
            self.puntos_humano += puntos
            self.decir_robot(f"Te llevaste la mano{sufijo} (+{puntos} pts). Bien jugado.")

        self._cerrar_aprendizaje_mano()
        self._verificar_fin_partido()

    # ---------- Envido (Humano <-> Robot) ----------
    def _valores_envido(self, canto: str) -> Tuple[int, int]:
        """Devuelve (puntos_si_quiere, puntos_si_no_quiere) al sumar `canto`."""
        no_quiero = max(1, self.puntos_envido_acumulados)
        if canto == "ENVIDO":
            return self.puntos_envido_acumulados + 2, no_quiero
        if canto == "ENVIDO_ENVIDO":
            return self.puntos_envido_acumulados + 2, max(2, no_quiero)
        if canto == "REAL_ENVIDO":
            return self.puntos_envido_acumulados + 3, no_quiero
        if canto == "FALTA_ENVIDO":
            pts_f = tr.puntos_falta_envido(self.puntos_humano, self.puntos_robot, self.meta_puntos)
            return pts_f, no_quiero
        return 2, 1

    def _opciones_subir_envido(self, canto_actual: str) -> List[str]:
        if canto_actual == "ENVIDO":
            return ["ENVIDO_ENVIDO", "REAL_ENVIDO", "FALTA_ENVIDO"]
        if canto_actual == "ENVIDO_ENVIDO":
            return ["REAL_ENVIDO", "FALTA_ENVIDO"]
        if canto_actual == "REAL_ENVIDO":
            return ["FALTA_ENVIDO"]
        return []

    def _robot_canta_envido(
        self,
        canto: str,
        frase: str,
        es_farol: bool = False,
        detalle_farol: str = "",
    ) -> None:
        pts_q, pts_nq = self._valores_envido(canto)
        self.ultimo_canto_envido = canto
        self.estadisticas_partida["envido"]["cantados_robot"] += 1

        if es_farol:
            self.estadisticas_partida["envido"]["bluffs_intentados"] += 1
            tantos_r = tr.calcular_envido(self.cartas_robot)
            evento_b = {
                "tipo": "ENVIDO",
                "canto": canto,
                "detalle": detalle_farol or f"Cantó {canto} con solo {tantos_r} de envido",
                "resultado": "pendiente",
            }
            self.bluffs_mano_actual.append(evento_b)
            self._ultimo_bluff_envido_mano = evento_b

        self.decir_robot(frase or f"¡{canto.replace('_', ' ')}!")
        if canto == "FALTA_ENVIDO":
            self._reaccion_robot("desafio")
        self.canto_pendiente = {
            "tipo": "ENVIDO",
            "canto": canto,
            "quien_canto": "robot",
            "es_farol": es_farol,
            "puntos_quiero": pts_q,
            "puntos_no_quiero": pts_nq,
            "opciones": ["QUIERO", "NO_QUIERO"] + self._opciones_subir_envido(canto),
        }

    def cantar_envido_humano(self, canto: str) -> Dict:
        """El humano inicia o sube el canto de Envido."""
        canto = canto.upper().strip()
        if self.mano_terminada:
            return {"error": "La mano ya terminó.", "estado": self.estado_publico()}

        if self.canto_pendiente is not None:
            if self.canto_pendiente["tipo"] == "TRUCO" and self.ronda_actual == 1 and not self.envido_resuelto:
                self.apuesta_truco = "NADA"
                self.puntos_truco_actuales = 1
                self.quien_tiene_quiero_truco = "ambos"
                self.canto_pendiente = None
            elif self.canto_pendiente["tipo"] == "ENVIDO":
                return self.responder_canto_humano(canto)
            else:
                return {"error": "Primero respondé al canto pendiente.", "estado": self.estado_publico()}

        if self.ronda_actual != 1 or self.envido_resuelto:
            return {"error": "El Envido solo se puede cantar en primera ronda.", "estado": self.estado_publico()}

        self._humano_canto_envido_mano = True
        pts_q, pts_nq = self._valores_envido(canto)
        self.ultimo_canto_envido = canto
        self.registrar(f"👤 Humano cantó: {canto.replace('_', ' ')}")

        resp, frase = self.ia.responder_envido(
            self.cartas_robot,
            canto,
            es_mano=(not self.es_mano_humano),
            puntos_acumulados_no_quiero=pts_nq,
            puntos_humano=self.puntos_humano,
            puntos_robot=self.puntos_robot,
            meta_puntos=self.meta_puntos,
        )

        if resp in ("ENVIDO_ENVIDO", "REAL_ENVIDO", "FALTA_ENVIDO"):
            self.puntos_envido_acumulados = pts_q
            self._robot_canta_envido(
                resp,
                frase,
                es_farol=self.ia.ultimo_canto_fue_farol,
                detalle_farol=self.ia.ultimo_detalle_farol,
            )
            return {"respuesta": resp, "frase": frase, "estado": self.estado_publico()}

        if resp == "QUIERO":
            self.puntos_envido_acumulados = pts_q
            res = self._resolver_tantos_envido(pts_q, frase_inicial=frase)
            self._avanzar_turno_robot_si_corresponde()
            return res

        # NO_QUIERO del robot
        self.envido_resuelto = True
        self.puntos_humano += pts_nq
        self.decir_robot(frase or "No quiero, son buenas.")
        self._reaccion_robot("mazo")
        self.resultado_envido_detalle = {
            "ganador": "humano",
            "puntos_ganados": pts_nq,
            "motivo": "No querido por Ottoman",
        }
        self._verificar_fin_partido()
        self._avanzar_turno_robot_si_corresponde()
        return {"respuesta": "NO_QUIERO", "frase": frase, "ganador": "humano", "estado": self.estado_publico()}

    def _resolver_tantos_envido(self, puntos_en_juego: int, frase_inicial: str = "") -> Dict:
        self.envido_resuelto = True
        self.canto_pendiente = None
        pts_robot = tr.calcular_envido(self.cartas_robot)
        pts_humano = tr.calcular_envido(self.cartas_humano)

        if pts_humano > pts_robot or (pts_humano == pts_robot and self.es_mano_humano):
            ganador_e = "humano"
            self.puntos_humano += puntos_en_juego
            msg = f"{frase_inicial} Tenés {pts_humano} contra mis {pts_robot}, ¡son buenas! (+{puntos_en_juego} pts)".strip()
            self.decir_robot(msg)
        else:
            ganador_e = "robot"
            self.puntos_robot += puntos_en_juego
            msg = f"{frase_inicial} ¡Canto {pts_robot} y matan tus {pts_humano}! (+{puntos_en_juego} pts para mí)".strip()
            self.decir_robot(msg)
            self._reaccion_robot("festejo")

        # Si el robot había hecho un farol de Envido y se llegó a mostrar los tantos:
        if self._ultimo_bluff_envido_mano is not None and self._ultimo_bluff_envido_mano["resultado"] == "pendiente":
            if ganador_e == "robot":
                self._ultimo_bluff_envido_mano["resultado"] = "exitoso"
                self.estadisticas_partida["envido"]["bluffs_exitosos"] += 1
            else:
                self._ultimo_bluff_envido_mano["resultado"] = "descubierto"
                self.estadisticas_partida["envido"]["bluffs_descubiertos"] += 1

        self.resultado_envido_detalle = {
            "ganador": ganador_e,
            "pts_humano": pts_humano,
            "pts_robot": pts_robot,
            "puntos_ganados": puntos_en_juego,
        }
        self._verificar_fin_partido()
        return {
            "respuesta": "QUIERO",
            "ganador": ganador_e,
            "pts_humano": pts_humano,
            "pts_robot": pts_robot,
            "puntos_ganados": puntos_en_juego,
            "estado": self.estado_publico(),
        }

    # ---------- Truco / Retruco / Vale Cuatro (Humano <-> Robot) ----------
    def _siguiente_nivel_truco(self) -> Optional[str]:
        orden = {"NADA": "TRUCO", "TRUCO": "RETRUCO", "RETRUCO": "VALE_CUATRO"}
        return orden.get(self.apuesta_truco)

    def _puntos_nivel_truco(self, nivel: str) -> Tuple[int, int]:
        mapa = {
            "TRUCO": (2, 1),
            "RETRUCO": (3, 2),
            "VALE_CUATRO": (4, 3),
        }
        return mapa.get(nivel, (1, 1))

    def _robot_canta_truco(
        self,
        nivel: str,
        frase: str,
        es_farol: bool = False,
        detalle_farol: str = "",
    ) -> None:
        pts_q, pts_nq = self._puntos_nivel_truco(nivel)
        self.estadisticas_partida["truco"]["cantados_robot"] += 1

        if es_farol:
            self.estadisticas_partida["truco"]["bluffs_intentados"] += 1
            evento_b = {
                "tipo": "TRUCO",
                "canto": nivel,
                "detalle": detalle_farol or f"Cantó {nivel} de farol con cartas bajas",
                "resultado": "pendiente",
            }
            self.bluffs_mano_actual.append(evento_b)
            self._ultimo_bluff_truco_mano = evento_b

        self.decir_robot(frase or f"¡{nivel.replace('_', ' ')}!")
        self._reaccion_robot("desafio")
        opciones = ["QUIERO", "NO_QUIERO"]
        if nivel == "TRUCO":
            opciones.append("RETRUCO")
        elif nivel == "RETRUCO":
            opciones.append("VALE_CUATRO")
        if self.ronda_actual == 1 and not self.envido_resuelto:
            opciones.extend(["ENVIDO", "REAL_ENVIDO", "FALTA_ENVIDO"])

        self.canto_pendiente = {
            "tipo": "TRUCO",
            "canto": nivel,
            "quien_canto": "robot",
            "es_farol": es_farol,
            "puntos_quiero": pts_q,
            "puntos_no_quiero": pts_nq,
            "opciones": opciones,
        }

    def cantar_truco_humano(self, nivel_pedido: Optional[str] = None) -> Dict:
        """El humano canta Truco, Retruco o Vale Cuatro."""
        if self.mano_terminada:
            return {"error": "La mano ya terminó.", "estado": self.estado_publico()}

        if self.canto_pendiente is not None:
            if self.canto_pendiente["tipo"] == "TRUCO":
                sig = nivel_pedido or self._siguiente_nivel_truco()
                if sig:
                    return self.responder_canto_humano(sig)
            return {"error": "Primero respondé al canto pendiente.", "estado": self.estado_publico()}

        if self.quien_tiene_quiero_truco not in ("ambos", "humano"):
            return {"error": "No tenés el 'quiero' para subir la apuesta ahora.", "estado": self.estado_publico()}

        sig_nivel = self._siguiente_nivel_truco()
        if not sig_nivel:
            return {"error": "Ya se está jugando el Vale Cuatro.", "estado": self.estado_publico()}

        nivel = sig_nivel
        self._humano_canto_truco_mano = True
        pts_q, pts_nq = self._puntos_nivel_truco(nivel)
        self.registrar(f"👤 Humano cantó: {nivel.replace('_', ' ')}")

        carta_humano_en_baza = next(
            (m["carta"] for m in self.mesa if m["ronda"] == self.ronda_actual and m["quien"] == "humano"),
            None,
        )

        if nivel == "TRUCO" and self.ronda_actual == 1 and not self.envido_resuelto:
            canto_e, _ = self.ia.decidir_iniciar_envido(
                self.cartas_robot,
                es_mano=(not self.es_mano_humano),
                puntos_humano=self.puntos_humano,
                puntos_robot=self.puntos_robot,
                meta_puntos=self.meta_puntos,
            )
            if canto_e:
                self._robot_canta_envido(
                    canto_e,
                    f"¡El envido está primero! ¡{canto_e.replace('_', ' ')}!",
                    es_farol=self.ia.ultimo_canto_fue_farol,
                    detalle_farol=self.ia.ultimo_detalle_farol,
                )
                return {"respuesta": canto_e, "estado": self.estado_publico()}

        resp, frase = self.ia.responder_canto_truco(
            self.cartas_robot_disponibles,
            nivel,
            self.ganadores_ronda,
            carta_rival_mesa=carta_humano_en_baza,
            puntos_humano=self.puntos_humano,
            meta_puntos=self.meta_puntos,
        )

        if resp in ("RETRUCO", "VALE_CUATRO"):
            self.apuesta_truco = nivel
            self.puntos_truco_actuales = pts_q
            self.quien_tiene_quiero_truco = "humano"
            self._robot_canta_truco(
                resp,
                frase,
                es_farol=self.ia.ultimo_canto_fue_farol,
                detalle_farol=self.ia.ultimo_detalle_farol,
            )
            return {"respuesta": resp, "frase": frase, "estado": self.estado_publico()}

        if resp == "QUIERO":
            self.apuesta_truco = nivel
            self.puntos_truco_actuales = pts_q
            self.quien_tiene_quiero_truco = "robot"
            self.decir_robot(frase or "¡Quiero!")
            self._avanzar_turno_robot_si_corresponde()
            return {"respuesta": "QUIERO", "frase": frase, "estado": self.estado_publico()}

        # NO_QUIERO -> El robot se va al mazo
        self.decir_robot(frase or "No quiero, me voy al mazo.")
        self._reaccion_robot("mazo")
        self._adjudicar_mano("humano", pts_nq, motivo=f"{nivel} no querido")
        return {"respuesta": "NO_QUIERO", "frase": frase, "estado": self.estado_publico()}

    def responder_canto_humano(self, respuesta: str) -> Dict:
        """El humano responde ('QUIERO', 'NO_QUIERO' o retruca/sube) al canto pendiente del robot."""
        if self.canto_pendiente is None:
            return {"error": "No hay ningún canto pendiente para responder.", "estado": self.estado_publico()}

        resp = respuesta.upper().strip()
        pendiente = self.canto_pendiente
        tipo = pendiente["tipo"]
        canto_robot = pendiente["canto"]
        era_farol = bool(pendiente.get("es_farol", False))

        if tipo == "ENVIDO":
            if resp == "QUIERO":
                self.ia.registrar_respuesta_humano_a_canto("ENVIDO", quiso=True, era_farol_robot=era_farol)
                pts_q = pendiente["puntos_quiero"]
                self.puntos_envido_acumulados = pts_q
                res = self._resolver_tantos_envido(pts_q)
                self._avanzar_turno_robot_si_corresponde()
                return res
            if resp == "NO_QUIERO":
                self.ia.registrar_respuesta_humano_a_canto("ENVIDO", quiso=False, era_farol_robot=era_farol)
                if self._ultimo_bluff_envido_mano is not None and self._ultimo_bluff_envido_mano["resultado"] == "pendiente":
                    self._ultimo_bluff_envido_mano["resultado"] = "exitoso"
                    self.estadisticas_partida["envido"]["bluffs_exitosos"] += 1
                pts_nq = pendiente["puntos_no_quiero"]
                self.envido_resuelto = True
                self.canto_pendiente = None
                self.puntos_robot += pts_nq
                self.decir_robot(f"¡Achicaste! Sumo {pts_nq} {'punto' if pts_nq == 1 else 'puntos'} del envido.")
                self._reaccion_robot("festejo")
                self._verificar_fin_partido()
                self._avanzar_turno_robot_si_corresponde()
                return {"respuesta": "NO_QUIERO", "ganador": "robot", "estado": self.estado_publico()}
            if resp in ("ENVIDO", "ENVIDO_ENVIDO", "REAL_ENVIDO", "FALTA_ENVIDO"):
                self.ia.registrar_respuesta_humano_a_canto("ENVIDO", quiso=True, era_farol_robot=era_farol)
                if self._ultimo_bluff_envido_mano is not None and self._ultimo_bluff_envido_mano["resultado"] == "pendiente":
                    self._ultimo_bluff_envido_mano["resultado"] = "descubierto"
                    self.estadisticas_partida["envido"]["bluffs_descubiertos"] += 1
                if resp == "ENVIDO" and canto_robot == "ENVIDO":
                    resp = "ENVIDO_ENVIDO"
                self.puntos_envido_acumulados = pendiente["puntos_quiero"]
                self.canto_pendiente = None
                return self.cantar_envido_humano(resp)

        elif tipo == "TRUCO":
            if resp in ("ENVIDO", "REAL_ENVIDO", "FALTA_ENVIDO") and self.ronda_actual == 1 and not self.envido_resuelto:
                return self.cantar_envido_humano(resp)

            if resp == "QUIERO":
                self.ia.registrar_respuesta_humano_a_canto("TRUCO", quiso=True, era_farol_robot=era_farol)
                if self._ultimo_bluff_truco_mano is not None and self._ultimo_bluff_truco_mano["resultado"] == "pendiente":
                    self._ultimo_bluff_truco_mano["resultado"] = "en_juego"
                self.apuesta_truco = canto_robot
                self.puntos_truco_actuales = pendiente["puntos_quiero"]
                self.quien_tiene_quiero_truco = "humano"
                self.canto_pendiente = None
                self.decir_robot(f"¡Así me gusta! Juguemos por el {canto_robot.replace('_', ' ')}.")
                self._avanzar_turno_robot_si_corresponde()
                return {"respuesta": "QUIERO", "estado": self.estado_publico()}

            if resp == "NO_QUIERO":
                self.ia.registrar_respuesta_humano_a_canto("TRUCO", quiso=False, era_farol_robot=era_farol)
                if self._ultimo_bluff_truco_mano is not None and self._ultimo_bluff_truco_mano["resultado"] == "pendiente":
                    self._ultimo_bluff_truco_mano["resultado"] = "exitoso"
                    self.estadisticas_partida["truco"]["bluffs_exitosos"] += 1
                pts_nq = pendiente["puntos_no_quiero"]
                self.canto_pendiente = None
                self._adjudicar_mano("robot", pts_nq, motivo=f"{canto_robot} no querido")
                return {"respuesta": "NO_QUIERO", "ganador": "robot", "estado": self.estado_publico()}

            if resp in ("RETRUCO", "VALE_CUATRO"):
                self.ia.registrar_respuesta_humano_a_canto("TRUCO", quiso=True, era_farol_robot=era_farol)
                if self._ultimo_bluff_truco_mano is not None and self._ultimo_bluff_truco_mano["resultado"] == "pendiente":
                    self._ultimo_bluff_truco_mano["resultado"] = "en_juego"
                self.apuesta_truco = canto_robot
                self.puntos_truco_actuales = pendiente["puntos_quiero"]
                self.quien_tiene_quiero_truco = "humano"
                self.canto_pendiente = None
                return self.cantar_truco_humano(resp)

        return {"error": f"Respuesta '{resp}' no válida en este momento.", "estado": self.estado_publico()}

    def irse_al_mazo_humano(self) -> Dict:
        """El humano decide irse al mazo."""
        if self.mano_terminada:
            return {"error": "La mano ya terminó.", "estado": self.estado_publico()}

        if self.canto_pendiente is not None and self.canto_pendiente["tipo"] == "TRUCO":
            return self.responder_canto_humano("NO_QUIERO")

        pts = self.puntos_truco_actuales
        if self.ronda_actual == 1 and not self.envido_resuelto and self.apuesta_truco == "NADA" and len(self.mesa) == 0:
            pts = 2

        self._adjudicar_mano("robot", pts, motivo="Humano se fue al mazo")
        return {"mensaje": "Te fuiste al mazo", "estado": self.estado_publico()}

    # ---------- comando por voz en la mesa de Truco ----------
    def procesar_voz_truco(self, texto: str) -> Dict:
        """Interpreta un canto o jugada dicha por micrófono durante la partida de Truco."""
        t = (texto or "").lower().strip()
        if not t:
            return {"error": "Texto vacío", "estado": self.estado_publico()}

        self.registrar(f"🎤 Voz recibida en mesa: \"{texto}\"")

        if re.search(r"\b(?:nueva\s+mano|repart[ií]|otra\s+mano)\b", t):
            return {"accion_voz": "NUEVA_MANO", "estado": self.nueva_mano()}

        if re.search(r"\b(?:no\s+quiero|son\s+buenas|paso)\b", t):
            if self.canto_pendiente is not None:
                return self.responder_canto_humano("NO_QUIERO")
            return self.irse_al_mazo_humano()

        if re.search(r"\b(?:mazo|me\s+voy)\b", t):
            return self.irse_al_mazo_humano()

        if re.search(r"\b(?:vale\s+cuatro|vale\s+4)\b", t):
            if self.canto_pendiente is not None:
                return self.responder_canto_humano("VALE_CUATRO")
            return self.cantar_truco_humano("VALE_CUATRO")

        if re.search(r"\b(?:re\s*truco|retruco)\b", t):
            if self.canto_pendiente is not None:
                return self.responder_canto_humano("RETRUCO")
            return self.cantar_truco_humano("RETRUCO")

        if re.search(r"\bfalta\s+envido\b", t):
            if self.canto_pendiente is not None:
                return self.responder_canto_humano("FALTA_ENVIDO")
            return self.cantar_envido_humano("FALTA_ENVIDO")

        if re.search(r"\breal\s+envido\b", t):
            if self.canto_pendiente is not None:
                return self.responder_canto_humano("REAL_ENVIDO")
            return self.cantar_envido_humano("REAL_ENVIDO")

        if re.search(r"\benvido\b", t):
            if self.canto_pendiente is not None:
                return self.responder_canto_humano("ENVIDO_ENVIDO")
            return self.cantar_envido_humano("ENVIDO")

        if re.search(r"\btruco\b", t):
            return self.cantar_truco_humano("TRUCO")

        if re.search(r"\bquiero\b", t):
            if self.canto_pendiente is not None:
                return self.responder_canto_humano("QUIERO")
            return {"error": "Dijiste 'Quiero', pero no hay ningún canto pendiente.", "estado": self.estado_publico()}

        carta_voz = tr.buscar_carta_en_texto(t, self.cartas_humano_disponibles)
        if carta_voz is not None:
            return self.jugar_carta_humano(carta_voz)

        if re.search(r"\b(?:carta|juego|jugar|tiro|tirar)\b", t):
            return self.jugar_carta_humano(None)

        return {
            "error": f"No reconocí una jugada de Truco en: \"{texto}\". Probá diciendo 'Envido', 'Truco', 'Quiero', 'Al mazo' o el nombre de tu carta.",
            "estado": self.estado_publico(),
        }

    # ---------- serialización de estado para la UI ----------
    def estado_publico(self) -> Dict:
        siguiente_truco = self._siguiente_nivel_truco() if self.quien_tiene_quiero_truco in ("ambos", "humano") else None

        # Al terminar la mano, revelar las 3 cartas de Ottoman y si hizo farol
        cartas_robot_reveladas = None
        tantos_robot_revelados = None
        if self.mano_terminada:
            jugadas_set = {
                (m["carta"]["valor"], m["carta"]["palo"])
                for m in self.mesa
                if m["quien"] == "robot"
            }
            cartas_robot_reveladas = [
                {
                    "valor": c["valor"],
                    "palo": c["palo"],
                    "nombre": c.get("nombre", tr.nombre_carta(c["valor"], c["palo"])),
                    "jugada": (c["valor"], c["palo"]) in jugadas_set,
                }
                for c in self.cartas_robot
            ]
            tantos_robot_revelados = tr.calcular_envido(self.cartas_robot)

        return {
            "tanteador": {
                "humano": self.puntos_humano,
                "robot": self.puntos_robot,
                "meta": self.meta_puntos,
            },
            "partido_terminado": self.partido_terminado,
            "ganador_partido": self.ganador_partido,
            "es_mano": "humano" if self.es_mano_humano else "robot",
            "ronda": self.ronda_actual,
            "turno_carta": self.turno_carta,
            "cartas_robot_restantes": len(self.cartas_robot_disponibles),
            "cartas_robot_reveladas": cartas_robot_reveladas,
            "tantos_robot": tantos_robot_revelados,
            "bluffs_mano_actual": self.bluffs_mano_actual if self.mano_terminada else [],
            "estadisticas_partida": self.estadisticas_partida,
            "aprendizaje_ia": self.ia.resumen_aprendizaje(),
            "cartas_humano": self.cartas_humano,
            "cartas_humano_disponibles": self.cartas_humano_disponibles,
            "tantos_humano": tr.calcular_envido(self.cartas_humano),
            "envido_disponible": self.puede_cantar_envido(),
            "envido_resuelto": self.envido_resuelto,
            "resultado_envido": self.resultado_envido_detalle,
            "apuesta_truco": self.apuesta_truco,
            "puntos_truco_actuales": self.puntos_truco_actuales,
            "siguiente_truco_humano": siguiente_truco,
            "canto_pendiente": self.canto_pendiente,
            "mesa": self.mesa,
            "ganadores_ronda": self.ganadores_ronda,
            "mano_terminada": self.mano_terminada,
            "ganador_mano": self.ganador_mano,
            "ultima_frase_robot": self.ultima_frase_robot,
            "log": list(reversed(self.log_mensajes[-25:])),
        }
