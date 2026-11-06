"""Motor de Inteligencia Artificial para el Agente Robot Jugador de Truco (Ottoman G1).

Utiliza las probabilidades precomputadas del dataset T1K22.json (474.000 manos)
combinadas con conciencia de tanteador, una cuota dinámica de farol (bluff) y
aprendizaje adaptativo persistente a lo largo de las partidas contra humanos:
- Cantar y responder Envido / Real Envido / Falta Envido
- Cantar y responder Truco / Retruco / Vale Cuatro
- Selección estratégica de carta según la baza (1ra, 2da, 3ra) y resultado previo
- Detección y registro explícito de mentiras/faroles propios y del rival humano
"""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import truco_reglas as tr


class AgenteTrucoIA:
    MODOS_DISPONIBLES = {
        "reactivo": {
            "codigo": "reactivo",
            "nombre": "Agente Reactivo Simple",
            "subtitulo": "Reglas fijas condición-acción (sin probabilidades 474k, sin farol ni aprendizaje)",
        },
        "utilidad": {
            "codigo": "utilidad",
            "nombre": "Agente Basado en Utilidad (474k manos)",
            "subtitulo": "Maximiza utilidad esperada con el corpus de 474.000 manos y tanteador (estático)",
        },
        "aprendizaje": {
            "codigo": "aprendizaje",
            "nombre": "Agente Híbrido con Aprendizaje Adaptativo",
            "subtitulo": "Utilidad esperada (474k) + cuota dinámica de farol + modelado online del rival humano",
        },
    }

    def __init__(
        self,
        stats_path: Optional[str] = None,
        historial_path: Optional[str] = None,
        prob_farol: float = 0.14,
        modo_agente: str = "aprendizaje",
        rng: random.Random | None = None,
    ):
        self.rng = rng if rng is not None else random.Random()
        base_dir = Path(__file__).resolve().parent
        if stats_path is None:
            stats_path = str(base_dir / "truco_stats.json")
        if historial_path is None:
            historial_path = str(base_dir / "historial_partidas.json")

        self.historial_path = Path(historial_path)

        with open(stats_path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)

        self.modo_agente = modo_agente if modo_agente in self.MODOS_DISPONIBLES else "aprendizaje"
        self.prob_farol = prob_farol
        self.prob_farol_envido = prob_farol
        self.prob_farol_truco = prob_farol
        self.umbral_quiero_envido_offset = 0.0
        self.umbral_quiero_truco_offset = 0.0

        # Metadatos del último canto del robot para auditar bluffs
        self.ultimo_canto_fue_farol: bool = False
        self.ultimo_detalle_farol: str = ""

        # Explicabilidad (XAI): trazas del pensamiento interno del agente
        self.ultimo_razonamiento: Optional[Dict] = None
        self.traza_razonamiento: List[Dict] = []

        # Estado persistente de evolución y partidas
        self.datos_historial: Dict = self._cargar_historial_inicial(prob_farol)
        self._sincronizar_atributos()

    def set_modo_agente(self, modo: str) -> Dict:
        """Cambia la arquitectura activa del agente ('reactivo', 'utilidad', 'aprendizaje')."""
        modo_limpio = str(modo).strip().lower()
        if modo_limpio in self.MODOS_DISPONIBLES:
            self.modo_agente = modo_limpio
        return self.MODOS_DISPONIBLES[self.modo_agente]

    def limpiar_traza_mano(self) -> None:
        """Reinicia la traza de razonamiento XAI al comenzar una nueva mano."""
        self.traza_razonamiento = []
        self.ultimo_razonamiento = None

    def _registrar_razonamiento(
        self,
        fase: str,
        decision: str,
        metrica: str,
        umbral: str,
        farol_info: str,
        explicacion: str,
        es_accion_principal: bool = True,
    ) -> None:
        info_modo = self.MODOS_DISPONIBLES.get(self.modo_agente, self.MODOS_DISPONIBLES["aprendizaje"])
        item = {
            "hora": datetime.datetime.now().strftime("%H:%M:%S"),
            "modo_codigo": self.modo_agente,
            "arquitectura": info_modo["nombre"],
            "fase": fase,
            "decision": decision,
            "metrica": metrica,
            "umbral": umbral,
            "farol_info": farol_info,
            "explicacion": explicacion,
        }
        self.traza_razonamiento.append(item)
        if len(self.traza_razonamiento) > 12:
            self.traza_razonamiento = self.traza_razonamiento[-12:]
        if es_accion_principal or self.ultimo_razonamiento is None:
            self.ultimo_razonamiento = item

    def _cargar_historial_inicial(self, prob_base: float) -> Dict:
        estructura_defecto = {
            "parametros_ia": {
                "prob_farol_envido": prob_base,
                "prob_farol_truco": prob_base,
                "umbral_quiero_envido_offset": 0.0,
                "umbral_quiero_truco_offset": 0.0,
                "partidas_jugadas": 0,
                "victorias_robot": 0,
                "victorias_humano": 0,
                "manos_totales": 0,
                "bluffs_humano_detectados": 0,
                "shows_humano_totales": 0,
                "veces_humano_quiso_envido": 0,
                "veces_humano_no_quiso_envido": 0,
                "veces_humano_quiso_truco": 0,
                "veces_humano_no_quiso_truco": 0,
            },
            "partidas": [],
        }
        if self.historial_path.is_file():
            try:
                with open(self.historial_path, "r", encoding="utf-8") as f:
                    cargado = json.load(f)
                if isinstance(cargado, dict) and "parametros_ia" in cargado:
                    for k, v in estructura_defecto["parametros_ia"].items():
                        cargado["parametros_ia"].setdefault(k, v)
                    cargado.setdefault("partidas", [])
                    return cargado
            except Exception:
                pass
        return estructura_defecto

    def _sincronizar_atributos(self) -> None:
        p = self.datos_historial["parametros_ia"]
        self.prob_farol_envido = float(p.get("prob_farol_envido", 0.14))
        self.prob_farol_truco = float(p.get("prob_farol_truco", 0.14))
        self.prob_farol = round((self.prob_farol_envido + self.prob_farol_truco) / 2.0, 3)
        self.umbral_quiero_envido_offset = float(p.get("umbral_quiero_envido_offset", 0.0))
        self.umbral_quiero_truco_offset = float(p.get("umbral_quiero_truco_offset", 0.0))

    def guardar_historial(self) -> None:
        p = self.datos_historial["parametros_ia"]
        p["prob_farol_envido"] = round(self.prob_farol_envido, 3)
        p["prob_farol_truco"] = round(self.prob_farol_truco, 3)
        p["umbral_quiero_envido_offset"] = round(self.umbral_quiero_envido_offset, 3)
        p["umbral_quiero_truco_offset"] = round(self.umbral_quiero_truco_offset, 3)
        try:
            with open(self.historial_path, "w", encoding="utf-8") as f:
                json.dump(self.datos_historial, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def resumen_aprendizaje(self) -> Dict:
        """Devuelve el estado evolutivo actual de la IA y su lectura del jugador humano."""
        p = self.datos_historial["parametros_ia"]
        shows = max(1, int(p.get("shows_humano_totales", 0)))
        bluffs_h = int(p.get("bluffs_humano_detectados", 0))
        tasa_bluff_h = round((bluffs_h / shows) * 100.0, 1) if p.get("shows_humano_totales", 0) > 0 else 0.0

        total_env_resp = int(p.get("veces_humano_quiso_envido", 0)) + int(p.get("veces_humano_no_quiso_envido", 0))
        tasa_fold_env = (
            round((int(p.get("veces_humano_no_quiso_envido", 0)) / total_env_resp) * 100.0, 1)
            if total_env_resp > 0
            else 50.0
        )

        total_tru_resp = int(p.get("veces_humano_quiso_truco", 0)) + int(p.get("veces_humano_no_quiso_truco", 0))
        tasa_fold_tru = (
            round((int(p.get("veces_humano_no_quiso_truco", 0)) / total_tru_resp) * 100.0, 1)
            if total_tru_resp > 0
            else 50.0
        )

        if tasa_bluff_h >= 28.0:
            perfil_humano = "Agresivo / Mentiroso (Ottoman te paga más seguido)"
        elif tasa_fold_env >= 60.0 or tasa_fold_tru >= 60.0:
            perfil_humano = "Cauteloso / Conservador (Ottoman te presiona con más faroles)"
        else:
            perfil_humano = "Equilibrado (Ottoman balancea estadística y engaño)"

        info_modo = self.MODOS_DISPONIBLES.get(self.modo_agente, self.MODOS_DISPONIBLES["aprendizaje"])
        return {
            "modo_agente": self.modo_agente,
            "arquitectura_nombre": info_modo["nombre"],
            "arquitectura_subtitulo": info_modo["subtitulo"],
            "modos_disponibles": list(self.MODOS_DISPONIBLES.values()),
            "prob_farol_envido_pct": round(self.prob_farol_envido * 100.0, 1),
            "prob_farol_truco_pct": round(self.prob_farol_truco * 100.0, 1),
            "umbral_envido_offset": round(self.umbral_quiero_envido_offset, 3),
            "umbral_truco_offset": round(self.umbral_quiero_truco_offset, 3),
            "partidas_jugadas": int(p.get("partidas_jugadas", 0)),
            "victorias_robot": int(p.get("victorias_robot", 0)),
            "victorias_humano": int(p.get("victorias_humano", 0)),
            "manos_totales": int(p.get("manos_totales", 0)),
            "bluffs_humano_detectados": bluffs_h,
            "tasa_bluff_humano_pct": tasa_bluff_h,
            "tasa_fold_envido_humano_pct": tasa_fold_env,
            "tasa_fold_truco_humano_pct": tasa_fold_tru,
            "perfil_humano_detectado": perfil_humano,
            "ultimo_razonamiento": self.ultimo_razonamiento,
            "traza_razonamiento": list(reversed(self.traza_razonamiento[-8:])),
            "partidas": list(reversed(self.datos_historial.get("partidas", [])[-30:])),
        }

    def registrar_respuesta_humano_a_canto(self, tipo: str, quiso: bool, era_farol_robot: bool) -> None:
        """Ajusta en línea las probabilidades de farol de Ottoman según si el humano quiso o no."""
        p = self.datos_historial["parametros_ia"]
        if tipo == "ENVIDO":
            if quiso:
                p["veces_humano_quiso_envido"] = int(p.get("veces_humano_quiso_envido", 0)) + 1
                if era_farol_robot:
                    self.prob_farol_envido = max(0.04, self.prob_farol_envido - 0.022)
            else:
                p["veces_humano_no_quiso_envido"] = int(p.get("veces_humano_no_quiso_envido", 0)) + 1
                self.prob_farol_envido = min(0.38, self.prob_farol_envido + 0.016)
        elif tipo == "TRUCO":
            if quiso:
                p["veces_humano_quiso_truco"] = int(p.get("veces_humano_quiso_truco", 0)) + 1
                if era_farol_robot:
                    self.prob_farol_truco = max(0.04, self.prob_farol_truco - 0.022)
            else:
                p["veces_humano_no_quiso_truco"] = int(p.get("veces_humano_no_quiso_truco", 0)) + 1
                self.prob_farol_truco = min(0.38, self.prob_farol_truco + 0.016)

        self.prob_farol = round((self.prob_farol_envido + self.prob_farol_truco) / 2.0, 3)
        self.guardar_historial()

    def analizar_mano_humano(
        self,
        cartas_humano: List[Dict],
        humano_canto_envido: bool,
        humano_canto_truco: bool,
    ) -> None:
        """Al terminar una mano, analiza si el humano mintió para adaptar los umbrales de aceptación."""
        p = self.datos_historial["parametros_ia"]
        p["manos_totales"] = int(p.get("manos_totales", 0)) + 1

        if humano_canto_envido:
            p["shows_humano_totales"] = int(p.get("shows_humano_totales", 0)) + 1
            tantos_h = tr.calcular_envido(cartas_humano)
            if tantos_h < 24:
                p["bluffs_humano_detectados"] = int(p.get("bluffs_humano_detectados", 0)) + 1
                # El humano miente en el envido: bajar umbral para pagarle más seguido
                self.umbral_quiero_envido_offset = max(-0.15, self.umbral_quiero_envido_offset - 0.025)
            elif tantos_h >= 29:
                self.umbral_quiero_envido_offset = min(0.10, self.umbral_quiero_envido_offset + 0.01)

        if humano_canto_truco:
            p["shows_humano_totales"] = int(p.get("shows_humano_totales", 0)) + 1
            fuerza_h = self.percentil_fuerza_truco(cartas_humano)
            if fuerza_h < 0.44:
                p["bluffs_humano_detectados"] = int(p.get("bluffs_humano_detectados", 0)) + 1
                # El humano miente en el truco: bajar umbral para quererle más seguido
                self.umbral_quiero_truco_offset = max(-0.15, self.umbral_quiero_truco_offset - 0.025)
            elif fuerza_h >= 0.78:
                self.umbral_quiero_truco_offset = min(0.10, self.umbral_quiero_truco_offset + 0.01)

        self.guardar_historial()

    def registrar_partida_finalizada(
        self,
        ganador: str,
        puntos_humano: int,
        puntos_robot: int,
        meta_puntos: int,
        estadisticas_partida: Dict,
    ) -> Dict:
        """Guarda una partida terminada en el historial y devuelve el registro creado."""
        p = self.datos_historial["parametros_ia"]
        p["partidas_jugadas"] = int(p.get("partidas_jugadas", 0)) + 1
        if ganador == "robot":
            p["victorias_robot"] = int(p.get("victorias_robot", 0)) + 1
        else:
            p["victorias_humano"] = int(p.get("victorias_humano", 0)) + 1

        env_stats = estadisticas_partida.get("envido", {})
        tru_stats = estadisticas_partida.get("truco", {})

        registro = {
            "id": int(p["partidas_jugadas"]),
            "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ganador": ganador,
            "puntos_humano": puntos_humano,
            "puntos_robot": puntos_robot,
            "meta_puntos": meta_puntos,
            "manos_jugadas": estadisticas_partida.get("manos_jugadas", 0),
            "bluffs_envido_intentados": env_stats.get("bluffs_intentados", 0),
            "bluffs_envido_exitosos": env_stats.get("bluffs_exitosos", 0),
            "bluffs_envido_descubiertos": env_stats.get("bluffs_descubiertos", 0),
            "bluffs_truco_intentados": tru_stats.get("bluffs_intentados", 0),
            "bluffs_truco_exitosos": tru_stats.get("bluffs_exitosos", 0),
            "bluffs_truco_descubiertos": tru_stats.get("bluffs_descubiertos", 0),
            "prob_farol_envido_post": round(self.prob_farol_envido * 100.0, 1),
            "prob_farol_truco_post": round(self.prob_farol_truco * 100.0, 1),
            "umbral_envido_offset_post": round(self.umbral_quiero_envido_offset, 3),
            "umbral_truco_offset_post": round(self.umbral_quiero_truco_offset, 3),
        }
        self.datos_historial.setdefault("partidas", []).append(registro)
        self.guardar_historial()
        return registro

    # ==================== PROBABILIDADES Y DECISIONES ====================
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

        Devuelve (canto_o_None, frase_criolla) y actualiza `self.ultimo_canto_fue_farol`.
        """
        self.ultimo_canto_fue_farol = False
        self.ultimo_detalle_farol = ""
        tantos, prob = self.prob_envido(mano_completa, es_mano)
        pos_str = "mano" if es_mano else "pie"

        # --- MODO 1: AGENTE REACTIVO SIMPLE (regla fija sin probabilidades ni farol) ---
        if self.modo_agente == "reactivo":
            if tantos >= 28:
                self._registrar_razonamiento(
                    fase="Evaluación de Envido (1ra ronda)",
                    decision="ENVIDO",
                    metrica=f"Tantos propios = {tantos} (regla fija: tantos ≥ 28)",
                    umbral="Condición fija: tantos ≥ 28",
                    farol_info="Desactivado (Agente Reactivo)",
                    explicacion=f"Regla condición-acción activada: como tiene {tantos} tantos (≥ 28), canta Envido.",
                )
                return "ENVIDO", "¡Envido!"
            self._registrar_razonamiento(
                fase="Evaluación de Envido (1ra ronda)",
                decision="NO CANTAR",
                metrica=f"Tantos propios = {tantos}",
                umbral="Condición fija: tantos ≥ 28",
                farol_info="Desactivado (Agente Reactivo)",
                explicacion=f"Tiene {tantos} tantos (< 28); por regla fija no inicia Envido.",
                es_accion_principal=False,
            )
            return None, ""

        # --- MODOS 2 Y 3: UTILIDAD (474k) Y APRENDIZAJE ADAPTATIVO ---
        falta_humano = meta_puntos - puntos_humano
        falta_robot = meta_puntos - puntos_robot
        prob_pct = round(prob * 100.0, 1)
        metrica_str = f"{tantos} tantos → P(ganar | {pos_str}) = {prob_pct}% (corpus 474k)"
        usa_aprendizaje = self.modo_agente == "aprendizaje"
        farol_cfg_str = (
            f"Cuota farol activa: {round(self.prob_farol_envido * 100, 1)}%"
            if usa_aprendizaje
            else "Farol desactivado (Utilidad pura)"
        )

        if falta_humano <= 1 and prob >= 0.65:
            self._registrar_razonamiento(
                fase="Evaluación de Envido (1ra ronda)",
                decision="FALTA_ENVIDO",
                metrica=metrica_str,
                umbral="Rival a 1 pto del triunfo y P(ganar) ≥ 65%",
                farol_info=farol_cfg_str,
                explicacion="El rival está a 1 punto de ganar el partido; maximiza utilidad forzando Falta Envido.",
            )
            return "FALTA_ENVIDO", "¡A todo o nada, Falta Envido!"

        if prob >= 0.94:  # 31, 32 o 33
            if falta_robot <= 4 or self.rng.random() < 0.45:
                self._registrar_razonamiento(
                    fase="Evaluación de Envido (1ra ronda)",
                    decision="FALTA_ENVIDO",
                    metrica=metrica_str,
                    umbral="P(ganar) ≥ 94.0% (mano extrema)",
                    farol_info=farol_cfg_str,
                    explicacion=f"Con {tantos} tantos la probabilidad empírica ({prob_pct}%) supera el 94%: maximiza ganancia con Falta Envido.",
                )
                return "FALTA_ENVIDO", "¡Falta Envido te canto!"
            self._registrar_razonamiento(
                fase="Evaluación de Envido (1ra ronda)",
                decision="REAL_ENVIDO",
                metrica=metrica_str,
                umbral="P(ganar) ≥ 94.0% (3 pts esperados)",
                farol_info=farol_cfg_str,
                explicacion=f"Probabilidad empírica muy alta ({prob_pct}%); inicia con Real Envido para extraer valor.",
            )
            return "REAL_ENVIDO", "¡Real Envido!"

        if prob >= 0.85:  # 29 o 30
            canto_sel = "REAL_ENVIDO" if self.rng.random() < 0.35 else "ENVIDO"
            frase_sel = "¡Real Envido!" if canto_sel == "REAL_ENVIDO" else "¡Envido!"
            self._registrar_razonamiento(
                fase="Evaluación de Envido (1ra ronda)",
                decision=canto_sel,
                metrica=metrica_str,
                umbral="P(ganar) ≥ 85.0% (óptimo para cantar)",
                farol_info=farol_cfg_str,
                explicacion=f"Utilidad esperada positiva ({prob_pct}% ≥ 85%): inicia canto de {canto_sel}.",
            )
            return canto_sel, frase_sel

        if prob >= 0.68:  # 26 a 28
            self._registrar_razonamiento(
                fase="Evaluación de Envido (1ra ronda)",
                decision="ENVIDO",
                metrica=metrica_str,
                umbral="P(ganar) ≥ 68.0% (umbral base Envido)",
                farol_info=farol_cfg_str,
                explicacion=f"Con {tantos} tantos ({prob_pct}% ≥ 68%) la utilidad esperada de cantar Envido supera pasar.",
            )
            return "ENVIDO", "¡Envido!"

        # Cuota dinámica de picardía / farol ("mentir" con pocos tantos) solo en modo aprendizaje
        if usa_aprendizaje:
            prob_f = self.prob_farol_envido if not es_mano else (self.prob_farol_envido * 0.65)
            if self.rng.random() < prob_f:
                self.ultimo_canto_fue_farol = True
                self.ultimo_detalle_farol = f"Cantó ENVIDO de farol con {tantos} tantos"
                self._registrar_razonamiento(
                    fase="Evaluación de Envido (1ra ronda)",
                    decision="ENVIDO (FAROL)",
                    metrica=metrica_str,
                    umbral="P(ganar) < 68.0% (activación de bluff)",
                    farol_info=f"Farol activado (p={round(prob_f * 100, 1)}%)",
                    explicacion=f"Mano baja ({tantos} tantos), pero la política adaptativa ejecuta un farol para presionar al rival.",
                )
                return "ENVIDO", "¡Envido! A ver si te animás..."

        self._registrar_razonamiento(
            fase="Evaluación de Envido (1ra ronda)",
            decision="NO CANTAR",
            metrica=metrica_str,
            umbral="P(ganar) < 68.0%",
            farol_info=farol_cfg_str,
            explicacion=f"Utilidad insuficiente ({prob_pct}% < 68%) y sin disparo de farol; conserva silencio.",
            es_accion_principal=False,
        )
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
        """Responde a un canto de envido del humano ('ENVIDO', 'ENVIDO_ENVIDO', 'REAL_ENVIDO', 'FALTA_ENVIDO')."""
        self.ultimo_canto_fue_farol = False
        self.ultimo_detalle_farol = ""
        tantos, prob = self.prob_envido(mano_completa, es_mano)
        pos_str = "mano" if es_mano else "pie"
        prob_pct = round(prob * 100.0, 1)

        # --- MODO 1: AGENTE REACTIVO SIMPLE ---
        if self.modo_agente == "reactivo":
            umbral_fijo = 27 if canto_actual == "ENVIDO" else 29
            if tantos >= umbral_fijo:
                self._registrar_razonamiento(
                    fase=f"Respuesta a {canto_actual}",
                    decision="QUIERO",
                    metrica=f"Tantos propios = {tantos}",
                    umbral=f"Regla fija: tantos ≥ {umbral_fijo}",
                    farol_info="Desactivado (Agente Reactivo)",
                    explicacion=f"Regla reactiva: acepta porque {tantos} ≥ {umbral_fijo}.",
                )
                return "QUIERO", f"¡Quiero! Tengo {tantos}."
            self._registrar_razonamiento(
                fase=f"Respuesta a {canto_actual}",
                decision="NO_QUIERO",
                metrica=f"Tantos propios = {tantos}",
                umbral=f"Regla fija: tantos ≥ {umbral_fijo}",
                farol_info="Desactivado (Agente Reactivo)",
                explicacion=f"Regla reactiva: rechaza porque {tantos} < {umbral_fijo}.",
            )
            return "NO_QUIERO", "Son buenas, no quiero."

        # --- MODOS 2 Y 3: UTILIDAD (474k) Y APRENDIZAJE ADAPTATIVO ---
        falta_humano = meta_puntos - puntos_humano
        usa_aprendizaje = self.modo_agente == "aprendizaje"
        off = self.umbral_quiero_envido_offset if usa_aprendizaje else 0.0
        off_str = f"{off:+.1%}" if usa_aprendizaje else "0.0% (estático)"
        metrica_str = f"{tantos} tantos → P(ganar | {pos_str}) = {prob_pct}% (474k)"

        if puntos_humano + puntos_acumulados_no_quiero >= meta_puntos:
            self._registrar_razonamiento(
                fase=f"Respuesta a {canto_actual}",
                decision="QUIERO (Forzado por Tanteador)",
                metrica=metrica_str,
                umbral="No Quiero otorga el partido al rival",
                farol_info="Utilidad de supervivencia",
                explicacion="Decir No Quiero le daría los puntos finales al rival; aceptar domina estrictamente.",
            )
            return "QUIERO", f"¡No me queda otra, quiero con {tantos}!"

        if canto_actual == "ENVIDO":
            umbral_q = max(0.42, 0.58 + off)
            umbral_str = f"Umbral Quiero = {round(umbral_q * 100, 1)}% (base 58.0%, ajuste rival {off_str})"
            if prob >= 0.93:
                self._registrar_razonamiento(
                    fase="Respuesta a ENVIDO",
                    decision="FALTA_ENVIDO",
                    metrica=metrica_str,
                    umbral="P(ganar) ≥ 93.0%",
                    farol_info="Subida por valor esperado máximo",
                    explicacion=f"Mano superior ({prob_pct}%): contraataca con Falta Envido.",
                )
                return "FALTA_ENVIDO", "¡Para qué achicarse, Falta Envido!"
            if prob >= 0.84:
                self._registrar_razonamiento(
                    fase="Respuesta a ENVIDO",
                    decision="REAL_ENVIDO",
                    metrica=metrica_str,
                    umbral="P(ganar) ≥ 84.0%",
                    farol_info="Subida por valor esperado",
                    explicacion=f"Alta probabilidad de victoria ({prob_pct}%): sube la apuesta a Real Envido.",
                )
                return "REAL_ENVIDO", "¡Envido está primero... Real Envido!"
            if prob >= 0.78 and self.rng.random() < 0.5:
                self._registrar_razonamiento(
                    fase="Respuesta a ENVIDO",
                    decision="ENVIDO_ENVIDO",
                    metrica=metrica_str,
                    umbral="P(ganar) ≥ 78.0%",
                    farol_info="Subida por valor esperado",
                    explicacion=f"Con {prob_pct}% de éxito esperado, sube a Envido-Envido (4 pts en juego).",
                )
                return "ENVIDO_ENVIDO", "¡Envido otra vez!"
            if prob >= umbral_q:
                self._registrar_razonamiento(
                    fase="Respuesta a ENVIDO",
                    decision="QUIERO",
                    metrica=metrica_str,
                    umbral=umbral_str,
                    farol_info="Aceptación por utilidad positiva",
                    explicacion=f"P(ganar)={prob_pct}% supera el umbral ajustado ({round(umbral_q * 100, 1)}%); acepta el canto.",
                )
                return "QUIERO", f"¡Quiero! Tengo {tantos}."
            if usa_aprendizaje and self.rng.random() < (self.prob_farol_envido * 0.55):
                self.ultimo_canto_fue_farol = True
                self.ultimo_detalle_farol = f"Subió a REAL ENVIDO de farol con {tantos} tantos"
                self._registrar_razonamiento(
                    fase="Respuesta a ENVIDO",
                    decision="REAL_ENVIDO (FAROL)",
                    metrica=metrica_str,
                    umbral=umbral_str,
                    farol_info=f"Contra-farol activado (cuota {round(self.prob_farol_envido * 100, 1)}%)",
                    explicacion=f"P(ganar)={prob_pct}% no alcanza para pagar, pero ejecuta contra-farol subiendo a Real Envido.",
                )
                return "REAL_ENVIDO", "¡Real Envido, si te da el cuero!"
            self._registrar_razonamiento(
                fase="Respuesta a ENVIDO",
                decision="NO_QUIERO",
                metrica=metrica_str,
                umbral=umbral_str,
                farol_info="Sin farol",
                explicacion=f"P(ganar)={prob_pct}% es menor al umbral requerido ({round(umbral_q * 100, 1)}%); minimiza pérdida (-1 pto).",
            )
            return "NO_QUIERO", "Son buenas, no quiero."

        if canto_actual == "ENVIDO_ENVIDO":
            umbral_q = max(0.50, 0.70 + off)
            umbral_str = f"Umbral Quiero = {round(umbral_q * 100, 1)}% (base 70.0%, ajuste {off_str})"
            if prob >= 0.94:
                self._registrar_razonamiento(
                    fase="Respuesta a ENVIDO_ENVIDO",
                    decision="FALTA_ENVIDO",
                    metrica=metrica_str,
                    umbral="P(ganar) ≥ 94.0%",
                    farol_info="Subida por valor máximo",
                    explicacion=f"Probabilidad {prob_pct}% ≥ 94%: remata con Falta Envido.",
                )
                return "FALTA_ENVIDO", "¡Falta Envido!"
            if prob >= umbral_q:
                self._registrar_razonamiento(
                    fase="Respuesta a ENVIDO_ENVIDO",
                    decision="QUIERO",
                    metrica=metrica_str,
                    umbral=umbral_str,
                    farol_info="Utilidad positiva",
                    explicacion=f"P(ganar)={prob_pct}% ≥ {round(umbral_q * 100, 1)}%: acepta los 4 puntos en juego.",
                )
                return "QUIERO", f"¡Quiero! Tengo {tantos}."
            self._registrar_razonamiento(
                fase="Respuesta a ENVIDO_ENVIDO",
                decision="NO_QUIERO",
                metrica=metrica_str,
                umbral=umbral_str,
                farol_info="Aversión al riesgo",
                explicacion=f"P(ganar)={prob_pct}% < {round(umbral_q * 100, 1)}%: cede 2 puntos para no arriesgar 4.",
            )
            return "NO_QUIERO", "No quiero, dejá nomás."

        if canto_actual == "REAL_ENVIDO":
            umbral_q = max(0.52, 0.72 + off)
            umbral_str = f"Umbral Quiero = {round(umbral_q * 100, 1)}% (base 72.0%, ajuste {off_str})"
            if prob >= 0.94:
                self._registrar_razonamiento(
                    fase="Respuesta a REAL_ENVIDO",
                    decision="FALTA_ENVIDO",
                    metrica=metrica_str,
                    umbral="P(ganar) ≥ 94.0%",
                    farol_info="Subida por valor máximo",
                    explicacion=f"Mano prácticamente invencible ({prob_pct}%): sube a Falta Envido.",
                )
                return "FALTA_ENVIDO", "¡Falta Envido carajo!"
            if prob >= umbral_q:
                self._registrar_razonamiento(
                    fase="Respuesta a REAL_ENVIDO",
                    decision="QUIERO",
                    metrica=metrica_str,
                    umbral=umbral_str,
                    farol_info="Utilidad positiva",
                    explicacion=f"P(ganar)={prob_pct}% supera el umbral ({round(umbral_q * 100, 1)}%): acepta el Real Envido.",
                )
                return "QUIERO", f"¡Quiero! {tantos} tengo."
            self._registrar_razonamiento(
                fase="Respuesta a REAL_ENVIDO",
                decision="NO_QUIERO",
                metrica=metrica_str,
                umbral=umbral_str,
                farol_info="Minimización de pérdida",
                explicacion=f"P(ganar)={prob_pct}% insuficiente frente a umbral de {round(umbral_q * 100, 1)}%.",
            )
            return "NO_QUIERO", "Paso, no quiero."

        if canto_actual == "FALTA_ENVIDO":
            pts_falta = tr.puntos_falta_envido(puntos_humano, puntos_robot, meta_puntos)
            umbral = max(0.55, (0.75 if (pts_falta <= 3 or falta_humano <= 2) else 0.87) + off)
            umbral_str = f"Umbral Falta = {round(umbral * 100, 1)}% ({pts_falta} pts en juego, ajuste {off_str})"
            if prob >= umbral:
                self._registrar_razonamiento(
                    fase="Respuesta a FALTA_ENVIDO",
                    decision="QUIERO",
                    metrica=metrica_str,
                    umbral=umbral_str,
                    farol_info="Utilidad positiva en Falta",
                    explicacion=f"P(ganar)={prob_pct}% ≥ {round(umbral * 100, 1)}%: acepta jugar por {pts_falta} puntos.",
                )
                return "QUIERO", f"¡Quiero la falta! Tengo {tantos}."
            self._registrar_razonamiento(
                fase="Respuesta a FALTA_ENVIDO",
                decision="NO_QUIERO",
                metrica=metrica_str,
                umbral=umbral_str,
                farol_info="Protección de tanteador",
                explicacion=f"P(ganar)={prob_pct}% < {round(umbral * 100, 1)}%: rechaza para no comprometer el partido.",
            )
            return "NO_QUIERO", "Demasiado riesgo, no quiero."

        return "NO_QUIERO", "No quiero."

    def percentil_fuerza_truco(self, cartas: List[Dict]) -> float:
        """Calcula el percentil de fuerza (0.0 a 1.0) de la mano en base al dataset."""
        if not cartas:
            return 0.0
        poder = tr.poder_mano_truco(cartas)
        if len(cartas) == 2:
            poder = poder * (1.9 / 1.6)
        elif len(cartas) == 1:
            poder = poder * 1.9
        poder_red = round(min(23.5, max(0.0, poder)), 1)
        p_str = str(poder_red)
        if p_str in self.stats["truco_poder_percentil"]:
            return float(self.stats["truco_poder_percentil"][p_str])
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
        self.ultimo_canto_fue_farol = False
        self.ultimo_detalle_farol = ""
        fuerza = self.evaluar_situacion_baza(cartas_disponibles, ganadores_ronda, carta_rival_mesa)
        fuerza_pct = round(fuerza * 100.0, 1)

        # --- MODO 1: AGENTE REACTIVO SIMPLE ---
        if self.modo_agente == "reactivo":
            max_jer = max((tr.valor_carta_truco(c["valor"], c["palo"]) for c in cartas_disponibles), default=0)
            if nivel_actual == "NADA" and max_jer >= 10:  # Tiene un 3 o una brava
                self._registrar_razonamiento(
                    fase="Evaluación de Truco",
                    decision="TRUCO",
                    metrica=f"Jerarquía máxima en mano = {max_jer}/14",
                    umbral="Regla fija: carta ≥ 10 (3 o superior)",
                    farol_info="Desactivado (Agente Reactivo)",
                    explicacion="Regla condición-acción: posee al menos un 3 o carta brava, por lo que canta Truco.",
                )
                return "TRUCO", "¡Truco!"
            return None, ""

        # --- MODOS 2 Y 3: UTILIDAD (474k) Y APRENDIZAJE ADAPTATIVO ---
        usa_aprendizaje = self.modo_agente == "aprendizaje"
        metrica_str = f"Fuerza efectiva = {fuerza_pct}% (percentil 474k + contexto baza)"
        farol_cfg_str = (
            f"Cuota farol Truco: {round(self.prob_farol_truco * 100, 1)}%"
            if usa_aprendizaje
            else "Farol desactivado (Utilidad pura)"
        )

        if nivel_actual == "NADA":
            if fuerza >= 0.72:
                self._registrar_razonamiento(
                    fase="Evaluación de Truco",
                    decision="TRUCO",
                    metrica=metrica_str,
                    umbral="Fuerza efectiva ≥ 72.0%",
                    farol_info=farol_cfg_str,
                    explicacion=f"Fuerza de baza alta ({fuerza_pct}% ≥ 72%): canta Truco por valor esperado positivo.",
                )
                return "TRUCO", "¡Truco!"
            if usa_aprendizaje:
                prob_f = self.prob_farol_truco if len(ganadores_ronda) >= 1 else (self.prob_farol_truco * 0.45)
                if self.rng.random() < prob_f:
                    self.ultimo_canto_fue_farol = True
                    self.ultimo_detalle_farol = f"Cantó TRUCO de farol (fuerza {int(fuerza * 100)}%)"
                    self._registrar_razonamiento(
                        fase="Evaluación de Truco",
                        decision="TRUCO (FAROL)",
                        metrica=metrica_str,
                        umbral="Fuerza < 72.0% (disparo de bluff)",
                        farol_info=f"Farol ejecutado (p={round(prob_f * 100, 1)}%)",
                        explicacion=f"Mano débil/media ({fuerza_pct}%), pero la política adaptativa lanza un farol de Truco.",
                    )
                    return "TRUCO", "¡Truco te canto, a ver qué tenés!"
        elif nivel_actual == "TRUCO":
            if fuerza >= 0.86:
                self._registrar_razonamiento(
                    fase="Evaluación de Retruco",
                    decision="RETRUCO",
                    metrica=metrica_str,
                    umbral="Fuerza efectiva ≥ 86.0%",
                    farol_info=farol_cfg_str,
                    explicacion=f"Dominio de baza ({fuerza_pct}% ≥ 86%): sube la apuesta a Retruco.",
                )
                return "RETRUCO", "¡Quiero Retruco!"
            if usa_aprendizaje and self.rng.random() < (self.prob_farol_truco * 0.35):
                self.ultimo_canto_fue_farol = True
                self.ultimo_detalle_farol = f"Cantó RETRUCO de farol (fuerza {int(fuerza * 100)}%)"
                self._registrar_razonamiento(
                    fase="Evaluación de Retruco",
                    decision="RETRUCO (FAROL)",
                    metrica=metrica_str,
                    umbral="Fuerza < 86.0% (disparo de bluff)",
                    farol_info=f"Farol de Retruco activado",
                    explicacion=f"Fuerza real {fuerza_pct}%, pero presiona con Retruco de farol según perfil del rival.",
                )
                return "RETRUCO", "¡Quiero Retruco!"
        elif nivel_actual == "RETRUCO":
            if fuerza >= 0.93:
                self._registrar_razonamiento(
                    fase="Evaluación de Vale Cuatro",
                    decision="VALE_CUATRO",
                    metrica=metrica_str,
                    umbral="Fuerza efectiva ≥ 93.0%",
                    farol_info=farol_cfg_str,
                    explicacion=f"Cartas casi imbatibles ({fuerza_pct}% ≥ 93%): remata con Vale Cuatro.",
                )
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
        """Responde cuando el humano canta TRUCO, RETRUCO o VALE_CUATRO."""
        self.ultimo_canto_fue_farol = False
        self.ultimo_detalle_farol = ""
        fuerza = self.evaluar_situacion_baza(cartas_disponibles, ganadores_ronda, carta_rival_mesa)
        fuerza_pct = round(fuerza * 100.0, 1)

        # --- MODO 1: AGENTE REACTIVO SIMPLE ---
        if self.modo_agente == "reactivo":
            max_jer = max((tr.valor_carta_truco(c["valor"], c["palo"]) for c in cartas_disponibles), default=0)
            umbral_jer = 8 if canto_rival == "TRUCO" else 11
            if max_jer >= umbral_jer:
                self._registrar_razonamiento(
                    fase=f"Respuesta a {canto_rival}",
                    decision="QUIERO",
                    metrica=f"Jerarquía máxima = {max_jer}/14",
                    umbral=f"Regla fija: jerarquía ≥ {umbral_jer}",
                    farol_info="Desactivado (Agente Reactivo)",
                    explicacion=f"Regla reactiva: acepta porque su mejor carta ({max_jer}) supera el corte fijo ({umbral_jer}).",
                )
                return "QUIERO", "¡Quiero!"
            self._registrar_razonamiento(
                fase=f"Respuesta a {canto_rival}",
                decision="NO_QUIERO",
                metrica=f"Jerarquía máxima = {max_jer}/14",
                umbral=f"Regla fija: jerarquía ≥ {umbral_jer}",
                farol_info="Desactivado (Agente Reactivo)",
                explicacion=f"Regla reactiva: se va al mazo porque {max_jer} < {umbral_jer}.",
            )
            return "NO_QUIERO", "Me voy al mazo, son buenas."

        # --- MODOS 2 Y 3: UTILIDAD (474k) Y APRENDIZAJE ADAPTATIVO ---
        puntos_no_quiero = {"TRUCO": 1, "RETRUCO": 2, "VALE_CUATRO": 3}.get(canto_rival, 1)
        usa_aprendizaje = self.modo_agente == "aprendizaje"
        off = self.umbral_quiero_truco_offset if usa_aprendizaje else 0.0
        off_str = f"{off:+.1%}" if usa_aprendizaje else "0.0% (estático)"
        metrica_str = f"Fuerza efectiva = {fuerza_pct}% (percentil 474k + contexto baza)"

        if puntos_humano + puntos_no_quiero >= meta_puntos and fuerza >= 0.25:
            self._registrar_razonamiento(
                fase=f"Respuesta a {canto_rival}",
                decision="QUIERO (Supervivencia)",
                metrica=metrica_str,
                umbral="Irse al mazo entrega el partido al rival",
                farol_info="Utilidad de borde de partido",
                explicacion="Rechazar hace que el humano gane el partido inmediatamente; acepta para seguir con vida.",
            )
            return "QUIERO", "¡No me puedo ir al mazo ahora, Quiero!"

        if canto_rival == "TRUCO":
            umbral_q = max(0.28, 0.44 + off)
            umbral_str = f"Umbral Quiero = {round(umbral_q * 100, 1)}% (base 44.0%, ajuste rival {off_str})"
            if fuerza >= 0.87:
                self._registrar_razonamiento(
                    fase="Respuesta a TRUCO",
                    decision="RETRUCO",
                    metrica=metrica_str,
                    umbral="Fuerza efectiva ≥ 87.0%",
                    farol_info="Subida por utilidad esperada",
                    explicacion=f"Mano muy fuerte ({fuerza_pct}%): maximiza puntos cantando Quiero Retruco.",
                )
                return "RETRUCO", "¡Quiero Retruco!"
            if fuerza >= umbral_q:
                self._registrar_razonamiento(
                    fase="Respuesta a TRUCO",
                    decision="QUIERO",
                    metrica=metrica_str,
                    umbral=umbral_str,
                    farol_info="Aceptación por valor esperado",
                    explicacion=f"Fuerza ({fuerza_pct}%) supera el umbral ({round(umbral_q * 100, 1)}%): acepta el Truco.",
                )
                return "QUIERO", "¡Quiero!"
            if usa_aprendizaje and self.rng.random() < (self.prob_farol_truco * 0.6):
                self.ultimo_canto_fue_farol = True
                self.ultimo_detalle_farol = f"Retrucó de farol con fuerza baja ({int(fuerza * 100)}%)"
                self._registrar_razonamiento(
                    fase="Respuesta a TRUCO",
                    decision="RETRUCO (FAROL)",
                    metrica=metrica_str,
                    umbral=umbral_str,
                    farol_info=f"Contra-farol activado (cuota {round(self.prob_farol_truco * 100, 1)}%)",
                    explicacion=f"Con fuerza baja ({fuerza_pct}%), ejecuta un contra-farol de Retruco para hacer achicar al humano.",
                )
                return "RETRUCO", "¡Quiero Retruco, no me asustás!"
            self._registrar_razonamiento(
                fase="Respuesta a TRUCO",
                decision="NO_QUIERO",
                metrica=metrica_str,
                umbral=umbral_str,
                farol_info="Sin farol",
                explicacion=f"Fuerza ({fuerza_pct}%) inferior al umbral ({round(umbral_q * 100, 1)}%): se va al mazo cediendo 1 pto.",
            )
            return "NO_QUIERO", "Me voy al mazo, son buenas."

        if canto_rival == "RETRUCO":
            umbral_q = max(0.38, 0.56 + off)
            umbral_str = f"Umbral Quiero = {round(umbral_q * 100, 1)}% (base 56.0%, ajuste {off_str})"
            if fuerza >= 0.92:
                self._registrar_razonamiento(
                    fase="Respuesta a RETRUCO",
                    decision="VALE_CUATRO",
                    metrica=metrica_str,
                    umbral="Fuerza efectiva ≥ 92.0%",
                    farol_info="Subida máxima",
                    explicacion=f"Fuerza superior ({fuerza_pct}%): sube a Quiero Vale Cuatro.",
                )
                return "VALE_CUATRO", "¡Quiero Vale Cuatro!"
            if fuerza >= umbral_q:
                self._registrar_razonamiento(
                    fase="Respuesta a RETRUCO",
                    decision="QUIERO",
                    metrica=metrica_str,
                    umbral=umbral_str,
                    farol_info="Utilidad positiva",
                    explicacion=f"Fuerza ({fuerza_pct}%) ≥ {round(umbral_q * 100, 1)}%: acepta el Retruco.",
                )
                return "QUIERO", "¡Quiero ese Retruco!"
            self._registrar_razonamiento(
                fase="Respuesta a RETRUCO",
                decision="NO_QUIERO",
                metrica=metrica_str,
                umbral=umbral_str,
                farol_info="Control de daños",
                explicacion=f"Fuerza ({fuerza_pct}%) < {round(umbral_q * 100, 1)}%: no arriesga el Retruco.",
            )
            return "NO_QUIERO", "Hasta acá llegué, me voy al mazo."

        if canto_rival == "VALE_CUATRO":
            umbral_q = max(0.48, 0.68 + off)
            umbral_str = f"Umbral Quiero = {round(umbral_q * 100, 1)}% (base 68.0%, ajuste {off_str})"
            if fuerza >= umbral_q:
                self._registrar_razonamiento(
                    fase="Respuesta a VALE_CUATRO",
                    decision="QUIERO",
                    metrica=metrica_str,
                    umbral=umbral_str,
                    farol_info="Utilidad positiva",
                    explicacion=f"Fuerza ({fuerza_pct}%) ≥ {round(umbral_q * 100, 1)}%: acepta jugar por los 4 puntos.",
                )
                return "QUIERO", "¡Quiero el Vale Cuatro con todas las de la ley!"
            self._registrar_razonamiento(
                fase="Respuesta a VALE_CUATRO",
                decision="NO_QUIERO",
                metrica=metrica_str,
                umbral=umbral_str,
                farol_info="Control de daños",
                explicacion=f"Fuerza ({fuerza_pct}%) < {round(umbral_q * 100, 1)}%: cede 3 puntos antes que perder 4.",
            )
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
        ganadores_ronda = ganadores_ronda or []
        cartas_ordenadas = sorted(
            cartas_disponibles,
            key=lambda c: tr.valor_carta_truco(c["valor"], c["palo"]),
        )

        if len(cartas_disponibles) == 1:
            unica = cartas_disponibles[0]
            nom = tr.nombre_carta(unica["valor"], unica["palo"])
            self._registrar_razonamiento(
                fase=f"Selección de Carta (Baza {ronda})",
                decision=f"Jugar {nom}",
                metrica=f"Única carta restante en mano (poder {tr.valor_carta_truco(unica['valor'], unica['palo'])}/14)",
                umbral="Sin alternativa (1 carta)",
                farol_info="Jugada obligada",
                explicacion=f"Tira su última carta disponible ({nom}).",
            )
            return unica

        # --- MODO 1: AGENTE REACTIVO SIMPLE (siempre tira la carta más alta disponible) ---
        if self.modo_agente == "reactivo":
            elegida = cartas_ordenadas[-1]
            nom = tr.nombre_carta(elegida["valor"], elegida["palo"])
            self._registrar_razonamiento(
                fase=f"Selección de Carta (Baza {ronda})",
                decision=f"Jugar {nom}",
                metrica=f"Poder de carta = {tr.valor_carta_truco(elegida['valor'], elegida['palo'])}/14",
                umbral="Heurística greedy: tirar siempre la más alta",
                farol_info="Desactivado (Agente Reactivo)",
                explicacion=f"El Agente Reactivo no administra recursos entre bazas: juega directo su carta más fuerte ({nom}).",
            )
            return elegida

        # --- MODOS 2 Y 3: SELECCIÓN TÁCTICA DE UTILIDAD ---
        if carta_rival_mesa is not None:
            nom_rival = tr.nombre_carta(carta_rival_mesa["valor"], carta_rival_mesa["palo"])
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
                elegida = matadoras[0]
                nom = tr.nombre_carta(elegida["valor"], elegida["palo"])
                self._registrar_razonamiento(
                    fase=f"Selección de Carta (Baza {ronda})",
                    decision=f"Jugar {nom} (Matadora Mínima)",
                    metrica=f"Rival tiró {nom_rival} (poder {poder_rival}/14) vs propia (poder {tr.valor_carta_truco(elegida['valor'], elegida['palo'])}/14)",
                    umbral="Principio de mínimo esfuerzo para ganar baza",
                    farol_info="Optimización de recursos",
                    explicacion=f"Supera al {nom_rival} del humano usando la carta ganadora más económica ({nom}) y reserva las más altas.",
                )
                return elegida
            if pardas and (ronda == 1 or (ganadores_ronda and ganadores_ronda[0] == "robot")):
                elegida = pardas[0]
                nom = tr.nombre_carta(elegida["valor"], elegida["palo"])
                self._registrar_razonamiento(
                    fase=f"Selección de Carta (Baza {ronda})",
                    decision=f"Jugar {nom} (Empardar Baza)",
                    metrica=f"Emparda al {nom_rival} (poder {poder_rival}/14)",
                    umbral="Empate táctico favorable",
                    farol_info="Optimización de baza",
                    explicacion=f"Emparda la baza con {nom} para definir con ventaja o asegurar la mano.",
                )
                return elegida
            elegida = cartas_ordenadas[0]
            nom = tr.nombre_carta(elegida["valor"], elegida["palo"])
            self._registrar_razonamiento(
                fase=f"Selección de Carta (Baza {ronda})",
                decision=f"Jugar {nom} (Descarte Económico)",
                metrica=f"Ninguna carta supera a {nom_rival} (poder {poder_rival}/14)",
                umbral="Sacrificio de menor valor",
                farol_info="Preservación de poder",
                explicacion=f"No puede superar al {nom_rival}; descarta su carta más baja ({nom}) para guardar poder en la siguiente baza.",
            )
            return elegida

        if ronda == 1:
            if len(cartas_ordenadas) == 3:
                intermedia = cartas_ordenadas[1]
                poder_int = tr.valor_carta_truco(intermedia["valor"], intermedia["palo"])
                elegida = intermedia if poder_int >= 7 else cartas_ordenadas[2]
                nom = tr.nombre_carta(elegida["valor"], elegida["palo"])
                self._registrar_razonamiento(
                    fase="Selección de Carta (Baza 1 - Mano)",
                    decision=f"Jugar {nom}",
                    metrica=f"Poder de carta = {tr.valor_carta_truco(elegida['valor'], elegida['palo'])}/14",
                    umbral="Apertura con carta media-fuerte (≥ 7/14) o máxima",
                    farol_info="Iniciativa de 1ra baza",
                    explicacion=f"Al abrir la 1ra baza busca ganar primera con {nom} preservando su carta más brava si la intermedia es competitiva.",
                )
                return elegida
            return cartas_ordenadas[-1]

        if ronda == 2:
            if ganadores_ronda and ganadores_ronda[0] == "parda":
                elegida = cartas_ordenadas[-1]
                nom = tr.nombre_carta(elegida["valor"], elegida["palo"])
                self._registrar_razonamiento(
                    fase="Selección de Carta (Baza 2 tras Parda)",
                    decision=f"Jugar {nom} (Definición por Parda)",
                    metrica=f"1ra fue parda → gana quien lleve 2da",
                    umbral="Máxima prioridad (100% de poder)",
                    farol_info="Definición directa",
                    explicacion=f"Con primera parda, la segunda define la mano: tira su carta más fuerte ({nom}).",
                )
                return elegida
            if len(cartas_ordenadas) == 2:
                alta = tr.valor_carta_truco(cartas_ordenadas[1]["valor"], cartas_ordenadas[1]["palo"])
                if alta >= 10:
                    elegida = cartas_ordenadas[0]
                    nom = tr.nombre_carta(elegida["valor"], elegida["palo"])
                    self._registrar_razonamiento(
                        fase="Selección de Carta (Baza 2 - Ganó 1ra)",
                        decision=f"Jugar {nom} (Sondeo con Reserva)",
                        metrica=f"Ganó 1ra baza y guarda carta brava (poder {alta}/14) para 3ra",
                        umbral="Reserva estratégica para 3ra baza",
                        farol_info="Trampa táctica",
                        explicacion=f"Como ya ganó primera y tiene una carta alta en reserva, tira {nom} para obligar al rival a gastar su mejor carta.",
                    )
                    return elegida

        elegida = cartas_ordenadas[-1]
        nom = tr.nombre_carta(elegida["valor"], elegida["palo"])
        self._registrar_razonamiento(
            fase=f"Selección de Carta (Baza {ronda})",
            decision=f"Jugar {nom}",
            metrica=f"Poder de carta = {tr.valor_carta_truco(elegida['valor'], elegida['palo'])}/14",
            umbral="Maximización de probabilidad de baza",
            farol_info="Valor esperado",
            explicacion=f"Juega {nom} para asegurar la baza.",
        )
        return elegida
