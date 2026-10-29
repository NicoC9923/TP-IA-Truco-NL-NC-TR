"""Módulo de Visión por Computadora para Detección de Cartas Españolas (OpenCV).

Permite:
1. Detectar el contorno rectangular de una carta en el frame de la webcam.
2. Extraer la región de interés (ROI) y corregir la perspectiva.
3. Clasificar el palo (Espada, Basto, Oro, Copa) mediante distribución de color en HSV.
4. Estimar el valor (1 al 12) por conteo de componentes conexas internas.
5. Hacer matcheo inteligente contra las cartas disponibles en la mano del jugador
   para máxima precisión bajo cualquier iluminación de webcam.
"""

from __future__ import annotations

import base64
from typing import Dict, List, Optional, Tuple
import truco_reglas as tr

try:
    import cv2
    import numpy as np
    _CV2_DISPONIBLE = True
except ImportError:
    _CV2_DISPONIBLE = False


class DetectorCartas:
    def __init__(self):
        self.disponible = _CV2_DISPONIBLE

    def procesar_frame_base64(
        self,
        b64_str: str,
        cartas_mano: Optional[List[Dict]] = None,
    ) -> Optional[Dict]:
        """Recibe una imagen en formato data:image/jpeg;base64,... y detecta la carta."""
        if not _CV2_DISPONIBLE or not b64_str:
            return None
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None
        if frame is None:
            return None
        return self.detectar_carta(frame, cartas_mano=cartas_mano)

    def detectar_carta(
        self,
        frame: "np.ndarray",
        cartas_mano: Optional[List[Dict]] = None,
    ) -> Dict:
        """Detecta la carta en la imagen y, si se provee `cartas_mano`, también
        encuentra la mejor coincidencia dentro de la mano actual del jugador.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 45, 145)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        carta_roi = None
        mejor_area = 0.0
        contorno_detectado = False
        h_frame, w_frame = frame.shape[:2]
        area_minima = (w_frame * h_frame) * 0.04

        for c in contours:
            area = cv2.contourArea(c)
            if area > area_minima and area > mejor_area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.025 * peri, True)
                if len(approx) == 4:
                    mejor_area = area
                    carta_roi = self._extraer_roi_rect(frame, approx)
                    contorno_detectado = True

        if carta_roi is None:
            margin_y = int(h_frame * 0.18)
            margin_x = int(w_frame * 0.22)
            carta_roi = frame[margin_y : h_frame - margin_y, margin_x : w_frame - margin_x]

        palo_libre, conf_palo, scores_palo = self._estimar_palo(carta_roi)
        valor_libre, conf_valor = self._estimar_valor(carta_roi)
        conf_libre = round((conf_palo + conf_valor) / 2.0, 2)

        resultado = {
            "palo": palo_libre,
            "valor": valor_libre,
            "nombre": tr.nombre_carta(valor_libre, palo_libre),
            "confianza": conf_libre,
            "contorno_detectado": contorno_detectado,
            "deteccion_libre": {
                "palo": palo_libre,
                "valor": valor_libre,
                "nombre": tr.nombre_carta(valor_libre, palo_libre),
                "confianza": conf_libre,
            },
        }

        if cartas_mano:
            mejor_carta, conf_mano = self._matchear_con_mano(
                cartas_mano, palo_libre, valor_libre, scores_palo
            )
            if mejor_carta is not None:
                resultado["palo"] = mejor_carta["palo"]
                resultado["valor"] = mejor_carta["valor"]
                resultado["nombre"] = tr.nombre_carta(mejor_carta["valor"], mejor_carta["palo"])
                resultado["confianza"] = conf_mano
                resultado["ajustada_a_mano"] = True

        return resultado

    def _matchear_con_mano(
        self,
        cartas_mano: List[Dict],
        palo_est: str,
        valor_est: int,
        scores_palo: Dict[str, int],
    ) -> Tuple[Optional[Dict], float]:
        """Puntúa cada carta disponible en la mano del jugador contra los rasgos visuales medidos."""
        if not cartas_mano:
            return None, 0.0
        if len(cartas_mano) == 1:
            return cartas_mano[0], 0.95

        total_px = sum(scores_palo.values()) + 1
        mejor_c = cartas_mano[0]
        mejor_score = -1.0

        for c in cartas_mano:
            p_c = str(c["palo"]).capitalize()
            v_c = int(c["valor"])

            # Afinidad de palo por color HSV
            ratio_palo = scores_palo.get(p_c, 0) / total_px
            bonus_palo = 0.45 if p_c == palo_est else (ratio_palo * 0.40)

            # Afinidad de valor por cercanía de componentes conexas
            diff_v = abs(v_c - valor_est)
            if diff_v == 0:
                bonus_val = 0.50
            elif diff_v == 1:
                bonus_val = 0.32
            elif diff_v <= 2:
                bonus_val = 0.18
            else:
                bonus_val = max(0.0, 0.12 - diff_v * 0.01)

            score_total = bonus_palo + bonus_val
            if score_total > mejor_score:
                mejor_score = score_total
                mejor_c = c

        conf = round(min(0.98, max(0.68, 0.55 + mejor_score * 0.5)), 2)
        return mejor_c, conf

    def _extraer_roi_rect(self, frame: "np.ndarray", pts: "np.ndarray") -> "np.ndarray":
        """Aplica transformación de perspectiva para obtener la carta de frente."""
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr_pt, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr_pt - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr_pt - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        if maxWidth <= 20 or maxHeight <= 20:
            return frame

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

    def _estimar_palo(self, roi: "np.ndarray") -> Tuple[str, float, Dict[str, int]]:
        """Estima el palo según la distribución de color predominante en HSV."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Oro: amarillo / dorado
        mask_oro = cv2.inRange(hsv, np.array([14, 85, 85]), np.array([35, 255, 255]))
        # Copa: rojo / naranja-rojizo
        mask_copa1 = cv2.inRange(hsv, np.array([0, 90, 80]), np.array([12, 255, 255]))
        mask_copa2 = cv2.inRange(hsv, np.array([162, 90, 80]), np.array([180, 255, 255]))
        mask_copa = cv2.bitwise_or(mask_copa1, mask_copa2)
        # Basto: verde
        mask_basto = cv2.inRange(hsv, np.array([36, 60, 55]), np.array([86, 255, 255]))
        # Espada: azul / celeste acero
        mask_espada = cv2.inRange(hsv, np.array([88, 55, 60]), np.array([135, 255, 255]))

        scores = {
            "Oro": int(cv2.countNonZero(mask_oro)),
            "Copa": int(cv2.countNonZero(mask_copa)),
            "Basto": int(cv2.countNonZero(mask_basto)),
            "Espada": int(cv2.countNonZero(mask_espada)),
        }

        palo_max = max(scores, key=scores.get)
        total_pixels = sum(scores.values()) + 1
        conf = min(0.96, max(0.45, scores[palo_max] / (total_pixels * 0.48 + 1e-5)))

        if scores[palo_max] < 40:
            return "Espada", 0.55, scores

        return palo_max, round(conf, 2), scores

    def _estimar_valor(self, roi: "np.ndarray") -> Tuple[int, float]:
        """Estima el número contando componentes conexas internas."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8
        )

        h, w = thresh.shape
        inner = thresh[int(h * 0.14) : int(h * 0.86), int(w * 0.14) : int(w * 0.86)]

        contours, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_min = (w * h) * 0.004
        area_max = (w * h) * 0.32
        simbolos = [c for c in contours if area_min < cv2.contourArea(c) < area_max]

        conteo = len(simbolos)
        if conteo in (1, 2, 3, 4, 5, 6, 7):
            return conteo, 0.78
        if conteo in (8, 9):
            return 10, 0.68
        if conteo == 10:
            return 11, 0.65
        if conteo > 10:
            return 12, 0.64
        return 1, 0.55
