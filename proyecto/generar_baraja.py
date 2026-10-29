"""Generador en Alta Resolución (344x432) de las 40 cartas de la Baraja Española.

Toma `Naipes_Españoles_Template.jfif` y:
1. Limpia el ruido de compresión JPEG del fondo y realza bordes/colores de las ilustraciones
   mediante filtrado bilateral + supermuestreo Lanczos 4x + refuerzo de trazos de tinta.
2. Redibuja en resolución nativa 344x432 el marco exterior, el recuadro interior con las
   pintas tradicionales y los números de las esquinas (1..7, 10, 11, 12) en las 40 cartas,
   para que todos los números y bordes se vean completamente nítidos.
"""

from __future__ import annotations

import os
import cv2
import numpy as np


PALOS = ["Oro", "Copa", "Espada", "Basto"]
VALORES = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]

# Escala 4x sobre el tamaño base (86x108 -> 344x432)
S = 4
W_HI = 86 * S   # 344 px
H_HI = 108 * S  # 432 px


def extraer_15_cartas_base(ruta_template: str) -> list[np.ndarray]:
    img = cv2.imread(ruta_template)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer el template: {ruta_template}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted(
        [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 2000],
        key=lambda b: (b[1] // 100, b[0]),
    )
    if len(boxes) != 15:
        raise RuntimeError(f"Se esperaban 15 cartas en el template, se hallaron {len(boxes)}")
    return [
        cv2.resize(img[y : y + h, x : x + w], (86, 108), interpolation=cv2.INTER_CUBIC)
        for (x, y, w, h) in boxes
    ]


def mejorar_arte_4x(bgr_low: np.ndarray) -> np.ndarray:
    """Escala un recorte de arte 4x eliminando ruido JPEG y reforzando trazos y color."""
    h, w = bgr_low.shape[:2]
    up = cv2.resize(bgr_low, (w * S, h * S), interpolation=cv2.INTER_LANCZOS4)

    # 1. Filtrado bilateral para quitar bloques JPEG preservando bordes de tinta
    suave = cv2.bilateralFilter(up, d=7, sigmaColor=38, sigmaSpace=38)

    # 2. Realce de nitidez (Unsharp Mask)
    blur = cv2.GaussianBlur(suave, (0, 0), 1.4)
    nitido = cv2.addWeighted(suave, 1.55, blur, -0.55, 0)

    # 3. Limpiar el fondo casi blanco (mosquito noise de JPEG) a blanco puro
    hsv = cv2.cvtColor(nitido, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(nitido, cv2.COLOR_BGR2GRAY)
    fondo_blanco = (gray > 224) & (hsv[:, :, 1] < 38)
    nitido[fondo_blanco] = (255, 255, 255)

    # 4. Saturar levemente los colores vivos (oros, copas, capas) y oscurecer trazos negros
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.18, 0, 255)
    nitido = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    nitido[fondo_blanco] = (255, 255, 255)

    trazo_oscuro = gray < 95
    nitido[trazo_oscuro] = (nitido[trazo_oscuro].astype(np.float32) * 0.72).astype(np.uint8)

    return nitido


def extraer_rgba_hi(bgr_low: np.ndarray, umbral_blanco: int = 232) -> np.ndarray:
    """Convierte un recorte de baja resolución en un sprite RGBA 4x limpio y nítido."""
    arte_hi = mejorar_arte_4x(bgr_low)
    gray = cv2.cvtColor(arte_hi, cv2.COLOR_BGR2GRAY)

    _, mask_bin = cv2.threshold(gray, umbral_blanco, 255, cv2.THRESH_BINARY_INV)
    # Rellenar huecos internos de la figura (amarillo del oro, piel, etc.)
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_fill = np.zeros_like(gray)
    for c in cnts:
        if cv2.contourArea(c) > 40:
            cv2.drawContours(mask_fill, [c], -1, 255, -1)

    # Suavizar el borde del canal alfa (antialiasing)
    alpha = cv2.GaussianBlur(mask_fill, (3, 3), 0.8)
    rgba = cv2.cvtColor(arte_hi, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def pegar_rgba(
    lienzo: np.ndarray,
    sprite_rgba: np.ndarray,
    cx: float,
    cy: float,
    escala_x: float = 1.0,
    escala_y: float = 1.0,
    rotar_grados: float = 0.0,
):
    """Pega un sprite RGBA 4x en las coordenadas (cx*S, cy*S) del lienzo de alta resolución."""
    h0, w0 = sprite_rgba.shape[:2]
    nw = max(8, int(round(w0 * escala_x)))
    nh = max(8, int(round(h0 * escala_y)))
    sp = cv2.resize(sprite_rgba, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    if abs(rotar_grados) > 1e-3:
        if abs(rotar_grados - 180.0) < 1e-3:
            sp = cv2.rotate(sp, cv2.ROTATE_180)
        else:
            M = cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), rotar_grados, 1.0)
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            bw = int((nh * sin) + (nw * cos))
            bh = int((nh * cos) + (nw * sin))
            M[0, 2] += (bw / 2.0) - nw / 2.0
            M[1, 2] += (bh / 2.0) - nh / 2.0
            sp = cv2.warpAffine(
                sp,
                M,
                (bw, bh),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255, 0),
            )

    sh, sw = sp.shape[:2]
    x1 = int(round(cx * S - sw / 2.0))
    y1 = int(round(cy * S - sh / 2.0))
    x2 = x1 + sw
    y2 = y1 + sh

    H, W = lienzo.shape[:2]
    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = sw - max(0, x2 - W)
    sy2 = sh - max(0, y2 - H)

    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = min(W, x2)
    dy2 = min(H, y2)

    if dx1 >= dx2 or dy1 >= dy2:
        return

    recorte = sp[sy1:sy2, sx1:sx2]
    alfa = (recorte[:, :, 3:4].astype(np.float32)) / 255.0
    destino = lienzo[dy1:dy2, dx1:dx2].astype(np.float32)
    lienzo[dy1:dy2, dx1:dx2] = np.clip(
        recorte[:, :, :3].astype(np.float32) * alfa + destino * (1.0 - alfa),
        0,
        255,
    ).astype(np.uint8)


def dibujar_marco_e_indices_hi(card_hi: np.ndarray, valor: int, palo: str) -> None:
    """Dibuja en resolución nativa 344x432 el borde exterior, el recuadro con pintas
    y los números de las esquinas completamente nítidos.
    """
    W, H = W_HI, H_HI  # 344 x 432

    # Limpiar los márgenes exteriores alrededor del recuadro interno (x=30..314, y=26..406)
    card_hi[0:28, :] = 255
    card_hi[404:, :] = 255
    card_hi[:, 0:31] = 255
    card_hi[:, 313:] = 255

    # Esquinas redondeadas exteriores oscuras (estilo naipe español clásico)
    mask_round = np.zeros((H, W), dtype=np.uint8)
    r_ext = 24
    cv2.rectangle(mask_round, (4, r_ext), (W - 5, H - r_ext - 1), 255, -1)
    cv2.rectangle(mask_round, (r_ext, 4), (W - r_ext - 1, H - 5), 255, -1)
    for cx, cy in [(r_ext, r_ext), (W - r_ext - 1, r_ext), (r_ext, H - r_ext - 1), (W - r_ext - 1, H - r_ext - 1)]:
        cv2.circle(mask_round, (cx, cy), r_ext - 4, 255, -1)

    card_hi[mask_round == 0] = (18, 18, 18)

    # Borde perimetral del naipe
    contours, _ = cv2.findContours(mask_round, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(card_hi, contours, -1, (40, 40, 40), 2, cv2.LINE_AA)

    # Recuadro interior tradicional (x=30..314, y=26..406)
    cv2.rectangle(card_hi, (30, 26), (314, 406), (25, 25, 25), 3, cv2.LINE_AA)

    # Pintas tradicionales en el borde superior (y=26) e inferior (y=406)
    cortes = {
        "Oro": [],
        "Copa": [172],
        "Espada": [136, 208],
        "Basto": [116, 172, 228],
    }[palo]
    for x in cortes:
        cv2.line(card_hi, (x - 10, 26), (x + 10, 26), (255, 255, 255), 5)
        cv2.line(card_hi, (x - 10, 406), (x + 10, 406), (255, 255, 255), 5)

    # Números de alta definición en esquina superior izquierda y esquina inferior derecha
    txt = str(valor)
    fuente = cv2.FONT_HERSHEY_TRIPLEX
    escala = 0.85 if len(txt) == 1 else 0.72
    grosor = 2
    color_num = (18, 18, 18)

    # Superior izquierda (con halo blanco suave para no cortar el dibujo)
    cv2.putText(card_hi, txt, (37, 58), fuente, escala, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(card_hi, txt, (37, 58), fuente, escala, color_num, grosor, cv2.LINE_AA)

    # Inferior derecha (rotada 180 grados con halo blanco suave)
    parche = np.full((46, 54, 3), 255, dtype=np.uint8)
    px = 10 if len(txt) == 1 else 3
    cv2.putText(parche, txt, (px, 34), fuente, escala, color_num, grosor, cv2.LINE_AA)
    parche = cv2.rotate(parche, cv2.ROTATE_180)
    mask_num = cv2.cvtColor(parche, cv2.COLOR_BGR2GRAY) < 200
    # Halo blanco alrededor del número inferior derecho
    kernel = np.ones((5, 5), np.uint8)
    halo = cv2.dilate(mask_num.astype(np.uint8), kernel, iterations=1) > 0
    roi_inf = card_hi[356:402, 256:310]
    roi_inf[halo] = 255
    roi_inf[mask_num] = parche[mask_num]


def crear_carta_directa_hi(carta_low: np.ndarray, valor: int, palo: str) -> np.ndarray:
    """Toma una de las 15 cartas del template, borra su marco e índices viejos borrosos,
    mejora el arte 4x y redibuja el marco y los números en alta resolución.
    """
    copia = carta_low.copy()
    # Borrar el marco interno viejo de baja resolución (x<=10, x>=75, y<=9, y>=98)
    copia[0:10, :] = 255
    copia[98:, :] = 255
    copia[:, 0:11] = 255
    copia[:, 75:] = 255
    # Borrar los números viejos de las esquinas
    copia[8:17, 8:17] = 255
    copia[90:99, 66:77] = 255
    card_hi = mejorar_arte_4x(copia)
    dibujar_marco_e_indices_hi(card_hi, valor, palo)
    return card_hi


def generar_baraja_40(ruta_template: str, dir_salida: str) -> dict[str, str]:
    os.makedirs(dir_salida, exist_ok=True)
    c = extraer_15_cartas_base(ruta_template)

    directas = {
        (1, "Basto"): c[0],
        (1, "Espada"): c[1],
        (1, "Copa"): c[2],
        (1, "Oro"): c[3],
        (10, "Copa"): c[4],
        (2, "Espada"): c[5],
        (3, "Basto"): c[6],
        (4, "Copa"): c[7],
        (5, "Oro"): c[8],
        (11, "Basto"): c[9],
        (6, "Oro"): c[10],
        (7, "Copa"): c[11],
        (12, "Basto"): c[12],
        (11, "Copa"): c[13],
        (12, "Espada"): c[14],
    }

    # Sprites 4x de cada palo extraídos de regiones limpias (lejos de los números de esquina)
    pip_oro = extraer_rgba_hi(c[10][41:65, 45:74])
    pip_copa = extraer_rgba_hi(c[7][11:41, 50:73])
    pip_espada = extraer_rgba_hi(c[5][14:96, 15:39])
    basto_crop = c[0][14:95, 17:71].copy()
    pip_basto_diag = extraer_rgba_hi(basto_crop)

    # Coordenadas estándar (cx, cy, rotacion) en unidades de 86x108
    pos_2 = [(43, 33, 0), (43, 75, 180)]
    pos_3 = [(43, 27, 0), (43, 54, 0), (43, 81, 180)]
    pos_3_espada = [(27, 54, 0), (43, 54, 180), (59, 54, 0)]
    pos_4 = [(28, 30, 0), (58, 30, 0), (28, 78, 180), (58, 78, 180)]
    pos_5 = [(28, 28, 0), (58, 28, 0), (43, 54, 0), (28, 80, 180), (58, 80, 180)]
    pos_6 = [
        (28, 26, 0), (58, 26, 0),
        (28, 54, 0), (58, 54, 0),
        (28, 82, 180), (58, 82, 180),
    ]
    pos_7 = [
        (26, 26, 0), (60, 26, 0),
        (43, 37, 0),
        (26, 54, 0), (60, 54, 0),
        (26, 82, 180), (60, 82, 180),
    ]
    layouts = {2: pos_2, 3: pos_3, 4: pos_4, 5: pos_5, 6: pos_6, 7: pos_7}

    rutas_generadas = {}

    for palo in PALOS:
        for valor in VALORES:
            if (valor, palo) in directas:
                card_hi = crear_carta_directa_hi(directas[(valor, palo)], valor, palo)
            elif valor in layouts:
                card_hi = np.full((H_HI, W_HI, 3), 255, dtype=np.uint8)
                coords = pos_3_espada if (palo == "Espada" and valor == 3) else layouts[valor]
                for (cx, cy, rot) in coords:
                    if palo == "Oro":
                        esc = 1.02 if valor <= 4 else 0.94
                        pegar_rgba(card_hi, pip_oro, cx, cy, esc, esc, 0)
                    elif palo == "Copa":
                        esc = 0.98 if valor <= 4 else 0.90
                        pegar_rgba(card_hi, pip_copa, cx, cy, esc, esc, 0)
                    elif palo == "Espada":
                        if valor == 3:
                            pegar_rgba(card_hi, pip_espada, cx, cy, 0.76, 0.84, rot)
                        elif valor in (4, 5):
                            pegar_rgba(card_hi, pip_espada, cx, cy, 0.68, 0.36, rot)
                        else:
                            pegar_rgba(card_hi, pip_espada, cx, cy, 0.64, 0.27, rot)
                    elif palo == "Basto":
                        ang = -25.0 + rot
                        if valor == 2:
                            pegar_rgba(card_hi, pip_basto_diag, cx, cy, 0.50, 0.42, ang)
                        elif valor in (4, 5):
                            pegar_rgba(card_hi, pip_basto_diag, cx, cy, 0.42, 0.34, ang)
                        else:
                            pegar_rgba(card_hi, pip_basto_diag, cx, cy, 0.36, 0.26, ang)
                dibujar_marco_e_indices_hi(card_hi, valor, palo)
            elif valor == 10:
                copia = c[4].copy()
                copia[8:27, 15:32] = 255
                card_hi = crear_carta_directa_hi(copia, 10, palo)
                if palo == "Oro":
                    pegar_rgba(card_hi, pip_oro, 24, 19, 0.62, 0.62, 0)
                elif palo == "Espada":
                    pegar_rgba(card_hi, pip_espada, 24, 21, 0.55, 0.28, 180)
                elif palo == "Basto":
                    pegar_rgba(card_hi, pip_basto_diag, 24, 20, 0.36, 0.30, -15)
                dibujar_marco_e_indices_hi(card_hi, 10, palo)
            elif valor == 11:
                copia = c[13].copy()
                copia[8:26, 20:36] = 255
                card_hi = crear_carta_directa_hi(copia, 11, palo)
                if palo == "Oro":
                    pegar_rgba(card_hi, pip_oro, 28, 18, 0.64, 0.64, 0)
                elif palo == "Espada":
                    pegar_rgba(card_hi, pip_espada, 28, 19, 0.56, 0.28, 180)
                dibujar_marco_e_indices_hi(card_hi, 11, palo)
            elif valor == 12:
                copia = c[14].copy()
                copia[10:52, 13:27] = 255
                card_hi = crear_carta_directa_hi(copia, 12, palo)
                if palo == "Oro":
                    pegar_rgba(card_hi, pip_oro, 22, 34, 0.66, 0.66, 0)
                elif palo == "Copa":
                    pegar_rgba(card_hi, pip_copa, 22, 33, 0.70, 0.70, 0)
                dibujar_marco_e_indices_hi(card_hi, 12, palo)

            nombre_archivo = f"{valor}_{palo}.png"
            ruta_out = os.path.join(dir_salida, nombre_archivo)
            cv2.imwrite(ruta_out, card_hi)
            rutas_generadas[f"{valor}_{palo}"] = ruta_out

    return rutas_generadas


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir_cartas = os.path.join(base_dir, "cartas")
    os.makedirs(dir_cartas, exist_ok=True)
    ruta_tpl = os.path.join(dir_cartas, "template_naipes.jpg")
    generadas = generar_baraja_40(ruta_tpl, dir_cartas)
    print(f"Generadas {len(generadas)} cartas HD (344x432) en {dir_cartas}")
