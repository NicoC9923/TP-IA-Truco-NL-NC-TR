"""Precomputa las estadísticas del dataset T1K22.json (79.000 repartos / 474.000 manos).

Calcula:
1. Distribución acumulada de puntos de envido (P(rival <= X) y P(rival < X)).
2. Distribución acumulada de poder de mano de Truco.
3. Guarda un archivo liviano 'truco_stats.json' para inferencia en tiempo real.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import truco_reglas as tr


def analizar_dataset(json_path: str, out_path: str) -> None:
    print(f"Cargando dataset {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    envido_counts: Counter = Counter()
    poder_counts: Counter = Counter()
    total_manos = 0

    print("Procesando 79.000 repartos...")
    for sub in data:
        for deal in sub:
            for hand in deal["Manojos"]:
                total_manos += 1
                e = tr.calcular_envido(hand)
                envido_counts[e] += 1

                p = tr.poder_mano_truco(hand)
                poder_counts[round(p, 1)] += 1

    print(f"Total de manos procesadas: {total_manos}")

    envido_stats = {}
    acum = 0
    for pts in range(0, 34):
        cnt = envido_counts.get(pts, 0)
        acum += cnt
        prob_menor_igual = round(acum / total_manos, 4)
        prob_menor = round((acum - cnt) / total_manos, 4)
        envido_stats[str(pts)] = {
            "frecuencia": cnt,
            "prob_ganar_siendo_mano": prob_menor_igual,
            "prob_ganar_siendo_pie": prob_menor,
        }

    poder_sorted = sorted(poder_counts.keys())
    poder_stats = {}
    acum_p = 0
    for p in poder_sorted:
        cnt = poder_counts[p]
        acum_p += cnt
        poder_stats[str(p)] = round(acum_p / total_manos, 4)

    stats = {
        "total_manos": total_manos,
        "envido": envido_stats,
        "truco_poder_percentil": poder_stats,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Estadísticas guardadas exitosamente en: {out_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    ds_path = base_dir / "T1K22.json"
    salida_path = Path(__file__).resolve().parent / "truco_stats.json"
    analizar_dataset(str(ds_path), str(salida_path))
