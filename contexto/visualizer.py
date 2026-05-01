"""
visualizer.py
Desenha as bounding boxes sobre o mosaico e salva o resultado.
"""

import cv2
import numpy as np
from typing import List
from detector import Detection

def _confidence_color(conf: float) -> tuple:
    if conf >= 0.70:
        return (0, 220, 0)      # verde
    elif conf >= 0.45:
        return (0, 200, 255)    # amarelo
    else:
        return (0, 60, 255)     # vermelho

def draw_detections(
    mosaic: np.ndarray,
    detections: List[Detection],
    max_display_size: int = 4000,
) -> np.ndarray:
    annotated = mosaic.copy()

    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        color = _confidence_color(det.confidence)
        label = f"{det.confidence:.2f}"

        # Caixa delimitadora
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Fundo do texto
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Redimensiona para visualização gerenciável
    h, w = annotated.shape[:2]
    scale = min(max_display_size / max(h, w), 1.0)
    if scale < 1.0:
        annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    return annotated

def export_results(
    annotated: np.ndarray,
    count: int,
    output_path: str = "output_crowd.jpg",
) -> None:
    h, w = annotated.shape[:2]
    banner = f"TOTAL DETECTADO: {count} pessoas"

    # Fundo semitransparente no topo
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    cv2.putText(annotated, banner, (15, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, annotated)
    print(f"[✓] Resultado salvo em: {output_path}")
