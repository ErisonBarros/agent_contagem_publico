"""
nms.py
Aplica Non-Maximum Suppression (NMS) global sobre todas as detecções
provenientes de múltiplos tiles para eliminar duplicatas de sobreposição.
"""

import numpy as np
import cv2
from typing import List
from detector import Detection

def apply_global_nms(
    detections: List[Detection],
    iou_threshold: float = 0.50,
) -> List[Detection]:
    if not detections:
        return []

    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
    scores = np.array([d.confidence for d in detections], dtype=np.float32)

    # cv2.dnn.NMSBoxes espera (x, y, w, h)
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]   # w = x2 - x1
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]   # h = y2 - y1

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,       # filtro já foi feito na inferência
        nms_threshold=iou_threshold,
    )

    if len(indices) == 0:
        return []

    kept = [detections[i] for i in indices.flatten()]
    return kept
