"""
detector.py
Executa YOLOv8 sobre cada tile e mapeia as bounding boxes
para coordenadas globais do mosaico.
"""

import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List

from tiling import Tile


# Classes do COCO que representam pessoas em imagens aéreas.
# Unificá-las reduz falsos negativos causados por variações de pose e altitude.
PERSON_CLASSES = {
    0: "person",       # COCO class 0
    # Se usar modelo especializado UAV, adicione IDs adicionais aqui
}


@dataclass
class Detection:
    """Detecção individual com coordenadas globais no mosaico."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str = "person"


def run_yolo_on_tile(
    model: YOLO,
    tile: Tile,
    conf_threshold: float = 0.30,
    imgsz: int = 736,
    iou_threshold: float = 0.45,
) -> List[Detection]:
    """
    Executa inferência em um tile e converte as caixas
    para coordenadas absolutas no mosaico original.

    Args:
        model          : Instância do YOLOv8 carregada.
        tile           : Tile com imagem e offsets.
        conf_threshold : Confiança mínima para aceitar uma detecção.
                         ↓ para cenas densas / ↑ para reduzir FP em cenas esparsas.
        imgsz          : Resolução de entrada da rede.
                         Valores ≥ 736 melhoram a detecção de pessoas pequenas.
        iou_threshold  : IoU para NMS interna do YOLO.
                         ↓ (ex: 0.35) em multidões densas para separar indivíduos.

    Returns:
        Lista de Detection com coordenadas globais.
    """
    results = model.predict(
        source=tile.image,
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        classes=list(PERSON_CLASSES.keys()),  # filtra apenas pessoas
        verbose=False,
        device="cuda" if _cuda_available() else "cpu",
    )

    detections: List[Detection] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            # Coordenadas locais ao tile
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Somente classes mapeadas como "pessoa"
            if cls_id not in PERSON_CLASSES:
                continue

            # Converte para coordenadas globais no mosaico
            detections.append(Detection(
                x1=x1 + tile.x_offset,
                y1=y1 + tile.y_offset,
                x2=x2 + tile.x_offset,
                y2=y2 + tile.y_offset,
                confidence=conf,
            ))

    return detections


def _cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()
