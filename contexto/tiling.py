"""
tiling.py
Responsável por fatiar o mosaico em tiles com sobreposição (overlap),
evitando perdas de detecção nas bordas.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Tile:
    """Representa um tile com sua imagem e offset global."""
    image: np.ndarray
    x_offset: int   # coluna (px) no mosaico original
    y_offset: int   # linha  (px) no mosaico original


def slice_mosaic(
    mosaic: np.ndarray,
    tile_size: int = 1280,
    overlap: float = 0.2
) -> List[Tile]:
    """
    Fatia o mosaico em tiles quadrados com sobreposição.

    Args:
        mosaic    : Imagem completa (H x W x 3, BGR).
        tile_size : Lado do tile em pixels.
        overlap   : Fração de sobreposição entre tiles adjacentes (0–1).

    Returns:
        Lista de objetos Tile com imagem e posição global.
    """
    h, w = mosaic.shape[:2]
    step = int(tile_size * (1 - overlap))   # passo com overlap
    tiles: List[Tile] = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            # Garante que o tile não ultrapasse as bordas do mosaico
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)

            crop = mosaic[y_start:y_end, x_start:x_end]
            tiles.append(Tile(image=crop, x_offset=x_start, y_offset=y_start))

    return tiles
