# 🚁 Detecção e Contagem de Multidões em Mosaicos de Drones com YOLOv8

## Estratégia de Detecção

A abordagem é baseada em **Sliding Window Tiling** com supressão de duplicatas entre tiles via  **NMS global** . O fluxo é:

```
Mosaico UAV → Fatiamento em Tiles → YOLOv8 por Tile → Merge de Detecções → NMS Global → Contagem Final
```

Isso resolve os três grandes desafios de imagens aéreas:

* **Escala** : tiles menores aumentam o tamanho relativo das pessoas
* **Densidade irregular** : cada tile é analisado independentemente
* **Oclusão** : o limiar de IoU é calibrável por cenário

---

## 📦 Instalação das Dependências

```bash
pip install ultralytics opencv-python-headless numpy torch
```

---

## 🧩 Módulo 1 — Fatiamento do Mosaico (`tiling.py`)

```python
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
```

---

## 🤖 Módulo 2 — Inferência YOLOv8 (`detector.py`)

```python
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
```

---

## 🔗 Módulo 3 — NMS Global e Merge (`nms.py`)

```python
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
    """
    NMS global: elimina caixas duplicadas geradas pela sobreposição de tiles.

    Por que é necessário?
    ----------------------
    Um indivíduo na região de overlap entre dois tiles pode ser detectado
    duas vezes. O NMS global usa o IoU (Intersection over Union) para
    manter apenas a caixa com maior confiança.

    Args:
        detections    : Todas as detecções combinadas dos tiles.
        iou_threshold : IoU acima do qual duas caixas são consideradas duplicatas.

    Returns:
        Lista filtrada de detecções únicas.
    """
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
```

---

## 🎨 Módulo 4 — Visualização e Exportação (`visualizer.py`)

```python
"""
visualizer.py
Desenha as bounding boxes sobre o mosaico e salva o resultado.
"""

import cv2
import numpy as np
from typing import List
from detector import Detection


# Paleta de cores: verde para alta confiança, amarelo para média, vermelho para baixa
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
    """
    Desenha bounding boxes e confiança sobre a imagem.
    Redimensiona para visualização se o mosaico for muito grande.

    Args:
        mosaic           : Imagem original (pode ser gigante).
        detections       : Lista de detecções pós-NMS.
        max_display_size : Limite de pixels para o maior lado na saída visual.

    Returns:
        Imagem anotada (pode ser redimensionada para exibição).
    """
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
    """Adiciona contador total na imagem e salva em disco."""
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
```

---

## 🚀 Script Principal (`main.py`)

```python
"""
main.py
Pipeline completo: carrega mosaico → fatia → detecta → NMS global → visualiza.
"""

import cv2
import argparse
from ultralytics import YOLO

from tiling import slice_mosaic
from detector import run_yolo_on_tile
from nms import apply_global_nms
from visualizer import draw_detections, export_results


def parse_args():
    parser = argparse.ArgumentParser(description="Crowd Counter UAV - YOLOv8")
    parser.add_argument("--image",   required=True,              help="Caminho do mosaico")
    parser.add_argument("--model",   default="yolov8n.pt",       help="Peso do modelo YOLO")
    parser.add_argument("--imgsz",   type=int,   default=736,    help="Resolução de inferência")
    parser.add_argument("--conf",    type=float, default=0.30,   help="Confiança mínima")
    parser.add_argument("--iou",     type=float, default=0.45,   help="IoU para NMS local")
    parser.add_argument("--gnms",    type=float, default=0.50,   help="IoU para NMS global")
    parser.add_argument("--tile",    type=int,   default=1280,   help="Tamanho do tile (px)")
    parser.add_argument("--overlap", type=float, default=0.20,   help="Sobreposição dos tiles")
    parser.add_argument("--output",  default="output_crowd.jpg", help="Caminho da saída")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Carrega mosaico ────────────────────────────────────────────────────
    print(f"[1/5] Carregando mosaico: {args.image}")
    mosaic = cv2.imread(args.image)
    if mosaic is None:
        raise FileNotFoundError(f"Imagem não encontrada: {args.image}")
    print(f"      Resolução: {mosaic.shape[1]}x{mosaic.shape[0]} px")

    # ── 2. Fatia em tiles ─────────────────────────────────────────────────────
    print(f"[2/5] Fatiando em tiles {args.tile}px com {args.overlap*100:.0f}% overlap...")
    tiles = slice_mosaic(mosaic, tile_size=args.tile, overlap=args.overlap)
    print(f"      Total de tiles gerados: {len(tiles)}")

    # ── 3. Carrega modelo YOLOv8 ──────────────────────────────────────────────
    print(f"[3/5] Carregando modelo: {args.model}")
    model = YOLO(args.model)

    # ── 4. Inferência por tile ────────────────────────────────────────────────
    print(f"[4/5] Executando inferência (imgsz={args.imgsz}, conf={args.conf})...")
    all_detections = []

    for i, tile in enumerate(tiles, 1):
        dets = run_yolo_on_tile(
            model=model,
            tile=tile,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            iou_threshold=args.iou,
        )
        all_detections.extend(dets)
        print(f"      Tile {i:03d}/{len(tiles)} → {len(dets):3d} detecções", end="\r")

    print(f"\n      Sub-total pré-NMS global: {len(all_detections)} caixas")

    # ── 5. NMS global ─────────────────────────────────────────────────────────
    final_detections = apply_global_nms(all_detections, iou_threshold=args.gnms)
    count = len(final_detections)
    print(f"\n[5/5] ✅ Contagem final: {count} pessoas detectadas")

    # ── 6. Exporta resultado ──────────────────────────────────────────────────
    annotated = draw_detections(mosaic, final_detections)
    export_results(annotated, count, output_path=args.output)


if __name__ == "__main__":
    main()
```

---

## ⚙️ Guia de Calibração de Hiperparâmetros

| Cenário                         | `--conf` | `--iou` | `--gnms` | `--imgsz` | `--model`    |
| -------------------------------- | ---------- | --------- | ---------- | ----------- | -------------- |
| Multidão densa, alta altitude   | `0.25`   | `0.35`  | `0.40`   | `1024`    | `yolov8m.pt` |
| Multidão moderada               | `0.30`   | `0.45`  | `0.50`   | `736`     | `yolov8n.pt` |
| Pessoas esparsas, baixa altitude | `0.45`   | `0.50`  | `0.55`   | `640`     | `yolov8n.pt` |
| Máxima precisão (lento)        | `0.25`   | `0.35`  | `0.45`   | `1280`    | `yolov8x.pt` |

---

## ▶️ Execução

```bash
# Caso básico
python main.py --image mosaico_drone.jpg

# Alta densidade, modelo maior
python main.py \
  --image mosaico_evento.tif \
  --model yolov8m.pt \
  --imgsz 1024 \
  --conf 0.25 \
  --iou 0.35 \
  --gnms 0.40 \
  --tile 1280 \
  --overlap 0.25 \
  --output resultado_evento.jpg
```

> **Dica:** Para mosaicos muito grandes (ex: GeoTIFF de 10k×10k px), considere usar `--tile 640` com `--overlap 0.30` para garantir cobertura total sem estourar a memória da GPU.
>
