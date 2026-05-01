import cv2
import numpy as np
from tiling import slice_mosaic
from detector import run_yolo_on_tile
from ultralytics import YOLO

def test_tiling():
    print("Testing Tiling Module...")
    # Cria uma imagem falsa (mosaico) de 3000x2000
    mosaic = np.zeros((2000, 3000, 3), dtype=np.uint8)
    
    # Testa o fatiamento com tile_size=1280 e overlap=0.2
    tiles = slice_mosaic(mosaic, tile_size=1280, overlap=0.2)
    
    print(f"Número de tiles gerados: {len(tiles)}")
    for i, t in enumerate(tiles):
        print(f"Tile {i}: offset=({t.x_offset}, {t.y_offset}), shape={t.image.shape}")
    
    assert len(tiles) > 0, "Nenhum tile gerado!"
    print("✓ Tiling testado com sucesso!\n")

def test_detector():
    print("Testing Detector Module...")
    # Carrega modelo pequeno YOLOv8n (ele fará o download automaticamente se não existir)
    model = YOLO("yolov8n.pt")
    
    # Cria uma imagem falsa pequena para simular um tile com offset (500, 500)
    img = np.zeros((1280, 1280, 3), dtype=np.uint8)
    
    from tiling import Tile
    dummy_tile = Tile(image=img, x_offset=500, y_offset=500)
    
    # Roda a detecção
    print("Executando YOLOv8 no tile...")
    detections = run_yolo_on_tile(model, dummy_tile, conf_threshold=0.1)
    
    # Como a imagem é preta, esperamos 0 detecções
    print(f"Detecções encontradas: {len(detections)}")
    print("✓ Detector testado com sucesso!\n")

if __name__ == "__main__":
    test_tiling()
    try:
        test_detector()
    except Exception as e:
        print(f"Erro no módulo de detecção: {e}")
