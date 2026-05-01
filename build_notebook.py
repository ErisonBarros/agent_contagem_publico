import json

def create_colab_notebook():
    cells = []
    
    def add_markdown(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [text]
        })
        
    def add_code(source):
        # source can be a string, we split by newline to keep notebook format
        lines = [line + "\n" for line in source.split('\n')]
        if lines:
            lines[-1] = lines[-1].strip('\n') # remove trailing newline on last line
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    add_markdown("# 🚁 Contagem de Público em Imagens de Drone (YOLOv8)\nEste notebook foi gerado automaticamente para rodar a aplicação completa no Google Colab.")
    
    add_markdown("## 1. Instalação de Dependências")
    add_code("!pip install ultralytics opencv-python-headless numpy matplotlib")
    
    add_markdown("## 2. Código da Aplicação (Módulos Integrados)")
    
    # We will read the local files and put them in a single code block
    modules = ['tiling.py', 'detector.py', 'nms.py', 'visualizer.py']
    app_code = "import cv2\nimport numpy as np\nfrom dataclasses import dataclass\nfrom typing import List, Tuple\nimport torch\nfrom ultralytics import YOLO\nimport matplotlib.pyplot as plt\n\n"
    
    for mod in modules:
        try:
            with open(mod, 'r', encoding='utf-8') as f:
                content = f.read()
                # strip out imports to avoid clutter, since we imported above
                lines = content.split('\n')
                clean_lines = []
                for line in lines:
                    if not (line.startswith('import ') or line.startswith('from ') or line.startswith('from typing') or line.startswith('from dataclasses')):
                        clean_lines.append(line)
                app_code += f"# --- {mod} ---\n"
                app_code += '\n'.join(clean_lines) + "\n\n"
        except FileNotFoundError:
            pass # skip if not found, we are in the right dir though
            
    add_code(app_code)
    
    add_markdown("## 3. Upload da Imagem\nExecute a célula abaixo para fazer o upload da imagem que deseja analisar.")
    
    upload_code = """from google.colab import files
import os

print("Selecione a imagem do mosaico do drone:")
uploaded = files.upload()

# Pega o nome do primeiro arquivo carregado
image_path = list(uploaded.keys())[0]
print(f"Imagem carregada com sucesso: {image_path}")"""
    add_code(upload_code)
    
    add_markdown("## 4. Execução do Pipeline\nO código abaixo executará o pipeline completo (Fatiamento, Detecção YOLO, NMS Global e Visualização).")
    
    exec_code = """# Parâmetros da aplicação
TILE_SIZE = 256
OVERLAP = 0.20
MODEL_WEIGHTS = "yolov8n.pt"
IMGSZ = 1280
CONF_THRESH = 0.30
IOU_LOCAL = 0.45
IOU_GLOBAL = 0.50
OUTPUT_PATH = "resultado_contagem.jpg"

print(f"[1/5] Carregando mosaico: {image_path}")
mosaic = cv2.imread(image_path)
if mosaic is None:
    raise FileNotFoundError("Erro ao carregar a imagem!")

print(f"      Resolução: {mosaic.shape[1]}x{mosaic.shape[0]} px")

print(f"[2/5] Fatiando em tiles {TILE_SIZE}px com {OVERLAP*100:.0f}% overlap...")
tiles = slice_mosaic(mosaic, tile_size=TILE_SIZE, overlap=OVERLAP)
print(f"      Total de tiles gerados: {len(tiles)}")

print(f"[3/5] Carregando modelo: {MODEL_WEIGHTS}")
model = YOLO(MODEL_WEIGHTS)

print(f"[4/5] Executando inferência (imgsz={IMGSZ}, conf={CONF_THRESH})...")
all_detections = []

for i, tile in enumerate(tiles, 1):
    dets = run_yolo_on_tile(
        model=model,
        tile=tile,
        conf_threshold=CONF_THRESH,
        imgsz=IMGSZ,
        iou_threshold=IOU_LOCAL,
    )
    all_detections.extend(dets)
    
print(f"\\n      Sub-total pré-NMS global: {len(all_detections)} caixas")

# 5. NMS global
final_detections = apply_global_nms(all_detections, iou_threshold=IOU_GLOBAL)
count = len(final_detections)
print(f"\\n[5/5] ✅ Contagem final: {count} pessoas detectadas")

# 6. Exporta e Visualiza
annotated = draw_detections(mosaic, final_detections)
export_results(annotated, count, output_path=OUTPUT_PATH)

print(f"Salvou o resultado em {OUTPUT_PATH}")"""
    add_code(exec_code)
    
    add_markdown("## 5. Visualizar o Resultado Final")
    show_code = """# Mostra a imagem com matplotlib
img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 15))
plt.imshow(img_rgb)
plt.axis('off')
plt.title(f"Pessoas Detectadas: {count}")
plt.show()

# Opcional: Download automático do resultado
# files.download(OUTPUT_PATH)"""
    add_code(show_code)
    
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "Aplicacao_Contagem_Drones.ipynb"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open('../Aplicacao_Contagem_Drones.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
        
    print("Notebook gerado em ../Aplicacao_Contagem_Drones.ipynb")

if __name__ == '__main__':
    create_colab_notebook()
