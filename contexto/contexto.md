# Contexto Inicial

A contagem de multidões (*Crowd Counting*) em imagens aéreas (Veículos Aéreos Não Tripulados - UAVs) apresenta desafios significativos, como variações de escala, densidade irregular e oclusão. A arquitetura YOLO (You Only Look Once) utiliza uma rede neural convolucional de passada única (*single-shot detection*), sendo considerada o estado da arte por oferecer alta velocidade e precisão na detecção e contagem de múltiplos objetos.

---

# Estratégia de Detecção para Mosaicos Aéreos

A detecção de pequenos alvos em mosaicos aéreos de alta resolução apresenta um gargalo fundamental: ao redimensionar uma imagem massiva (ex: 8000x8000 pixels) para a entrada padrão de uma rede neural (ex: 640x640), a perda de informação degrada a assinatura visual das pessoas, tornando-as indetectáveis. 

Para solucionar este problema de *Crowd Counting*, adotamos a técnica de **Slicing Aided Inference** utilizando OpenCV e YOLOv8. A estratégia consiste em:

1. **Fatiamento (Tiling) com Sobreposição (Overlap):** Dividimos o mosaico em blocos menores (*tiles*). A sobreposição é crucial para garantir que pessoas cortadas na borda de um bloco sejam detectadas integralmente no bloco adjacente.
2. **Inferência Isolada e Filtragem de Classes:** Cada bloco passa pelo YOLOv8. Configuramos o modelo para focar exclusivamente na classe "pessoa" (ID 0 no dataset COCO), que já consolida conceitos de pedestres e indivíduos.
3. **Mapeamento de Coordenadas:** As coordenadas das caixas delimitadoras (*bounding boxes*) de cada bloco são transladadas de volta para o referencial de coordenadas global do mosaico original.
4. **Supressão Máxima Não-Linear (Global NMS):** Como a sobreposição de blocos gera detecções duplicadas de um mesmo indivíduo, aplicamos o algoritmo de *Non-Maximum Suppression* baseando-se na métrica de Interseção sobre União (IoU) para unificar as caixas sobrepostas.

---

# Implementação em Python

Abaixo apresento um código modular e pragmático utilizando `opencv-python`, `ultralytics` e `torchvision` (para o NMS global).

```python
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision

class AerialCrowdCounter:
    def __init__(self, model_path='yolov8n.pt', tile_size=736, overlap=0.2):
        """
        Inicializa o pipeline de detecção.
        :param model_path: Caminho para os pesos do modelo YOLOv8. Recomenda-se yolov8n.pt para velocidade ou yolov8x.pt para máxima precisão.
        :param tile_size: Tamanho do bloco fatiado. O mesmo valor será usado no parâmetro `imgsz` da inferência.
        :param overlap: Taxa de sobreposição entre os blocos (0.0 a 1.0).
        """
        # Carrega o modelo pré-treinado
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        
    def _slice_image(self, image):
        """ Gira sobre a imagem gerando coordenadas para fatiamento. """
        h, w = image.shape[:2]
        step = int(self.tile_size * (1 - self.overlap))
        
        tiles = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Garante que o bloco não ultrapasse as bordas da imagem original
                y1, x1 = y, x
                y2, x2 = min(y + self.tile_size, h), min(x + self.tile_size, w)
                
                # Ajusta a origem caso estejamos na borda para manter o tamanho do tile
                if y2 - y1 < self.tile_size: y1 = max(0, y2 - self.tile_size)
                if x2 - x1 < self.tile_size: x1 = max(0, x2 - self.tile_size)
                
                tiles.append({
                    'image': image[y1:y2, x1:x2],
                    'offset': (x1, y1)
                })
        return tiles

    def count_crowd(self, image_path, conf_thresh=0.25, iou_thresh=0.45, output_path='output_mosaic.jpg'):
        """
        Executa a inferência no mosaico e gera a contagem final.
        """
        # 1. Pré-processamento com OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem em {image_path}")
            
        tiles = self._slice_image(img)
        
        global_boxes = []
        global_scores = []
        
        # 2 e 3. Configuração YOLOv8 e Otimização de Classes
        # Iteramos sobre os blocos e fazemos a inferência
        for tile in tiles:
            tile_img = tile['image']
            offset_x, offset_y = tile['offset']
            
            # Ajuste de Hiperparâmetros: imgsz otimizado para o tamanho do tile
            # classes=[0] restringe a detecção exclusivamente à classe "pessoa"
            results = self.model.predict(
                source=tile_img,
                imgsz=self.tile_size,
                conf=conf_thresh,
                classes=[0], 
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
                scores = result.boxes.conf.cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    # Correção de coordenadas para o referencial do mosaico inteiro
                    x1, y1, x2, y2 = box
                    global_boxes.append([x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y])
                    global_scores.append(score)
                    
        # 4. Processamento de Resultados (NMS Global)
        if not global_boxes:
            print("Nenhuma pessoa detectada.")
            return 0
            
        boxes_tensor = torch.tensor(global_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(global_scores, dtype=torch.float32)
        
        # Aplica NMS para remover bounding boxes duplicadas oriundas da sobreposição dos blocos
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_thresh)
        
        final_boxes = boxes_tensor[keep_indices].numpy()
        final_scores = scores_tensor[keep_indices].numpy()
        
        # 5. Desenhar Bounding Boxes e Metadados
        final_count = len(final_boxes)
        for box, score in zip(final_boxes, final_scores):
            x1, y1, x2, y2 = map(int, box)
            # Desenha a caixa
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Insere a confiança da detecção
            label = f"{score:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Adiciona a contagem total no topo da imagem
        cv2.putText(img, f"Total Count: {final_count} people", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imwrite(output_path, img)
        print(f"Processamento concluído. {final_count} pessoas detectadas.")
        print(f"Imagem salva em: {output_path}")
        
        return final_count

# Exemplo de Uso
if __name__ == "__main__":
    # Instancia a classe otimizada para tiles de 736x736
    counter = AerialCrowdCounter(model_path='yolov8n.pt', tile_size=736, overlap=0.2)
    
    # Executa a contagem
    counter.count_crowd(
        image_path='mosaico_drone.jpg',
        conf_thresh=0.30, # Limiar de confiança (ajustável)
        iou_thresh=0.40,  # Limiar de NMS (ajustável)
        output_path='resultado_contagem.jpg'
    )
```

---

# Sugestões para Calibração de Hiperparâmetros

A geometria das imagens aéreas impõe desafios únicos que exigem ajustes rigorosos baseados na densidade da multidão e na altitude do voo:

### 1. Tamanho do Tile e da Imagem (`tile_size` / `imgsz`)
* **Multidões Extremamente Densas ou Altitude Alta:** Aumente o `tile_size` (ex: `1024` ou `1280`). Redes YOLO retêm mais características locais em resoluções mais altas. Garanta que a memória RAM/VRAM do sistema suporte este incremento.
* **Baixa Altitude:** Valores como `640` ou `736` oferecem um excelente balanço entre velocidade e precisão.

### 2. Limiar de Confiança (`conf_thresh`)
Este parâmetro define quão "seguro" o modelo deve estar de que o padrão detectado é, de fato, uma pessoa.
* **Baixa Oclusão (Pessoas espalhadas):** Utilize valores mais altos (ex: `0.40` a `0.50`) para evitar Falsos Positivos (detectar lixeiras, postes ou hidrantes como pessoas).
* **Alta Oclusão (Aglomerações intensas):** Reduza o valor (ex: `0.15` a `0.25`). Em multidões, muitas vezes apenas a cabeça ou parte do ombro está visível. Modelos tendem a ter baixa confiança nesses casos, logo, penalizar baixas confianças pode causar muitos Falsos Negativos.

### 3. Limiar de IoU no NMS (`iou_thresh`)
O limiar de Interseção sobre União determina o rigor para fundir caixas sobrepostas no NMS.
* **Aglomerações Intensas:** Reduza o IoU (ex: `0.30` a `0.40`). Em uma multidão densa, caixas de pessoas distintas irão naturalmente se sobrepor. Um IoU muito alto (ex: `0.70`) fundirá duas pessoas coladas em uma só.
* **Pessoas Esparsas:** Aumente o IoU (ex: `0.50` a `0.60`) para garantir que a sobreposição gerada pelos recortes (tiles) do script seja rigidamente suprimida, mantendo uma única caixa por indivíduo.
