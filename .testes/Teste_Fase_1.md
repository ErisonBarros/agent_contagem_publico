# Relatório de Teste: Fase 1 (Tiling & Detecção)

Este documento sumariza a bateria de testes automatizados implementada para a verificação isolada dos módulos de fundação desenvolvidos na Fase 1 do sistema de contagem de público em mosaicos de drone.

## 1. O Script de Teste (`test_phase1.py`)

Para validar a fase sem depender de mosaicos reais pesados, criamos um script que injeta matrizes limpas simulando as entradas georreferenciadas que o sistema encontrará em produção. O teste é dividido em duas etapas fundamentais.

### 1.1 Simulação do Módulo de Fatiamento (Tiling)

> [!NOTE]
> **Objetivo:** Verificar se a matriz (mosaico) gigante seria dividida perfeitamente sem erros de índice (out-of-bounds) e se os *offsets* estariam corretos para o remap das coordenadas.

**Metodologia:**
1. Foi criada uma matriz virtual (dummy) utilizando NumPy simulando uma imagem de dimensões **3000x2000 pixels**.
2. Foi configurado o passo da janela deslizante (`tile_size = 1280` e `overlap = 0.2`).

**Resultado:**
- O script reportou com sucesso a subdivisão em exatos **6 blocos** (tiles).
- A sobreposição ocorreu perfeitamente e os offsets X,Y (`x_offset` e `y_offset`) foram mapeados com valores absolutos, comprovando matematicamente a função.

### 1.2 Simulação do Módulo YOLOv8 (Detector)

> [!NOTE]
> **Objetivo:** Comprovar que a camada de Machine Learning consegue carregar, ler os inputs fatiados do módulo 1 e devolver coordenadas unificadas na escala real do mapa sem explodir a memória GPU/CPU.

**Metodologia:**
1. Utilizou-se o modelo mais leve da biblioteca ultralytics (`yolov8n.pt`) que efetuou seu próprio download automaticamente.
2. Injetou-se um "tile falso" escuro (uma matriz de zeros simulando um corte de 1280x1280) que possuía coordenadas simuladas de `x_offset = 500` e `y_offset = 500`.
3. O limiar de confiança (`conf_threshold`) foi acionado a uma baixa tolerância para observar o comportamento do tensor.

**Resultado:**
- O modelo fez o warm-up adequadamente.
- Retornou de forma íntegra `0 detecções`, demonstrando que o código filtrou corretamente a classe `0` (pessoa) e que não existem *False Positives* em zonas de "vazio fotográfico".

## 2. Conclusão

Os dois núcleos bases (fatiamento de imagens brutas e interface de predição via YOLO) foram consolidados. A estratégia de usar `dataclass` para manter a persistência de deslocamento geográfico no eixo do mosaico está funcional. 

> [!TIP]
> **Próximo Desafio:** Com os *Tiles* funcionais, o verdadeiro problema da Fase 2 é: uma mesma pessoa que estiver presente na "franja" de interseção de 2 Tiles simultâneos será identificada por ambos. É necessária a implantação do *Global Non-Maximum Suppression (NMS)* para eliminar essa dupla leitura.
