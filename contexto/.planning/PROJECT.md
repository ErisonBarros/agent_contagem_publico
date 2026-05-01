# Contagem de Público em Imagens Aéreas de Drones (YOLOv8 + Sliding Window)

## O que é isso
Um sistema em Python projetado para detectar e contar multidões em mosaicos georreferenciados gigantes gerados a partir de imagens de drones (UAV). Utiliza inferência do YOLOv8 e soluciona os problemas característicos de imagens aéreas — escala, variações de densidade e densidade irregular — fatiando o mosaico em blocos (Sliding Window Tiling) e filtrando duplicatas sobrepostas utilizando supressão não máxima (Global NMS).

## Objetivo Principal
Prover precisão confiável na contagem de pessoas em cenas densas ou espalhadas, rodando inferências de aprendizado de máquina em pedaços do mapa com overlap e unificando o resultado no mosaico global para gerar um produto visual final ou relatório.

## Core Value
Garantir **precisão de contagem e robustez** frente a diferentes densidades de multidões, escalando para mosaicos com milhares de pixels sem causar estouro de memória da GPU.

---

## Requirements

### Validated
- [x] Construir o módulo `tiling.py` que gere recortes paramétricos com sobreposição.
- [x] Construir o módulo `detector.py` integrado à biblioteca `ultralytics` para aplicar a inferência.
- [x] Mapear coordenadas locais do YOLOv8 para o espaço global do mosaico.

### Active
- [ ] Desenvolver a lógica de Global NMS em `nms.py` utilizando `cv2.dnn.NMSBoxes` para unificar bounding boxes repetidas nas bordas.
- [ ] Produzir saídas anotadas legíveis através de `visualizer.py` informando a quantidade contada no mosaico resultante.
- [ ] Integrar todos os módulos numa interface CLI (`main.py`) facilmente usável com hyper-parâmetros expostos (imgsz, tile_size, overal, etc).

### Out of Scope
- [ ] Aplicação de Interface Gráfica (Desktop/Web App) — *Foco apenas na ferramenta de backend CLI robusta*.
- [ ] Suporte inicial a vídeos — *Processamento exclusivo para imagens estáticas de alta resolução (Mosaicos)*.
- [ ] Treinamento customizado de rede YOLO — *Vamos utilizar inicialmente os modelos pré-treinados do YOLO (ex: yolov8n.pt, yolov8m.pt).*

---

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Uso de abordagem Tiling | Evita distorção das pessoas causada pelo resize automático do YOLO para grandes imagens. | — Pending |
| Supressão Não Máxima (NMS) Global baseada em IoU | Necessária para filtrar duplicatas nas margens sobrepostas geradas pelos mosaicos divididos. | — Pending |
| Apenas classe 'person' | Em imagens aéreas, outras categorias do COCO não geram valor para contagem de aglomeração; remove falso-positivos indesejados. | — Pending |

---

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-01 after initialization*
