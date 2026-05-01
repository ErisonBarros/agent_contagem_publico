# 🚁 Agente de Contagem de Público em Imagens de Drones
### Sistema de Visão Computacional com YOLOv8 + Sliding Window Tiling + NMS Global

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Visão Geral do Projeto

Este repositório implementa um **pipeline completo de Visão Computacional** para detectar e contar pessoas em **mosaicos de imagens de alta resolução** capturadas por drones (UAVs). O sistema foi projetado para superar desafios técnicos de escala que inviabilizariam a aplicação direta de detectores tradicionais.

### 🎯 Problema Resolvido

Imagens aéreas de drones em eventos de grande público chegam a ter **dezenas de milhares de pixels** de largura e altura. Modelos como YOLOv8, quando alimentados com imagens tão grandes, precisam redimensioná-las para 640×640 pixels, o que **esmaga e distorce** os objetos, tornando o reconhecimento de pedestres praticamente impossível.

### 💡 Solução Implementada

1. **Sliding Window Tiling**: A imagem é particionada em tiles sobrepostos de tamanho fixo (ex: 1280×1280).
2. **Inferência Isolada por Tile**: Cada tile é processado individualmente pelo YOLOv8.
3. **Remapeamento Global**: As coordenadas locais de cada tile são convertidas para as coordenadas absolutas do mosaico.
4. **NMS Global**: *Non-Maximum Suppression* elimina detecções duplicadas nas zonas de sobreposição entre tiles adjacentes.
5. **Relatório Técnico Automatizado**: Gráficos, estatísticas e relatório Markdown gerados automaticamente.

---

## 🗂️ Estrutura de Pastas

```
agent_contagem_publico/
│
├── README.md                    ← Este arquivo (visão geral do projeto)
│
├── contexto/                    ← 🧠 Núcleo principal da aplicação
│   ├── main.py                  ← Orquestrador principal (CLI)
│   ├── tiling.py                ← Módulo de Sliding Window Tiling
│   ├── detector.py              ← Detecção YOLOv8 por tile
│   ├── nms.py                   ← NMS Global (remoção de duplicatas)
│   ├── visualizer.py            ← Anotação visual da imagem final
│   ├── stats.py                 ← Estatísticas: Heatmap e Histograma
│   ├── exporter.py              ← Exportação de dados para CSV
│   ├── report_generator.py      ← Gerador automático de relatório técnico
│   ├── build_notebook.py        ← Utilitário para montar notebooks Colab
│   ├── test_phase1.py           ← Bateria de testes da Fase 1
│   │
│   ├── Prompts/                 ← Prompts usados para construção com IA
│   │
│   ├── .planning/               ← Documentação de planejamento (GSD)
│   │   ├── PROJECT.md           ← Especificação do projeto
│   │   └── ...                  ← Roadmap e planos de fase
│   │
│   ├── claude.md                ← Documentação técnica de referência
│   ├── contexto.md              ← Contexto de negócio e tecnologia
│   ├── diagramas.md             ← Diagramas Mermaid do sistema
│   └── relatorio_tecnico.md     ← Template/exemplo de relatório gerado
```

---

## 📄 Descrição dos Arquivos Principais

### 🐍 Módulos Python (`contexto/`)

#### `main.py` — Orquestrador do Pipeline (CLI)
Script principal que integra e coordena todo o pipeline. Aceita parâmetros via linha de comando:
- `--image`: Caminho para o mosaico de entrada.
- `--conf`: Limiar de confiança mínima (default: `0.3`).
- `--iou`: Limiar IoU para o NMS (default: `0.45`).
- `--tile`: Tamanho dos tiles em pixels (default: `1280`).
- `--overlap`: Porcentagem de sobreposição entre tiles (default: `0.2`).

**Uso:**
```bash
python main.py --image "imagem_multidao.jpg" --conf 0.3 --tile 1280
```

---

#### `tiling.py` — Sliding Window Tiling
Responsável por fatiar o mosaico de entrada em tiles menores com sobreposição configurável.

**Função principal:** `generate_tiles(image, tile_size, overlap)`

- Calcula automaticamente o *stride* (passo) com base no overlap configurado.
- Retorna uma lista de tuplas `(tile, x_offset, y_offset)` para rastreamento de posição.

---

#### `detector.py` — Detecção YOLOv8 por Tile
Executa o modelo YOLOv8 em cada tile individualmente e retorna as detecções com coordenadas locais.

**Função principal:** `detect_on_tiles(tiles, model, conf_threshold)`

- Carrega o modelo `yolov8n.pt` (ou modelo customizado especificado).
- Filtra detecções para a **classe 0** (pessoa/pedestre) exclusivamente.
- Retorna bounding boxes no formato `[x1_local, y1_local, x2_local, y2_local, confiança]`.

---

#### `nms.py` — NMS Global
Aplica a Supressão Não-Máxima nas coordenadas absolutas do mosaico global, eliminando detecções duplicadas geradas nas zonas de overlap.

**Função principal:** `apply_global_nms(all_detections, iou_threshold)`

- Converte detecções locais para coordenadas globais usando os offsets de cada tile.
- Utiliza `cv2.dnn.NMSBoxes` para a supressão eficiente.
- Garante que cada pessoa seja contada **uma única vez**, mesmo que apareça em tiles adjacentes.

---

#### `visualizer.py` — Anotação Visual
Desenha as bounding boxes, etiquetas de confiança e o contador total de pessoas sobre a imagem do mosaico.

**Função principal:** `annotate_image(image, final_detections, count)`

- Produz o arquivo `output_crowd.jpg` com todas as marcações visuais.
- Inclui o total de pessoas detectadas em destaque na imagem.

---

#### `stats.py` — Estatísticas e Gráficos
Gera os gráficos analíticos que comprovam a qualidade da detecção.

- **`generate_confidence_histogram(detections)`**: Histograma da distribuição de confiança (salvo como `conf_histogram.png`).
- **`generate_density_heatmap(image_shape, detections)`**: Mapa de calor de densidade espacial (salvo como `density_heatmap.png`). Visualiza as zonas de maior concentração de público.

---

#### `exporter.py` — Exportação CSV
Exporta os dados brutos de todas as detecções finais para um arquivo tabular.

**Função principal:** `export_to_csv(detections, filepath)`

- Arquivo de saída: `detections.csv`
- Colunas: `x1`, `y1`, `x2`, `y2`, `confiança`, `centro_x`, `centro_y`
- Útil para auditorias, validações externas e análises em planilhas.

---

#### `report_generator.py` — Gerador de Relatório Técnico
Consolida todos os resultados em um relatório Markdown estruturado e profissional.

**Função principal:** `generate_full_report(process_name, expert_name, image_path, count, ...)`

- Arquivo de saída: `relatorio_tecnico.md`
- Conteúdo do relatório:
  1. Identificação do processo e perito responsável.
  2. Resumo executivo com a contagem total.
  3. Parâmetros computacionais utilizados (conf, iou, tile_size).
  4. Imagem processada com anotações.
  5. Gráfico de histograma de confiança.
  6. Mapa de calor de densidade.
  7. Tabela com os dados de precisão estatística.

---

#### `build_notebook.py` — Construtor de Notebooks
Utilitário para converter os módulos Python do pipeline em um **Notebook Jupyter/Google Colab** unificado e auto-contido.

- Lê os arquivos `.py` do módulo.
- Gera um notebook `.ipynb` com as células na ordem correta do pipeline.
- Facilita demonstrações e execuções em ambientes de nuvem como o Google Colab.

---

#### `test_phase1.py` — Testes da Fase 1
Bateria de testes isolados para validação dos módulos `tiling.py` e `detector.py`.

- Testa a geração de tiles em imagens sintéticas.
- Verifica o warm-up do modelo YOLOv8.
- Confirma que detecções de falsos positivos são filtradas corretamente.

---

### 📝 Arquivos de Documentação Markdown (`contexto/*.md`)

#### `claude.md` — Documentação Técnica de Referência
Documentação técnica completa do sistema. Inclui:
- Arquitetura dos módulos e responsabilidades.
- Matemática por trás do Sliding Window Tiling (cálculo de stride e overlap).
- Fundamentação do uso do YOLOv8 para detecção de pedestres em perspectiva aérea.
- Guia de calibração dos hiperparâmetros (`conf`, `iou`, `tile_size`).

#### `contexto.md` — Contexto de Negócio e Tecnologia
Documento orientado para stakeholders e clientes. Aborda:
- O desafio de contar público em imagens de alta resolução.
- Por que o redimensionamento direto falha.
- A solução de tiling como abordagem robusta e escalável.
- Casos de uso: perícias judiciais, segurança pública, planejamento de eventos.

#### `diagramas.md` — Diagramas do Sistema (Mermaid)
Contém diagramas técnicos em formato Mermaid:
- **Diagrama de Fluxo do Pipeline:** Do mosaico de entrada até o relatório final.
- **Diagrama de Classes:** Relação entre os módulos Python.
- **Diagrama do Problema NMS:** Visualização das zonas de overlap e como o NMS resolve duplicatas.

#### `relatorio_tecnico.md` — Exemplo de Relatório Gerado
Template e exemplo de saída do relatório técnico automaticamente gerado pelo `report_generator.py`. Serve como referência de formato e estrutura.

---

## ⚙️ Pré-requisitos e Instalação

### Dependências Python
```bash
pip install ultralytics opencv-python-headless matplotlib seaborn numpy pandas
```

### Modelo YOLOv8
O modelo base `yolov8n.pt` é baixado automaticamente pelo Ultralytics na primeira execução. Para usar um modelo customizado treinado em pedestres de perspectiva aérea, forneça o caminho via parâmetro.

---

## 🚀 Execução Rápida

### Via CLI (Terminal)
```bash
# Processamento básico
python contexto/main.py --image "caminho/para/mosaico.jpg"

# Com parâmetros customizados
python contexto/main.py --image "multidao_drone.jpg" --conf 0.4 --tile 1280 --overlap 0.2
```

### Via Google Colab
Acesse o notebook `contexto/build_notebook.py` para gerar o `.ipynb` completo, ou utilize diretamente o notebook `.testes/Aplicacao_Contagem_Drones.ipynb`.

---

## 📊 Saídas Geradas

Após a execução, os seguintes arquivos são gerados automaticamente:

| Arquivo | Descrição |
|---|---|
| `output_crowd.jpg` | Imagem anotada com bounding boxes e contagem total |
| `detections.csv` | Dados brutos de cada detecção (coordenadas + confiança) |
| `conf_histogram.png` | Histograma da distribuição de confiança do modelo |
| `density_heatmap.png` | Mapa de calor espacial de densidade do público |
| `relatorio_tecnico.md` | Relatório técnico completo em Markdown |

---

## 📐 Arquitetura do Pipeline

```
[Mosaico .jpg de Alta Resolução]
         │
         ▼
┌─────────────────────┐
│   tiling.py         │  ← Sliding Window (tile_size, overlap)
│   Gera N tiles      │
└──────────┬──────────┘
           │  Lista de (tile, x_offset, y_offset)
           ▼
┌─────────────────────┐
│   detector.py       │  ← YOLOv8 (classe=0, conf > threshold)
│   Detecção por tile │
└──────────┬──────────┘
           │  Detecções com coordenadas locais
           ▼
┌─────────────────────┐
│   nms.py            │  ← Remap para coordenadas globais
│   NMS Global        │  ← cv2.dnn.NMSBoxes (iou_threshold)
└──────────┬──────────┘
           │  Detecções finais únicas
           ▼
┌──────────────────────────────────────────┐
│  visualizer.py  │  stats.py  │  exporter │
│  Anotação       │  Gráficos  │  CSV      │
└──────────┬───────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  report_generator   │  ← Relatório Técnico Markdown
│  .py                │
└─────────────────────┘
```

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, abra uma *issue* descrevendo a melhoria ou *bug* encontrado antes de submeter um *pull request*.

---

## 📜 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

*Desenvolvido com assistência de Inteligência Artificial (Google Antigravity / PeritoGeo AI) como ferramenta de suporte à perícia cartográfica e análise de multidões em imagens aéreas.*
