# Diagramas de Fluxo e Arquitetura - Contagem de Público com YOLOv8

Abaixo estão os diagramas representando o funcionamento e a arquitetura do pipeline de detecção e contagem de multidões em mosaicos de drones descrito no documento `claude.md`.

## 1. Fluxo de Processamento Principal (Pipeline)

Este diagrama detalha o passo a passo da execução principal (do `main.py`), mostrando a transformação dos dados desde a entrada do mosaico até a exportação da imagem com a contagem final.

```mermaid
graph TD
    A[Mosaico UAV Original] -->|Carregamento| B[Módulo 1: Fatiamento]
    B -->|slice_mosaic| C[Lista de Tiles com Sobreposição]
    
    subgraph Inferência Local
        C --> D[Módulo 2: Detector YOLOv8]
        D -->|run_yolo_on_tile| E[Detecções em Coordenadas Locais]
        E -->|Mapeamento| F[Conversão para Coordenadas Globais]
    end
    
    F --> G[Merge de Detecções de todos os Tiles]
    G --> H[Módulo 3: NMS Global]
    H -->|apply_global_nms| I[Lista de Detecções Únicas]
    
    I --> J[Módulo 4: Visualizador]
    J -->|draw_detections| K[Desenho das Bounding Boxes]
    K -->|export_results| L[Imagem Final Anotada e Contagem]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style L fill:#bbf,stroke:#333,stroke-width:2px
```

## 2. Estrutura de Módulos e Classes

Este diagrama de classes ilustra como o código está organizado e quais estruturas de dados (como `Tile` e `Detection`) conectam as diferentes etapas.

```mermaid
classDiagram
    class main {
        +parse_args()
        +main()
    }
    
    class tiling {
        +slice_mosaic(mosaic, tile_size, overlap) List~Tile~
    }
    
    class Tile {
        +image: np.ndarray
        +x_offset: int
        +y_offset: int
    }
    
    class detector {
        +PERSON_CLASSES: dict
        +run_yolo_on_tile(model, tile, conf, imgsz, iou) List~Detection~
    }
    
    class Detection {
        +x1: float
        +y1: float
        +x2: float
        +y2: float
        +confidence: float
        +class_name: str
    }
    
    class nms {
        +apply_global_nms(detections, iou_threshold) List~Detection~
    }
    
    class visualizer {
        +draw_detections(mosaic, detections, max_display_size) np.ndarray
        +export_results(annotated, count, output_path)
    }

    main --> tiling : Utiliza
    main --> detector : Utiliza
    main --> nms : Utiliza
    main --> visualizer : Utiliza
    
    tiling ..> Tile : Instancia
    detector ..> Tile : Consome
    detector ..> Detection : Instancia
    nms ..> Detection : Filtra
    visualizer ..> Detection : Consome
```

## 3. Lógica da Janela Deslizante e NMS (Sliding Window Tiling)

Este diagrama explica de forma conceitual como os tiles são gerados com `overlap` (sobreposição) e por que o NMS (Non-Maximum Suppression) é necessário nas bordas.

```mermaid
graph LR
    subgraph Fatiamento
        M(Mosaico) --> T1(Tile 1)
        M --> T2(Tile 2)
        T1 -.->|Overlap| T2
    end
    
    subgraph Problema da Borda
        T1 --> D1[Detecção Pessoa X]
        T2 --> D2[Detecção Pessoa X]
        D1 -->|Duplicata| DP{Mesmo Indivíduo}
        D2 -->|Duplicata| DP
    end
    
    subgraph Solução NMS
        DP --> N[NMS Global baseado no IoU]
        N -->|Seleciona Maior Confiança| F[Detecção Única Pessoa X]
    end
```
