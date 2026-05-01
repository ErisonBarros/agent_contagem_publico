# Agente Contagem de Público (Aerial Crowd Counter)

Este projeto implementa um pipeline de Visão Computacional baseado em Inteligência Artificial para detectar e contar multidões em mosaicos de imagens capturadas por drones (UAVs). Utilizando a arquitetura YOLOv8 e técnicas de particionamento de imagem, o sistema supera os desafios de escala, densidade irregular e oclusão inerentes às imagens aéreas.

## Estrutura do Projeto

O projeto está organizado nas seguintes pastas principais:

### 1. `.agents/`
Contém as definições e instruções base do agente de Inteligência Artificial que idealizou e construiu a arquitetura do projeto.
* `agent.md`: Documento de diretrizes e regras de comportamento para a IA (foco em visão computacional aérea, configurações do YOLOv8 e OpenCV, e formatação técnica).

### 2. `.testes/`
Armazena os arquivos de testes, simulações em Jupyter Notebook e artefatos gerados nas fases de validação.
* `Aplicacao_Contagem_Drones.ipynb`: Notebook principal consolidado com as execuções e validações do sistema (rodando as rotinas de relatórios e contagem).
* `Simulacao_Fase1_Contagem.ipynb`: Notebook de simulação e teste da integração da Fase 1 (Tiling e Detecção).
* `Teste_Fase_1.md`: Relatório documentando a simulação inicial (Fase 1), testando o fatiamento (tiling) de imagens virtuais, warm-up do YOLOv8 e verificação de falsos positivos.

### 3. `contexto/`
É o núcleo principal da aplicação Python. Contém os módulos independentes da arquitetura e as documentações geradas.
* `main.py`: Script principal de orquestração do pipeline de contagem.
* `tiling.py`: Módulo responsável pelo fatiamento (Sliding Window Tiling) de imagens gigantes em tiles menores com sobreposição.
* `detector.py`: Módulo que executa o YOLOv8 em cada tile para a detecção de pessoas.
* `nms.py`: Aplica a Supressão Não-Máxima (NMS) global sobre as detecções, unificando sobreposições.
* `stats.py`: Gera análises estatísticas como histogramas de confiança e mapas de calor de densidade.
* `exporter.py`: Módulo de exportação dos dados brutos em formato tabular (CSV).
* `visualizer.py`: Desenha as caixas delimitadoras (*bounding boxes*) e dados analíticos sobre a imagem resultante.
* `report_generator.py`: Script de geração automatizada de relatórios técnicos contendo dados e gráficos.
* `build_notebook.py`: Utilitário para construir os scripts nativos do Python em um bloco Jupyter/Colab unificado.
* `test_phase1.py`: Código para rodar a bateria isolada de testes da Fase 1.

**Arquivos de Documentação (`*.md`) na pasta `contexto/`:**
* `claude.md`: Documentação técnica completa detalhando a estrutura, métodos de fatiamento de mosaicos, YOLOv8 e inferência para contagem de multidões, incluindo trechos de código base e sugestões de calibração.
* `contexto.md`: Contexto explicativo de negócio/tecnologia abordando os desafios de redimensionamento em imagens de alta resolução e a matemática implementada na solução.
* `diagramas.md`: Diagramas Mermaid ilustrando o fluxo principal (pipeline), diagrama de classes do sistema e o fluxo do problema e solução em zonas de *overlap* (NMS).
* `relatorio_tecnico.md`: Exemplo ou *template* estruturado do relatório analítico final gerado automaticamente, evidenciando contagens e estatísticas.

## Como Funciona

1. **Fatiamento em Tiles (Slicing):** O mosaico é dividido em recortes menores (*tiles*), com uma porcentagem de *overlap* (sobreposição), garantindo que alvos nas bordas não sejam cortados ou perdidos.
2. **Inferência Isolada:** Cada *tile* é submetido ao modelo de IA (YOLOv8n ou superior) treinado e configurado para filtrar restritamente classes do tipo pessoa/pedestre.
3. **Mapeamento (Remap):** As coordenadas geradas em cada tile recortado são convertidas (somadas ao *offset*) de volta para as dimensões absolutas do mosaico global.
4. **NMS Global (Merge):** Algoritmo *Non-Maximum Suppression* avalia o limiar *Intersection over Union* (IoU) nas áreas de sobreposição para assegurar que um mesmo indivíduo, detectado por dois tiles adjacentes, seja computado apenas uma vez.
5. **Relatório e Análises:** O sistema ilustra as marcações na imagem, exporta as coordenadas das detecções para CSV, desenha histogramas e heatmaps de precisão e finaliza criando um Relatório Técnico autônomo formatado em Markdown.

6. ## 👨‍💻 Autor

**Erison Barros**

[![ORCID](https://img.shields.io/badge/ORCID-0000--0003--4879--6880-a6ce39?logo=orcid&logoColor=white)](https://orcid.org/0000-0003-4879-6880)

Erison Barros
[GitHub: ErisonBarros] [ORCID: 0000-0003-4879-6880] [Linktree]


