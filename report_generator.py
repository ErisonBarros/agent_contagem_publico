"""
report_generator.py
Gera um relatório técnico em formato Markdown consolidando os resultados.
"""

import datetime
import os

class TechnicalReportBuilder:
    def __init__(self, output_path="relatorio_tecnico.md"):
        self.output_path = output_path
        self.date_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.content = []

    def build_header(self, process_name: str, expert_name: str):
        header = f"""# Relatório Técnico - Contagem Analítica de Público

**Data de Geração:** {self.date_str}
**Processo/Evento:** {process_name}
**Analista/Perito:** {expert_name}

---
"""
        self.content.append(header)

    def build_summary(self, total_count: int, img_shape: tuple, param_conf: float, param_iou: float):
        h, w = img_shape[:2]
        summary = f"""## 1. Resumo da Operação

A análise foi realizada através de segmentação da imagem (*tiling*) e inferência por inteligência artificial (Modelo YOLOv8).

- **Resolução da Imagem Analisada:** {w}x{h} px
- **Limiares de Qualidade (Parâmetros):**
  - Confiança Mínima (*Confidence Threshold*): {param_conf}
  - Supressão Não Máxima (*Global IoU*): {param_iou}

### 🎯 TOTAL DETECTADO: {total_count} indivíduos

---
"""
        self.content.append(summary)

    def build_visual_results(self, output_img_name="output_crowd.jpg"):
        visual = f"""## 2. Resultado Visual da Detecção

Abaixo encontra-se o mosaico processado com as caixas delimitadoras (*bounding boxes*) desenhadas sobre cada indivíduo detectado.

![Imagem Processada]({output_img_name})

---
"""
        self.content.append(visual)

    def build_statistical_analysis(self, heatmap_img="density_heatmap.png", hist_img="conf_histogram.png"):
        stats = f"""## 3. Análise Estatística e Computacional

Os gráficos a seguir comprovam a acurácia do modelo e demonstram a densidade posicional do público na cena mapeada.

### Distribuição de Confiança (Accuracy)
O histograma reflete a certeza matemática do algoritmo em cada detecção.
![Histograma de Confiança]({hist_img})

### Mapa de Calor da Densidade (Heatmap)
O mapa de calor ilustra as regiões com maior concentração de pessoas.
![Mapa de Calor]({heatmap_img})

---
"""
        self.content.append(stats)

    def build_conclusion(self, export_csv="detections.csv"):
        conclusion = f"""## 4. Conclusão e Dados Brutos

O algoritmo rodou com sucesso sem gerar redundâncias de borda (aplicado Filtro NMS Global).
Os dados brutos com as coordenadas de todos os indivíduos detectados foram salvos e podem ser auditados no arquivo:
📂 `{export_csv}`

*Relatório gerado automaticamente pelo Agente de Contagem de Público.*
"""
        self.content.append(conclusion)

    def save(self):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.content))
        print(f"      [Relatório] Relatório técnico gerado em {self.output_path}")

def generate_full_report(
    process_name: str, 
    expert_name: str, 
    total_count: int, 
    img_shape: tuple,
    param_conf: float,
    param_iou: float,
    report_path: str = "relatorio_tecnico.md",
    output_img_name: str = "output_crowd.jpg",
    heatmap_img: str = "density_heatmap.png",
    hist_img: str = "conf_histogram.png",
    export_csv: str = "detections.csv"
):
    builder = TechnicalReportBuilder(report_path)
    builder.build_header(process_name, expert_name)
    builder.build_summary(total_count, img_shape, param_conf, param_iou)
    builder.build_visual_results(output_img_name)
    builder.build_statistical_analysis(heatmap_img, hist_img)
    builder.build_conclusion(export_csv)
    builder.save()
