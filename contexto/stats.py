"""
statistics.py
Gera gráficos estatísticos computacionais (Heatmap e Histograma) 
para agregar valor analítico e didático ao relatório final.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List
from detector import Detection

def generate_confidence_histogram(detections: List[Detection], output_path: str = "conf_histogram.png"):
    """
    Gera um histograma mostrando a distribuição das taxas de confiança das detecções.
    """
    confidences = [d.confidence for d in detections]
    
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribuição da Confiança (YOLOv8)', fontsize=14)
    plt.xlabel('Grau de Confiança', fontsize=12)
    plt.ylabel('Frequência (Quantidade de Pessoas)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Linha de média
    if confidences:
        mean_conf = np.mean(confidences)
        plt.axvline(mean_conf, color='red', linestyle='dashed', linewidth=2, label=f'Média: {mean_conf:.2f}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"      [Gráfico] Histograma salvo em {output_path}")

def generate_density_heatmap(mosaic_shape, detections: List[Detection], output_path: str = "density_heatmap.png"):
    """
    Gera um mapa de calor 2D mostrando as zonas de maior aglomeração.
    mosaic_shape: (height, width, channels)
    """
    h, w = mosaic_shape[:2]
    
    # Extrai o centroide de cada detecção
    x_centers = [(d.x1 + d.x2) / 2.0 for d in detections]
    y_centers = [(d.y1 + d.y2) / 2.0 for d in detections]
    
    # Define os bins para o heatmap (ex: grade 50x50 na imagem original)
    bins_x = 50
    bins_y = 50
    
    plt.figure(figsize=(8, 6))
    
    # Heatmap em 2D usando as coordenadas invertidas (Y no matplotlib cresce pra cima por padrão, 
    # mas em imagem cresce pra baixo, então temos que inverter o eixo Y)
    h2d, xedges, yedges, im = plt.hist2d(
        x_centers, 
        y_centers, 
        bins=[bins_x, bins_y], 
        range=[[0, w], [0, h]], 
        cmap='magma'
    )
    plt.gca().invert_yaxis()  # Para igualar à orientação da imagem
    
    plt.title('Mapa de Calor da Densidade de Público', fontsize=14)
    plt.xlabel('Eixo X (Pixels)', fontsize=12)
    plt.ylabel('Eixo Y (Pixels)', fontsize=12)
    plt.colorbar(label='Indivíduos por Célula da Grade')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"      [Gráfico] Mapa de calor salvo em {output_path}")
