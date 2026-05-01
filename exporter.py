"""
exporter.py
Responsável por exportar os dados analíticos das detecções para CSV ou JSON.
"""

import csv
from typing import List
from detector import Detection

def export_to_csv(detections: List[Detection], output_csv: str):
    """
    Exporta a lista de detecções para um arquivo CSV.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Cabeçalho
        writer.writerow(["id", "x1", "y1", "x2", "y2", "center_x", "center_y", "confidence", "class"])
        
        for i, d in enumerate(detections, 1):
            center_x = (d.x1 + d.x2) / 2.0
            center_y = (d.y1 + d.y2) / 2.0
            writer.writerow([
                i,
                f"{d.x1:.2f}",
                f"{d.y1:.2f}",
                f"{d.x2:.2f}",
                f"{d.y2:.2f}",
                f"{center_x:.2f}",
                f"{center_y:.2f}",
                f"{d.confidence:.4f}",
                d.class_name
            ])
    
    print(f"      [CSV] Exportou {len(detections)} detecções para {output_csv}")
