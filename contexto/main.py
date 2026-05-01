"""
main.py
Pipeline completo: carrega mosaico → fatia → detecta → NMS global → visualiza → relatório estatístico.
"""

import cv2
import argparse
from ultralytics import YOLO

from tiling import slice_mosaic
from detector import run_yolo_on_tile
from nms import apply_global_nms
from visualizer import draw_detections, export_results
from exporter import export_to_csv
from stats import generate_confidence_histogram, generate_density_heatmap
from report_generator import generate_full_report

def parse_args():
    parser = argparse.ArgumentParser(description="Crowd Counter UAV - YOLOv8")
    parser.add_argument("--image",   required=True,              help="Caminho do mosaico")
    parser.add_argument("--model",   default="yolov8n.pt",       help="Peso do modelo YOLO")
    parser.add_argument("--imgsz",   type=int,   default=736,    help="Resolução de inferência")
    parser.add_argument("--conf",    type=float, default=0.30,   help="Confiança mínima")
    parser.add_argument("--iou",     type=float, default=0.45,   help="IoU para NMS local")
    parser.add_argument("--gnms",    type=float, default=0.50,   help="IoU para NMS global")
    parser.add_argument("--tile",    type=int,   default=1280,   help="Tamanho do tile (px)")
    parser.add_argument("--overlap", type=float, default=0.20,   help="Sobreposição dos tiles")
    parser.add_argument("--output",  default="output_crowd.jpg", help="Caminho da saída")
    parser.add_argument("--csv",     default="detections.csv",   help="Caminho exportação CSV")
    parser.add_argument("--event",   default="Inspeção Aérea de Área Pública", help="Nome do evento para o relatório")
    parser.add_argument("--analyst", default="PeritoGeo AI",     help="Nome do analista para o relatório")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Carrega mosaico
    print(f"[1/6] Carregando mosaico: {args.image}")
    mosaic = cv2.imread(args.image)
    if mosaic is None:
        raise FileNotFoundError(f"Imagem não encontrada: {args.image}")
    print(f"      Resolução: {mosaic.shape[1]}x{mosaic.shape[0]} px")

    # 2. Fatia em tiles
    print(f"[2/6] Fatiando em tiles {args.tile}px com {args.overlap*100:.0f}% overlap...")
    tiles = slice_mosaic(mosaic, tile_size=args.tile, overlap=args.overlap)
    print(f"      Total de tiles gerados: {len(tiles)}")

    # 3. Carrega modelo YOLOv8
    print(f"[3/6] Carregando modelo: {args.model}")
    model = YOLO(args.model)

    # 4. Inferência por tile
    print(f"[4/6] Executando inferência (imgsz={args.imgsz}, conf={args.conf})...")
    all_detections = []

    for i, tile in enumerate(tiles, 1):
        dets = run_yolo_on_tile(
            model=model,
            tile=tile,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            iou_threshold=args.iou,
        )
        all_detections.extend(dets)
        print(f"      Tile {i:03d}/{len(tiles)} -> {len(dets):3d} detecções", end="\r")

    print(f"\n      Sub-total pré-NMS global: {len(all_detections)} caixas")

    # 5. NMS global
    final_detections = apply_global_nms(all_detections, iou_threshold=args.gnms)
    count = len(final_detections)
    print(f"\n[5/6] ✅ Contagem final: {count} pessoas detectadas")

    # 6. Exporta resultado visual
    annotated = draw_detections(mosaic, final_detections)
    export_results(annotated, count, output_path=args.output)

    # 7. Relatórios e Estatísticas
    print("\n[6/6] 📊 Gerando dados analíticos e relatório técnico...")
    
    # Exporta CSV
    export_to_csv(final_detections, output_csv=args.csv)
    
    # Gera Gráficos
    heatmap_name = "density_heatmap.png"
    hist_name = "conf_histogram.png"
    generate_confidence_histogram(final_detections, output_path=hist_name)
    generate_density_heatmap(mosaic.shape, final_detections, output_path=heatmap_name)
    
    # Gera Relatório Markdown
    report_name = "relatorio_tecnico.md"
    generate_full_report(
        process_name=args.event,
        expert_name=args.analyst,
        total_count=count,
        img_shape=mosaic.shape,
        param_conf=args.conf,
        param_iou=args.gnms,
        report_path=report_name,
        output_img_name=args.output,
        heatmap_img=heatmap_name,
        hist_img=hist_name,
        export_csv=args.csv
    )
    
    print(f"\n      Processo finalizado com sucesso! Arquivos salvos no diretório atual.")

if __name__ == "__main__":
    main()
