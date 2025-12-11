import json
import cv2
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .face_detector import FaceDetector
from .preprocessing import calculate_iou

def match_predictions_to_ground_truth(gt_bboxes, pred_bboxes, iou_threshold=0.5):
    possible_matches = []
    for i, gt in enumerate(gt_bboxes):
        for j, pred in enumerate(pred_bboxes):
            iou = calculate_iou(gt, pred[:4]) 
            if iou >= iou_threshold:
                possible_matches.append((iou, i, j))
    
    possible_matches.sort(key=lambda x: x[0], reverse=True)
    
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    for iou, gt_idx, pred_idx in possible_matches:
        if gt_idx not in matched_gt_indices and pred_idx not in matched_pred_indices:
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(pred_idx)
            
    tp = len(matched_gt_indices)
    fp = len(pred_bboxes) - len(matched_pred_indices)
    fn = len(gt_bboxes) - len(matched_gt_indices)
    
    return tp, fp, fn, matched_gt_indices, matched_pred_indices

def draw_grouped_results(image, gt_bboxes, detections, matched_gt_indices, matched_pred_indices):
    img_copy = image.copy()
    
    # 1. Ground Truth (Verde=Ok, Laranja=Perdeu/FN)
    for i, bbox in enumerate(gt_bboxes):
        if i in matched_gt_indices:
            color = (0, 255, 0)   
        else:
            color = (0, 165, 255) # Laranja (FN)
            cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            # Sem texto, apenas a cor indica o erro

    # 2. Predições (Azul=Ok, Vermelho=Erro/FP)
    for i, det in enumerate(detections):
        x1, y1, x2, y2, score = det
        if i in matched_pred_indices:
            color = (255, 0, 0) # Azul
            # Não desenhamos a caixa azul se já tem a verde
        else:
            color = (0, 0, 255) # Vermelho (FP)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_copy, f"{score:.2f}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    return img_copy

def evaluate_detection_performance(model_path, test_set_json_path, iou_threshold=0.5, confidence_threshold=0.9):
    visuals_dir = "results/detection_visuals"
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Limpa arquivos anteriores
    for f in os.listdir(visuals_dir):
        os.remove(os.path.join(visuals_dir, f))

    print(f"\n--- AVALIAÇÃO DE DETECÇÃO (Salva apenas erros) ---")
    with open(test_set_json_path, 'r') as f:
        raw_data = json.load(f)

    images_db = defaultdict(list)
    for item in raw_data:
        images_db[item['image_path']].append(item['bbox'])
    
    detector = FaceDetector(model_path=model_path, confidence_threshold=confidence_threshold)    
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for image_path, gt_bboxes_list in tqdm(images_db.items(), desc="Avaliando"):
        # 1. Detectar
        _, detections = detector.detect(image_path)
        
        # 2. Match
        tp, fp, fn, match_gt_set, match_pred_set = match_predictions_to_ground_truth(
            gt_bboxes_list, detections, iou_threshold
        )
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # --- LÓGICA DE VELOCIDADE: ---
        # Se NÃO teve erro (não teve FN nem FP), PULA.
        if fn == 0 and fp == 0:
            continue

        # 3. Se teve erro, carrega imagem, desenha e salva
        image_color = cv2.imread(image_path)
        if image_color is None: continue

        annotated_image = draw_grouped_results(image_color, gt_bboxes_list, detections, match_gt_set, match_pred_set)
        
        # Salva apenas com o nome original (sem prefixos)
        base_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(visuals_dir, base_name), annotated_image)

    # Métricas Finais
    denom_precision = total_tp + total_fp
    denom_recall = total_tp + total_fn
    precision = total_tp / denom_precision if denom_precision > 0 else 0.0
    recall = total_tp / denom_recall if denom_recall > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "="*40)
    print(" RESULTADOS FINAIS ")
    print("="*40)
    print(f"Acertos (TP): {total_tp}")
    print(f"Erros de Falso Positivo (FP): {total_fp}")
    print(f"Erros de Não Detectado (FN): {total_fn}")
    print("-" * 40)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("="*40)
    print(f"Imagens contendo erros salvas em: {visuals_dir}")