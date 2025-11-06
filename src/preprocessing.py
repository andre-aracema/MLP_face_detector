import os
import glob
import pandas as pd
import random
import cv2
import numpy as np
import dlib
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

# Calcula a Interseção sobre União (IoU) de duas caixas
def calculate_iou(box1: List[int], box2: List[int]) -> float:

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = float(box1_area + box2_area - intersection_area)

    return (intersection_area / union_area) if union_area > 0 else 0.0

# Vira horizontalmente a imagem
def image_flip(image: np.ndarray) -> np.ndarray:

    return cv2.flip(image, 1)

# Aplica brilho aleatório e mudança de contraste
def image_brightness_contrast(
    image: np.ndarray,
    alpha_range: Tuple[float, float] = (0.8, 1.2), 
    beta_range: Tuple[int, int] = (-20, 20)
) -> np.ndarray:

    alpha = random.uniform(alpha_range[0], alpha_range[1])
    beta = random.randint(beta_range[0], beta_range[1])

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

"""--------------------------- Funções Auxiliares da Geração dos Dados ------------------------------------"""

# Aplica aumentações padrão em uma amostra de face (acrescenta)
def _get_face_augmentations(normalized_face: np.ndarray) -> List[np.ndarray]:
    
    return [
        normalized_face,
        image_flip(normalized_face),
        image_brightness_contrast(normalized_face)
    ]

# Extrai, redimensiona e processa uma única amostra de face
def _process_face_sample(image_gray: np.ndarray, face_bbox: List[int], clahe: cv2.CLAHE, window_size: tuple) -> List[np.ndarray]:
    
    x1, y1, x2, y2 = face_bbox
    face_crop = image_gray[y1:y2, x1:x2]
    
    if face_crop.size == 0:
        return []

    resized_face = cv2.resize(face_crop, window_size, interpolation=cv2.INTER_AREA)
    face_normalized = clahe.apply(resized_face)

    return _get_face_augmentations(face_normalized)

# Gera múltiplas amostras de não-face de uma imagem
def _generate_non_face_samples(
    image_gray: np.ndarray, face_bbox: List[int], window_size: tuple,
    iou_threshold: float, clahe: cv2.CLAHE, max_attempts: int = 10
) -> List[np.ndarray]:

    samples = []
    img_h, img_w = image_gray.shape
    
    if not (img_w >= window_size[0] and img_h >= window_size[1]):
        return []

    for _ in range(max_attempts):
        rand_x1 = random.randint(0, img_w - window_size[0])
        rand_y1 = random.randint(0, img_h - window_size[1])
        random_box = [rand_x1, rand_y1, rand_x1 + window_size[0], rand_y1 + window_size[1]]

        if calculate_iou(random_box, face_bbox) < iou_threshold:
            crop = image_gray[random_box[1]:random_box[3], random_box[0]:random_box[2]]
            samples.append(clahe.apply(crop))
            
    return samples

# Processa uma imagem para faces e não-faces
def _process_image_worker(
    image_info: Dict[str, Any],
    window_size: Tuple[int, int], iou_threshold_neg: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    try:
        image_color = cv2.imread(image_info['image_path'])
        if image_color is None: return [], []

        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        face_bbox = image_info['bbox']
        
        face_samples = _process_face_sample(
            image_gray, face_bbox, clahe, window_size
        )
        non_face_samples = _generate_non_face_samples(
            image_gray, face_bbox, window_size, iou_threshold_neg, clahe
        )

        return face_samples, non_face_samples
        
    except Exception:
        return [], []

"""--------------------------- Funções auxiliares da função Principal -------------------------------------"""

# Lê um arquivo .csv e retorna uma lista de infos
def _parse_annotation_file(file_path: str, base_path: str) -> List[Dict[str, Any]]:

    annotations = []

    try:
        df_file = pd.read_csv(file_path)

        for _, row in df_file.iterrows():
            image_path = os.path.join(base_path, row['filename'])
            if os.path.exists(image_path):
                bbox = [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])]
                annotations.append({'image_path': image_path, 'bbox': bbox})

    except Exception as e:
        print(f"Warning: Error reading file {file_path}: {e}")

    return annotations

# Encontra todos os arquivos de anotação de forma paralela
def _load_annotations(base_paths: List[str]) -> List[Dict[str, Any]]:

    all_dataset_info = []

    print("Lendo arquivos de anotações.txt ...")

    for base_path in base_paths:
        annotation_files = glob.glob(os.path.join(base_path, '*_annotations.txt'))
        for file_path in annotation_files:
            all_dataset_info.extend(_parse_annotation_file(file_path, base_path))

    return all_dataset_info

# Embaralha e seleciona um subconjunto de amostras para processar
def _select_samples_to_process(all_info: List[Dict], num_samples: int) -> List[Dict]:

    random.shuffle(all_info)
    num_to_process = min(num_samples, len(all_info))

    return all_info[:num_to_process]

#Agrega os resultados paralelos e salva em disco
def _aggregate_and_save_results(results: List[tuple], save_path: str | None) -> Tuple[np.ndarray, np.ndarray]:

    face_samples, non_face_samples = [], []

    for local_faces, local_non_faces in results:
        face_samples.extend(local_faces)
        non_face_samples.extend(local_non_faces)

    num_faces = len(face_samples)
    num_non_faces = len(non_face_samples)

    if num_non_faces > num_faces and num_faces > 0:
        print(f"Balanceando dataset: Reduzindo {num_non_faces} não-faces para {num_faces} (1:1)...")
        random.shuffle(non_face_samples)
        non_face_samples = non_face_samples[:num_faces]
    
    elif num_faces > num_non_faces and num_non_faces > 0:
        print(f"Balanceando dataset: Reduzindo {num_faces} faces para {num_non_faces} (1:1)...")
        random.shuffle(face_samples)
        face_samples = face_samples[:num_non_faces]
        
    face_array = np.array(face_samples)
    non_face_array = np.array(non_face_samples)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(f"{save_path}_face.npy", face_array)
        np.save(f"{save_path}_non_face.npy", non_face_array)
        print(f"Dados salvos em {save_path}_[face/non_face].npy")

    return face_array, non_face_array

"""---------------------------------------------------------------------------------------------------------"""

# Orquestra o pipeline de geração de dados
def generate_data(
    base_paths: List[str], 
    num_samples: int, 
    window_size: Tuple[int, int] = (32, 32), 
    iou_threshold_neg: float = 0.1,
    save_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
  
    all_info = _load_annotations(base_paths)
    dataset_to_process = _select_samples_to_process(all_info, num_samples)

    print("Começando processamento paralelo ...")

    results = Parallel(n_jobs=-1)(
        delayed(_process_image_worker)(
            image_info=info,
            window_size=window_size,
            iou_threshold_neg=iou_threshold_neg
        )
        for info in tqdm(dataset_to_process, desc="Processing Images")
    )

    face_array, non_face_array = _aggregate_and_save_results(results, save_path)
    
    print(f"\nProcesso completo!")
    print(f"Total de amostras geradas: {len(face_array)} faces, {len(non_face_array)} non-faces.")
    
    return face_array, non_face_array