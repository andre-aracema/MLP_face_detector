import os
import glob
import pandas as pd
import random
import cv2
import numpy as np
import dlib
from joblib import Parallel, delayed
from tqdm import tqdm


"""-------------------------------- Funções Auxiliares -----------------------------------------------------"""

# Calcula a Interseção sobre União (IoU) de duas caixas
def calculate_iou(box1, box2):
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
def image_flip(image):
    return cv2.flip(image, 1)

# Aplica brilho aleatório e mudança de contraste
def image_brightness_contrast(image, alpha_range= (0.8, 1.2), beta_range= (-20, 20)):
    alpha = random.uniform(alpha_range[0], alpha_range[1])
    beta = random.randint(beta_range[0], beta_range[1])

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def image_slight_rotation(image, angle_range=(-5, 5)):
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # usa borda replicada para evitar bordas pretas
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

def image_slight_translation(image, trans_range=(-2, 2)):
    x_trans = random.randint(trans_range[0], trans_range[1])
    y_trans = random.randint(trans_range[0], trans_range[1])
    matrix = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
    
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)


"""--------------------------- Funções Auxiliares da Geração dos Dados ------------------------------------"""

# alinha as amostra que já foram recortadas
def align_face(image, predictor, window_size):

    try:
        face = dlib.rectangle(0, 0, image.shape[1], image.shape[0])
        landmarks = predictor(image, face)
        left_eye_pts = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye_pts = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        
        left_eye_center = np.mean(left_eye_pts, axis=0).astype("int")
        right_eye_center = np.mean(right_eye_pts, axis=0).astype("int")

        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = 0.3 * window_size[0] 
        scale = desired_dist / dist

        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                    (left_eye_center[1] + right_eye_center[1]) // 2)

        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        tX = window_size[0] * 0.5 - eyes_center[0]
        tY = window_size[1] * 0.4 - eyes_center[1]
        M[0, 2] += tX
        M[1, 2] += tY
        (w, h) = window_size
        aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return aligned_face
    
    except Exception:

        return None

# Aplica aumentações padrão em uma amostra de face (acrescenta)
def _get_face_augmentations(normalized_face):
    return [
        normalized_face,
        image_flip(normalized_face),
        image_brightness_contrast(normalized_face),
        image_slight_rotation(normalized_face),
        image_slight_translation(normalized_face)
    ]

# Extrai, redimensiona e processa uma única amostra de face
def _process_face_sample(image_gray_normalized, face_bbox, window_size, predictor, clahe):
    x1, y1, x2, y2 = face_bbox
    face_crop = image_gray_normalized[y1:y2, x1:x2]
    
    if face_crop.size == 0:
        return []
    
    aligned_face =  align_face(face_crop, predictor, window_size) 

    if aligned_face is None:
        aligned_face = cv2.resize(face_crop, window_size, interpolation=cv2.INTER_AREA)

    image_norm = clahe.apply(aligned_face)

    return _get_face_augmentations(image_norm)

# Gera múltiplas amostras de não-face de uma imagem
def _generate_non_face_samples(image_gray, face_bbox, window_size, iou_threshold, clahe, target_count, max_attempts= 20):

    image_gray_normalized = clahe.apply(image_gray)
    samples = []
    img_h, img_w = image_gray_normalized.shape 
    
    if not (img_w >= window_size[0] and img_h >= window_size[1]):
        return []

    attempts = 0
    # Continua até atingir o alvo ou o máximo de tentativas
    while len(samples) < target_count and attempts < max_attempts:
        attempts += 1
        rand_x1 = random.randint(0, img_w - window_size[0])
        rand_y1 = random.randint(0, img_h - window_size[1])
        random_box = [rand_x1, rand_y1, rand_x1 + window_size[0], rand_y1 + window_size[1]]

        if calculate_iou(random_box, face_bbox) < iou_threshold:
            crop = image_gray_normalized[random_box[1]:random_box[3], random_box[0]:random_box[2]]
            samples.append(crop)
            
    return samples

# Processa uma imagem para faces e não-faces
def _process_image_worker(image_info, window_size, iou_threshold_neg):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    try:
        predictor = dlib.shape_predictor("landMarks/shape_predictor_68_face_landmarks.dat")
    except Exception:
        print("Erro ao carregar shape_predictor_68_face_landmarks.dat. Certifique-se de que o arquivo está no diretório correto.")

    try:
        image_color = cv2.imread(image_info['image_path'])
        if image_color is None:
            return [], []

        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        face_bbox = image_info['bbox']
        
        face_samples = _process_face_sample(image_gray, face_bbox, window_size, predictor, clahe)
        num_faces_generated = len(face_samples)

        non_face_samples = _generate_non_face_samples(image_gray, face_bbox, window_size, iou_threshold_neg, clahe, target_count=num_faces_generated)

        return face_samples, non_face_samples
        
    except Exception:
        return [], []

# Lê um arquivo .csv e retorna uma lista de infos
def _parse_annotation_file(file_path, base_path):
    annotations = []

    try:
        df_file = pd.read_csv(file_path)

        for _, row in df_file.iterrows():
            image_path = os.path.join(base_path, row['filename'])
            if os.path.exists(image_path):
                bbox = [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])]
                annotations.append({'image_path': image_path, 'bbox': bbox})

    except Exception as e:
        print(f"Warning: Erro ao ler {file_path}: {e}")

    return annotations

# Encontra todos os arquivos de anotação de forma paralela
def _load_annotations(base_paths):
    all_dataset_info = []

    print("Lendo arquivos de anotações.txt ...")

    for base_path in base_paths:
        annotation_files = glob.glob(os.path.join(base_path, '*_annotations.txt'))
        for file_path in annotation_files:
            all_dataset_info.extend(_parse_annotation_file(file_path, base_path))

    return all_dataset_info

# Embaralha e seleciona um subconjunto de amostras para processar
def _select_samples_to_process(all_info, num_samples):
    random.shuffle(all_info)
    num_to_process = min(num_samples, len(all_info))

    return all_info[:num_to_process]

# Agrega os resultados paralelos e salva em disco
def _aggregate_and_save_results(results, save_path):
    face_samples_list, non_face_samples_list = [], []

    for local_faces, local_non_faces in results:
        if local_faces:
            face_samples_list.extend(local_faces)
        if local_non_faces:
            non_face_samples_list.extend(local_non_faces)

    print(f"Agregação concluída. Amostras válidas encontradas: {len(face_samples_list)} faces, {len(non_face_samples_list)} não-faces.")
        
    face_array = np.array(face_samples_list)
    non_face_array = np.array(non_face_samples_list)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(f"{save_path}_face.npy", face_array)
        np.save(f"{save_path}_non_face.npy", non_face_array)
        print(f"Dados salvos em {save_path}_[face/non_face].npy")

    return face_array, non_face_array

"""---------------------------------------------------------------------------------------------------------"""

# Orquestra o pipeline de geração de dados
def generate_data(base_paths, num_samples, window_size= (32, 32), iou_threshold_neg= 0.1, save_path= None):
    all_info = _load_annotations(base_paths)
    dataset_to_process = _select_samples_to_process(all_info, num_samples)

    print("Começando " \
    "processamento paralelo ...")

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