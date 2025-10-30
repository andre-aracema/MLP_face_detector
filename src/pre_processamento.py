import os
import glob
import pandas as pd
import random
import cv2
import numpy as np
import dlib
from joblib import Parallel, delayed
from tqdm import tqdm

# Calcula a Interseção sobre União (IoU)
def calcular_iou(box1, box2):
    # Cantos superior esquerdo e inferior direito
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = float(box1_area + box2_area - intersection_area)

    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

# Inverte a imagem horizontalmente
def image_flip(image):
    return cv2.flip(image, 1)

# Altera o brilho e o contraste da imagem de forma aleatória
def image_brightness_contrast(image, alpha_range=(0.8, 1.2), beta_range=(-20, 20)):
    alpha = random.uniform(alpha_range[0], alpha_range[1])
    beta = random.randint(beta_range[0], beta_range[1])
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def align_face(image, predictor, tam_janela=(32, 32)):
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

        eyes_center = (int((left_eye_center[0] + right_eye_center[0]) / 2),
                        int((left_eye_center[1] + right_eye_center[1]) / 2))

        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = 0.3 * tam_janela[0] 
        scale = desired_dist / dist

        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        tX = tam_janela[0] * 0.5 - eyes_center[0]
        tY = tam_janela[1] * 0.4 - eyes_center[1]
        M[0, 2] += tX
        M[1, 2] += tY
        (w, h) = tam_janela

        aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return aligned_face
    
    except Exception as e:
        print(f"[DEBUG] Falha no alinhamento: {e}") 
        return None

# Processamento Paralelo (Otimização)
def _processar_imagem(info, tam_janela, iou_threshold_neg):
    try:
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    except Exception as e:
        print(f"Erro no worker ao inicializar dlib/clahe: {e}")
        return [], []
    
    local_face_samples = []
    local_non_face_samples = []
    
    try:
        image_color = cv2.imread(info['image_path'])
        if image_color is None:
            return [], []

        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        face_bbox = info['bbox']
        x1, y1, x2, y2 = face_bbox
        
        # Processamento da Amostra de FACE
        face_crop = image_gray[y1:y2, x1:x2]
        if face_crop.size > 0:
            aligned_face = align_face(face_crop, predictor, tam_janela)
            if aligned_face is not None:
                face_normalized = clahe.apply(aligned_face)
                local_face_samples.append(face_normalized)
                local_face_samples.append(image_flip(face_normalized))
                local_face_samples.append(image_brightness_contrast(face_normalized))

        # Geração de Amostras de NÃO-FACE
        img_h, img_w = image_gray.shape
        max_attempts = 10 
        for _ in range(max_attempts):
            if img_w >= tam_janela[0] and img_h >= tam_janela[1]:
                rand_x1 = random.randint(0, img_w - tam_janela[0])
                rand_y1 = random.randint(0, img_h - tam_janela[1])
                random_box = [rand_x1, rand_y1, rand_x1 + tam_janela[0], rand_y1 + tam_janela[1]]

                if calcular_iou(random_box, face_bbox) < iou_threshold_neg:
                    non_face_crop = image_gray[random_box[1]:random_box[3], random_box[0]:random_box[2]]
                    non_face_normalized = clahe.apply(non_face_crop)
                    local_non_face_samples.append(non_face_normalized)
                    
    except Exception:
        return [], []
        
    return local_face_samples, local_non_face_samples

# Gera amostras de face e não-face a partir de um dataset
def gerar_dados(bases_path, num_samples, tam_janela=(32, 32), iou_threshold_neg=0.1):
    # Lista onde cada item contém as coordenadas dos bouding box e caminho para imagem
    all_dataset_info = []

    print("Lendo os arquivos _annotation")

    for base_path in bases_path:
        annotation_files = glob.glob(os.path.join(base_path, '*_annotations.txt'))
        for file_path in annotation_files:
            try:
                df_file = pd.read_csv(file_path)
                for _, row in df_file.iterrows():
                    image_path = os.path.join(base_path, row['filename'])
                    if os.path.exists(image_path):
                        bbox = [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])]
                        all_dataset_info.append({'image_path': image_path, 'bbox': bbox})

            except Exception:
                print("Erro ao ler arquivo")

    # Embaralha o dataset para garantir variedade
    random.shuffle(all_dataset_info)
    dataset_info_processar = all_dataset_info[:int(num_samples)]

    print(f"Iniciando processamento paralelo de {len(dataset_info_processar)} imagens...")
    
    resultados = Parallel(n_jobs=-1)(
        delayed(_processar_imagem)(info, tam_janela, iou_threshold_neg)
        for info in tqdm(dataset_info_processar, desc="Processando Imagens")
    )

    # Agrega os resultados de todos os processos paralelos
    face_samples = []
    non_face_samples = []
    for local_faces, local_non_faces in resultados:
        face_samples.extend(local_faces)
        non_face_samples.extend(local_non_faces)
        
    print(f"\nProcessamento concluído!")
    print(f"Total de amostras geradas: {len(face_samples)} faces, {len(non_face_samples)} não-faces.")

    return np.array(face_samples), np.array(non_face_samples)