import os
import glob
import pandas as pd
import random
import cv2
import numpy as np

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
    
    augmented_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return augmented_image

# Gera amostras de face e não-face a partir de um dataset
def gerar_dados(bases_path, num_samples, tam_janela=(32, 32), iou_threshold_neg=0.1):
    # Lista onde cada item contém as coordenadas dos bouding box e caminho para imagem
    all_dataset_info = []

    print("Lendo as arquivos _annotation")

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

    face_samples = []
    non_face_samples = []
    processed_images = 0

    tam_img_treino = num_samples * 3

    print("Criando imagens de face e no face para treinamento da MLP")

    # Processo de normalização/criação das amostras
    while len(face_samples) < tam_img_treino or len(non_face_samples) < tam_img_treino:
        if processed_images >= len(all_dataset_info):
            break

        # Percorre as imagens
        info = all_dataset_info[processed_images]
        processed_images += 1

        try:
            image_color = cv2.imread(info['image_path'])
            if image_color is None:
                continue

            image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
            face_bbox = info['bbox']

            # Amostras Face
            if len(face_samples) < tam_img_treino:
                x1, y1, x2, y2 = face_bbox
                w, h = x2 - x1, y2 - y1

                if w > 0 and h > 0:
                    face_crop = image[y1:y2, x1:x2]
                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                        face_resized = cv2.resize(face_crop, tam_janela)
                        face_samples.append(face_resized)

                        face_flipped = image_flip(face_resized)

                        face_samples.append(face_resized)
                        face_samples.append(image_flip(face_resized))
                        face_samples.append(image_brightness_contrast(face_resized))
                        
            # Amostras No Face
            if len(non_face_samples) < tam_img_treino:
                max_attempts = 50
                for _ in range(max_attempts):
                    img_h, img_w = image.shape
                    if img_w >= tam_janela[0] and img_h >= tam_janela[1]:
                        rand_x1 = random.randint(0, img_w - tam_janela[0])
                        rand_y1 = random.randint(0, img_h - tam_janela[1])
                        rand_x2 = rand_x1 + tam_janela[0]
                        rand_y2 = rand_y1 + tam_janela[1]
                        random_box = [rand_x1, rand_y1, rand_x2, rand_y2]

                        if calcular_iou(random_box, face_bbox) < iou_threshold_neg:
                            non_face_crop = image[rand_y1:rand_y2, rand_x1:rand_x2]
                            non_face_samples.append(non_face_crop)
                            break

        except Exception:
            print("Erro ao ler imagem")

    # Retorna dois array NumPy (face e non_face)
    return np.array(face_samples), np.array(non_face_samples)