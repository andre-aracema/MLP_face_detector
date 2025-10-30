import cv2
import os
import numpy as np
from .treinamento import treinar, pipeline_treinamento
from .detector import FaceDetector
from .pre_processamento import gerar_dados
from .arquitetura_modelo import construir_mlp
from sklearn.model_selection import train_test_split


def bootstrap():
    # Parâmetros do dataset inicial
    bases_path_treino = ['data/Derived_YTFaces_160x160/Only_famous_high_quality', 'data/Derived_YTFaces_160x160/Only_famous_low_quality']
    num_samples_treino = 20000  
    tam_janela = (32, 32)

    # Parâmetros para a etapa de mineração
    bases_path_mineracao = ['data/Derived_YTFaces_160x160/Only_famous_low_quality'] 
    num_samples_mineracao = 30000

    # Caminhos dos modelos
    model_v1_path = 'models/detector_faces_v1.keras'
    model_v2_path = 'models/detector_faces_v2.keras'

    print("ETAPA 1: Preparando e Treinando o Modelo v1")
    
    face_samples_v1, non_face_samples_v1 = gerar_dados(bases_path_treino, num_samples_treino, tam_janela)

    if face_samples_v1.size == 0:
        print("Nenhuma amostra de face foi gerada. Abortando o processo.")
        return

    # Monta o dataset de treino para o v1
    X_v1 = np.concatenate((face_samples_v1, non_face_samples_v1))
    y_v1 = np.concatenate((np.ones(len(face_samples_v1)), np.zeros(len(non_face_samples_v1))))
    
    print(f"\nIniciando treinamento do modelo v1 com {len(X_v1)} amostras...")
    pipeline_treinamento(X_v1, y_v1, model_v1_path)
    print("--- Modelo v1 treinado com sucesso! ---")


    print("\nETAPA 2: Iniciando Mineração de Hard Negatives")
    
    _, non_face_samples_para_minerar = gerar_dados(bases_path_mineracao, num_samples_mineracao, tam_janela)

    if non_face_samples_para_minerar.size == 0:
        print("Nenhuma amostra de não-face foi gerada para mineração. Abortando.")
        return

    print("Carregando modelo v1 para encontrar falsos positivos...")
    detector = FaceDetector(model_path=model_v1_path)

    # Prepara as amostras para o modelo (reshape e normalização)
    non_faces_normalized = non_face_samples_para_minerar.reshape(len(non_face_samples_para_minerar), -1) / 255.0
    
    print("Avaliando amostras de não-face com o modelo v1...")
    scores = detector.model.predict(non_faces_normalized, batch_size=256, verbose=1)

    hard_negatives = non_face_samples_para_minerar[scores.flatten() > 0.99]

    if len(hard_negatives) == 0:
        print("\nNenhum hard negative encontrado com o limiar atual.")
        return

    print(f"--- Mineração concluída! Encontrados {len(hard_negatives)} hard negatives. ---")

    print("\nETAPA 3: Preparando e Treinando o Modelo v2 (Enriquecido)")

    # Combina as amostras de não-face originais com os hard negatives recém-descobertos
    non_faces_aprimorado = np.concatenate((non_face_samples_v1, hard_negatives), axis=0)
    
    face_samples_v2 = face_samples_v1

    X_v2 = np.concatenate((face_samples_v2, non_faces_aprimorado))
    y_v2 = np.concatenate((np.ones(len(face_samples_v2)), np.zeros(len(non_faces_aprimorado))))

    print(f"Dataset v2 criado com {len(face_samples_v2)} faces e {len(non_faces_aprimorado)} não-faces (incluindo hard negatives).")
    print(f"Iniciando treinamento do modelo v2 com {len(X_v2)} amostras...")
    
    pipeline_treinamento(X_v2, y_v2, model_v2_path)
    print("--- Modelo v2 treinado com sucesso! ---")
    print("\nProcesso de Bootstrap concluído.")