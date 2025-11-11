import numpy as np
import os
import sys
import glob
import numpy.random as npr

from .training_pipeline import run_training_pipeline
from .face_detector import FaceDetector


# Treinar modelo 1
def _train_v1(data_path_v1, model_v1_path, window_size, input_size, learn_rate, epochs, batch_size):
    print("ETAPA 1: Treinando o Modelo 1")
    
    if os.path.exists(model_v1_path):
        print(f"Modelo v1 já existe em '{model_v1_path}'. Pulando treino.")
        return

    print(f"Iniciando treinamento do modelo v1 com dados de '{data_path_v1}'...")
    
    run_training_pipeline(
        data_path_base=data_path_v1,
        window_size=window_size,
        input_size=input_size,
        model_save_path=model_v1_path,
        learning_rate=learn_rate,
        epochs=epochs,
        batch_size=batch_size
    )

    print("Modelo 1 treinado com sucesso!")


# Minerar Hard Negatives
def _mine_hard_negatives(base_paths_for_mining, model_v1_path, data_path_v1_for_count, hard_negative_threshold):
    print("\nETAPA 2: Iniciando Mineração de Hard Negatives (Automática)")

    try:
        base_face_samples = np.load(f"{data_path_v1_for_count}_face.npy")
        target_count = len(base_face_samples)
        print(f"Meta: Encontrar {target_count} hard negatives (igual ao número de faces V1).")
    except FileNotFoundError:
        print(f"Erro: Não foi possível carregar faces V1 de '{data_path_v1_for_count}' para contagem.")
        raise

    print(f"Carregando modelo 1 ({model_v1_path}) para encontrar falsos positivos...")
    detector_v1 = FaceDetector(model_path=model_v1_path)

    hard_negatives = detector_v1.mine_hard_negatives_from_dataset(
        base_paths_list=base_paths_for_mining,
        target_count=target_count, 
        confidence_threshold=hard_negative_threshold
    )

    if hard_negatives.size == 0:
        print("AVISO: Mineração da Etapa 2 concluída, mas nenhum hard negative foi encontrado.")
        return np.array([])

    print(f"Mineração concluída! Encontrados {len(hard_negatives)} hard negatives.")

    return hard_negatives

# Salva os dados 2 
def _save_v2_data(data_path_v1, hard_negatives, data_path_v2):
    print("\nETAPA 3: Preparando e Salvando Dataset 2 (com amostragem)")

    try:
        base_face_samples = np.load(f"{data_path_v1}_face.npy")
    except FileNotFoundError:
        print(f"Erro: Não foi possível carregar dados 1 (faces) de {data_path_v1}")
        raise

    num_faces = len(base_face_samples)
    num_hard = len(hard_negatives)
    print(f"Dataset base: {num_faces} rostos.")
    print(f"Minerados: {num_hard} hard negatives.")

    if num_hard == 0:
        print("AVISO: Nenhum hard negative encontrado. O dataset 2 não será salvo.")
        return

    print(f"Balanceando dataset... Selecionando {num_hard} rostos aleatórios (Undersampling).")

    # Gera 'num_hard' índices aleatórios, sem repetição, do array de faces
    face_indices = npr.choice(num_faces, num_hard, replace=False)

    # Seleciona apenas as faces desses índices
    final_face_samples = base_face_samples[face_indices]
    
    final_non_faces = hard_negatives

    # Salvar o novo dataset V2 em disco
    os.makedirs(os.path.dirname(data_path_v2), exist_ok=True)
    np.save(f"{data_path_v2}_face.npy", base_face_samples)
    np.save(f"{data_path_v2}_non_face.npy", final_non_faces)
    
    print(f"Dataset 2 salvo com sucesso em '{data_path_v2}_[...]'.")
    print(f"Total: {len(base_face_samples)} rostos, {len(final_non_faces)} não-rostos.")


def run_bootstrap_process(
    data_path_v1,         
    base_paths_for_mining,    
    data_path_v2,         
    model_v1_path,        
    window_size,
    input_size,
    learn_rate_v1,
    epochs_v1,
    batch_size,
    hard_negative_threshold
):
    
    try:
        # Treinar o modelo 1
        _train_v1(
            data_path_v1=data_path_v1, model_v1_path=model_v1_path,
            window_size=window_size, input_size=input_size,
            learn_rate=learn_rate_v1, epochs=epochs_v1, batch_size=batch_size
        )
        
        # Minerar
        hard_negatives_proc = _mine_hard_negatives(
            base_paths_for_mining=base_paths_for_mining,
            model_v1_path=model_v1_path,
            data_path_v1_for_count=data_path_v1,
            hard_negative_threshold=hard_negative_threshold
        )
        
        # Salvar
        _save_v2_data(
            data_path_v1=data_path_v1,
            hard_negatives=hard_negatives_proc,
            data_path_v2=data_path_v2
        )
        
        print(f"\nProcesso de Bootstrap (mineração e salvamento) concluído.")
        print(f"Modelo 1 treinado: {model_v1_path}")
        print(f"Dados 2 salvos em: {data_path_v2}")
        print(f"\nPróximo passo: Treine o modelo 2 com 'python main.py train --config v2'")
        
    except Exception as e:
        print(f"\n--- ERRO NO PROCESSO DE BOOTSTRAP ---")
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()