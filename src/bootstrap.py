import numpy as np
import sys
from typing import Tuple

from .training_pipeline import run_training_pipeline, load_preprocessed_data
from .face_detector import FaceDetector

"""-------------------------------Funções Auxiliáres---------------------------------------------"""

# Carrega dados pré-processados e treina o modelo v1
def _step_1_train_v1(
    data_path: str, window_size: tuple, input_size: int,
    model_path: str, learn_rate: float, epochs: int, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    print("--- ETAPA 1: Preparando e Treinando o Modelo v1 ---")
    
    X_v1, y_v1, _ = load_preprocessed_data(data_path, window_size)

    print(f"\nIniciando treinamento do modelo v1 com {len(X_v1)} amostras ...")

    run_training_pipeline(
        X_data=X_v1,
        y_data=y_v1,
        input_size=input_size,
        model_save_path=model_path,
        learning_rate=learn_rate,
        epochs=epochs,
        batch_size=batch_size
    )

    print("--- Modelo v1 treinado com sucesso! ---")
    
    face_samples_processed = X_v1[y_v1 == 1]
    non_face_samples_processed = X_v1[y_v1 == 0]
    
    return face_samples_processed, non_face_samples_processed

# Carrega dados de mineração e encontra hard negatives
def _step_2_mine_hard_negatives(
    data_path_mining: str, window_size: tuple, model_v1_path: str,
    hard_negative_threshold: float
) -> np.ndarray:
    
    print("\n--- ETAPA 2: Iniciando Mineração de Hard Negatives ---")
    
    X_mining, y_mining, _ = load_preprocessed_data(data_path_mining, window_size)
    non_faces_for_mining = X_mining[y_mining == 0]
    
    if non_faces_for_mining.size == 0:
        raise RuntimeError("Mineração da Etapa 2 falhou: Nenhuma não-face encontrada.")

    print(f"Carregando modelo v1 ({model_v1_path}) para encontrar falsos positivos ...")

    detector_v1 = FaceDetector(model_path=model_v1_path)

    print(f"Avaliando {len(non_faces_for_mining)} amostras de não-face ...")

    scores = detector_v1.predict_processed_patches(non_faces_for_mining)
    hard_negatives = non_faces_for_mining[scores > hard_negative_threshold]

    if len(hard_negatives) == 0:
        raise RuntimeError("Mineração da Etapa 2 falhou: Nenhum hard negative encontrado.")

    print(f"--- Mineração concluída! Encontrados {len(hard_negatives)} hard negatives. ---")

    return hard_negatives

# Combina dados 1D (originais + hard negatives) e treina o v2
def _step_3_train_v2(
    base_face_samples: np.ndarray, base_non_face_samples: np.ndarray, 
    hard_negatives: np.ndarray,
    input_size: int, model_path: str, learn_rate: float, 
    epochs: int, batch_size: int
):
    
    print("\n--- ETAPA 3: Preparando e Treinando o Modelo v2 (Enriquecido) ---")

    enhanced_non_faces = np.concatenate((base_non_face_samples, hard_negatives))
    face_samples_v2 = base_face_samples

    X_v2 = np.concatenate((face_samples_v2, enhanced_non_faces))
    y_v2 = np.concatenate((np.ones(len(face_samples_v2)), np.zeros(len(enhanced_non_faces))))

    print(f"Dataset v2 criado com {len(face_samples_v2)} faces e {len(enhanced_non_faces)} não-faces.")
    print(f"Iniciando treinamento do modelo v2 com {len(X_v2)} amostras ...")
    
    run_training_pipeline(
        X_data=X_v2, y_data=y_v2, input_size=input_size,
        model_save_path=model_path, learning_rate=learn_rate,
        epochs=epochs, batch_size=batch_size
    )

    print("--- Modelo v2 treinado com sucesso! ---")

"""--------------------------------------------------------------------------------------------------"""

# Orquestra o pipeline de bootstrapping
def run_bootstrap_process(
    data_path_v1: str, data_path_mining: str,
    model_v1_path: str, model_v2_path: str,
    window_size: tuple, input_size: int,
    learn_rate: float, epochs: int, batch_size: int,
    hard_negative_threshold: float
):
    
    try:

        if os.path.exists(model_v1_path):
            # O modelo v1 JÁ EXISTE. Pule o treino.
            print(f"Modelo v1 encontrado em '{model_v1_path}'. Pulando Etapa 1 (Treino).")
            
            print("Carregando dados v1 do disco para a Etapa 3 ...")

            X_v1, y_v1, _ = load_preprocessed_data(data_path_v1, window_size)
            face_v1_proc = X_v1[y_v1 == 1]
            non_face_v1_proc = X_v1[y_v1 == 0]
            
        else:
            # O modelo v1 NÃO EXISTE. Execute o treino.
            face_v1_proc, non_face_v1_proc = _step_1_train_v1(
                data_path=data_path_v1, window_size=window_size, 
                input_size=input_size, model_path=model_v1_path,
                learn_rate=learn_rate, epochs=epochs, batch_size=batch_size
            )
        
        hard_negatives_proc = _step_2_mine_hard_negatives(
            data_path_mining=data_path_mining, window_size=window_size,
            model_v1_path=model_v1_path, # Usa o v1 (treinado ou carregado)
            hard_negative_threshold=hard_negative_threshold
        )
        
        _step_3_train_v2(
            base_face_samples=face_v1_proc, 
            base_non_face_samples=non_face_v1_proc, 
            hard_negatives=hard_negatives_proc,
            input_size=input_size, model_path=model_v2_path,
            learn_rate=learn_rate, epochs=epochs, batch_size=batch_size
        )
        
        print("\nProcesso de Bootstrap concluído com sucesso.")
        
    except Exception as e:
        print(f"\n--- ERRO NO PROCESSO DE BOOTSTRAP ---")
        print(f"Erro: {e}")
        raise e