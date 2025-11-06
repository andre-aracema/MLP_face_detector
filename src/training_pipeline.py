import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model

from .preprocessing import generate_data 
from .model_architecture import build_mlp_architecture, compile_model

# Chama 'generate_data' para criar e salvar amostras de rosto e não rosto em um arquivo .npy
def preprocess_and_save_data(base_paths: list, num_samples: int, window_size: tuple, save_path_base: str):

    print("Começando o Pré processamento ...")
    print(f"Gerando {num_samples} amostras e salvando em '{save_path_base}_[face/non_face].npy'")

    generate_data(base_paths, num_samples, window_size, save_path=save_path_base)

    print("Pré processamento completo.")

# Carrega os arquivos.npy pré-processados, combina-os, normaliza e formata para treinamento.
def load_preprocessed_data(load_path_base: str, window_size: tuple) -> tuple:

    print(f"Carregando dados do {load_path_base}")

    face_data_file = f"{load_path_base}_face.npy"
    non_face_date_file = f"{load_path_base}_non_face.npy"

    if not (os.path.exists(face_data_file) and os.path.exists(non_face_date_file)):
        print("ERRO: Dados pré processados não encontrados.")

    face_samples = np.load(face_data_file)
    non_face_samples = np.load(non_face_data_file)

    # Preparação
    labels_face = np.ones(len(face_samples))
    labels_non_face = np.zeros(len(non_face_samples))

    X = np.concatenate((face_samples, non_face_samples), axis=0)
    y = np.concatenate((labels_face, labels_non_face), axis=0)

    input_size = window_size[0] * window_size[1]
    X_processed = X.reshape(-1, input_size).astype(np.float32) / 255.0
    
    print(f"Dados carregados. Formas: X={X_processed.shape}, y={y.shape}")

    return X_processed, y, input_size

# Executa o pipeline para dividir, compilar, treinar e avaliar o modelo.
def run_training_pipeline(
    X_data: np.ndarray, y_data: np.ndarray, 
    input_size: int, model_save_path: str, 
    learning_rate: float, eochs: int, batch_size: int
) -> Model:

    print("Dividindo os dados em dados de traino, validação e teste ...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_data, y_data, test_size=0.4, random_state=21, stratify=y_data
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=21, stratify=y_temp
    )

    # Construindo e Compilando
    model = build_mlp_architecture(input_shape=(input_size,))
    model = compile_model(model, learning_rate=learning_rate)
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, verbose=1, restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        model_save_path, monitor='val_loss', save_best_only=True, verbose=1
    )
    
    # Treinamento
    print("\nIniciando treinamento do modelo ...")

    model.fit(
        X_train, y_train,
        epochs=epochs, 
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )

    print("Treino completo.")

    # Avaliação
    print("\nAvaliando o modelo ...")

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f} | Test Loss: {loss:.4f}")

    return model