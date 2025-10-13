import numpy as np
from sklearn.model_selection import train_test_split
import os

from .pre_processamento import gerar_dados
from .arquitetura_modelo import construir_mlp

def pre_processar_salvar():
    bases_path = ['data/Derived_YTFaces_160x160/Only_famous_high_quality', 'data/Derived_YTFaces_160x160/Only_famous_low_quality']
    num_samples = 100000
    tam_janela = (32, 32)
    dados_path = 'data/amostras_preprocessadas' 
    
    print("Gerando e salvando novos dados para o treinamento...")
    
    # Chama a função gerar_dados e passa o save_path
    gerar_dados(bases_path, num_samples, tam_janela, save_path=dados_path)
    
    print("Pré-processamento concluído e dados salvos.")

# Pipeline de treinamento do modelo
def treinar():
    # Parâmetros
    tam_janela = (32, 32)
    model_save_path = 'models/detector_faces.keras'
    dados_path = 'data/amostras_preprocessadas'

    print("Começando treinamento")

    face_data_file = f"{dados_path}_face.npy"
    non_face_data_file = f"{dados_path}_non_face.npy"

    if not (os.path.exists(face_data_file) or os.path.exists(non_face_data_file)):
        print("ERRO: Dados pré-processados não encontrados.")
        print("Execute: python main.py preprocessar")
        return

    print("Carregando dados pré-processados...")
    face_samples = np.load(face_data_file)
    non_face_samples = np.load(non_face_data_file)

    # Preparação para o treinamento
    labels_face = np.ones(len(face_samples))
    labels_non_face = np.zeros(len(non_face_samples))

    X = np.concatenate((face_samples, non_face_samples), axis=0)
    y = np.concatenate((labels_face, labels_non_face), axis=0)

    X_normalized = X.reshape(-1, tam_janela[0] * tam_janela[1]).astype(np.float32) / 255.0

    # Divisão 60-20-20 (treino-teste-validação)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_normalized, y, test_size=0.4, random_state=21, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=21, stratify=y_temp
    )

    print("Criando a MLP")

    # Construção e Treinamento do Modelo
    model = construir_mlp(input_shape=(tam_janela[0] * tam_janela[1],))
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=20, 
        batch_size=32,
        validation_data=(X_val, y_val)
    )

    loss, accuracy = model.evaluate(X_test, y_test)

    print(f"Acurácia no teste: {accuracy:.4f}")
    print(f"Perda no teste: {loss:.4f}")

    model.save(model_save_path)