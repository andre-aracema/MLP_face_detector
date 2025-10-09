import numpy as np
from sklearn.model_selection import train_test_split

from .pre_processamento import gerar_dados
from .arquitetura_modelo import construir_mlp

# Pipeline de treinamento do modelo
def treinar():
    # Parâmetros
    bases_path = ['data/Derived_YTFaces_160x160/Only_famous_high_quality', 'data/Derived_YTFaces_160x160/Only_famous_low_quality']
    num_samples = 100000     # Quantidade de imagens que ele ira pegar da dataset
    tam_janela = (32, 32)
    model_save_path = 'models/detector_faces.keras'

    print("Começando treinamento")

    # Geração dos Dados
    face_samples, non_face_samples = gerar_dados(bases_path, num_samples, tam_janela)

    # Preparação para o treinamento
    labels_face = np.ones(len(face_samples))
    labels_non_face = np.zeros(len(non_face_samples))

    X = np.concatenate((face_samples, non_face_samples), axis=0)
    y = np.concatenate((labels_face, labels_non_face), axis=0)

    X_normalized = X.reshape(-1, tam_janela[0] * tam_janela[1]) / 255.0

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