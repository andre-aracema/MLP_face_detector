import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .pre_processamento import gerar_dados
from .arquitetura_modelo import construir_mlp

def pipeline_treinamento(X_data, y_data, model_save_path):
    X_normalized = X_data.reshape(-1, 32 * 32) / 255.0
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_normalized, y_data, test_size=0.4, random_state=21, stratify=y_data
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=21, stratify=y_temp
    )

    model = construir_mlp(input_shape=(32 * 32,))
    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        model_save_path, monitor='val_loss', save_best_only=True, verbose=1
    )
    
    model.fit(
        X_train, y_train,
        epochs=50, 
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Acurácia no teste: {accuracy:.4f} | Perda: {loss:.4f}")

    return model


def treinar():
    # Parâmetros
    bases_path = ['data/Derived_YTFaces_160x160/Only_famous_high_quality', 'data/Derived_YTFaces_160x160/Only_famous_low_quality']
    num_samples = 20000    
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

    pipeline_treinamento(X, y, 'models/detector_faces.keras')