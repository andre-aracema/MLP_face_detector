import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .preprocessing import generate_data 
from .model_architecture import build_mlp_architecture, build_robust_mlp_architecture, compile_model


# Carregar apenas um lote de dados do disco usando "memory mapping" (mmap)
class FaceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path_base, window_size, indices: np.ndarray, batch_size, is_training= True):
        self.data_path_base = data_path_base
        self.indices = indices
        self.batch_size = batch_size
        self.input_size = window_size[0] * window_size[1]
        self.is_training = is_training 

        # mmap_mode='r' "mapeia" o arquivo no disco.
        try:
            self.face_mmap = np.load(f"{self.data_path_base}_face.npy", mmap_mode='r')
            self.non_face_mmap = np.load(f"{self.data_path_base}_non_face.npy", mmap_mode='r')

        except FileNotFoundError:
            print(f"Erro: Gerador não encontrou dados em {self.data_path_base}")
            raise
            
        self.num_faces = len(self.face_mmap)
        self.num_non_faces = len(self.non_face_mmap)
        self.on_epoch_end()

    # Retorna o número de lotes (batches) por época
    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    # Embaralha os índices no final de cada época (apenas para treino)
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)

    # Gera um lote (batch) de dados
    def __getitem__(self, idx):
        # Pega os índices para este lote
        batch_indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        
        # Prepara os arrays vazios para este lote
        batch_x = np.empty((self.batch_size, self.input_size), dtype=np.float32)
        batch_y = np.empty((self.batch_size), dtype=np.int8)

        for i, global_idx in enumerate(batch_indices):
            # O índice global nos diz se é uma face ou não-face
            if global_idx < self.num_faces:
                # É uma face
                sample_data = self.face_mmap[global_idx]
                batch_y[i] = 1
            else:
                # É uma não-face (precisamos subtrair o offset das faces)
                non_face_idx = global_idx - self.num_faces
                sample_data = self.non_face_mmap[non_face_idx]
                batch_y[i] = 0
            
            # Processa (achata e normaliza) APENAS esta amostra
            batch_x[i] = sample_data.reshape(-1).astype(np.float32) / 255.0
            
        return batch_x, batch_y

# Chama 'generate_data' para criar e salvar amostras de rosto e não rosto em um arquivo .npy
def preprocess_and_save_data(base_paths, num_samples, window_size, save_path_base):
    print("Começando o Pré processamento ...")
    print(f"Gerando amostras e salvando em '{save_path_base}_[face/non_face].npy'")

    generate_data(base_paths, num_samples, window_size, save_path=save_path_base)

# Carrega os arquivos.npy pré-processados, combina-os, normaliza e formata para treinamento.
def load_preprocessed_data(load_path_base, window_size):
    print(f"Carregando dados do {load_path_base}")

    face_data_file = f"{load_path_base}_face.npy"
    non_face_data_file = f"{load_path_base}_non_face.npy"

    if not (os.path.exists(face_data_file) and os.path.exists(non_face_data_file)):
        print("ERRO: Dados pré processados não encontrados.")
        raise FileNotFoundError(f"Arquivos não encontrados: {face_data_file} ou {non_face_data_file}")

    face_samples = np.load(face_data_file)
    non_face_samples = np.load(non_face_data_file)

    # Preparação
    labels_face = np.ones(len(face_samples))
    labels_non_face = np.zeros(len(non_face_samples))

    X = np.concatenate((face_samples, non_face_samples), axis=0)
    y = np.concatenate((labels_face, labels_non_face), axis=0)

    input_size = window_size[0] * window_size[1]
    # Processa os dados brutos (2D) para 1D normalizado
    X_processed = X.reshape(-1, input_size).astype(np.float32) / 255.0
    
    print(f"Dados carregados. Formas: X={X_processed.shape}, y={y.shape}")

    return X_processed, y, input_size

# Executa o pipeline para dividir, compilar, treinar e avaliar o modelo.
def run_training_pipeline(data_path_base, window_size, input_size, model_save_path, learning_rate, epochs, batch_size):
    print("Dividindo os dados em dados de treino, validação e teste ...")

    # Mapeia os dados para pegar os tamanhos
    face_map = np.load(f"{data_path_base}_face.npy", mmap_mode='r')
    non_face_map = np.load(f"{data_path_base}_non_face.npy", mmap_mode='r')
    num_faces = len(face_map)
    num_non_faces = len(non_face_map)
    del face_map, non_face_map # Fecha os mapas

    # Cria um array de ÍNDICES
    face_indices = np.arange(num_faces)
    non_face_indices = np.arange(num_faces, num_faces + num_non_faces)
    
    # Divisão 80% treino, 20% temp (val+test)
    train_face_idx, temp_face_idx = train_test_split(face_indices, test_size=0.2, random_state=21)
    train_non_face_idx, temp_non_face_idx = train_test_split(non_face_indices, test_size=0.2, random_state=21)

    # Divisão do 'temp' (20% do total) ao meio (50/50),
    val_face_idx, test_face_idx = train_test_split(temp_face_idx, test_size=0.5, random_state=21)
    val_non_face_idx, test_non_face_idx = train_test_split(temp_non_face_idx, test_size=0.5, random_state=21)

    # Combina os índices
    train_indices = np.concatenate((train_face_idx, train_non_face_idx))
    val_indices = np.concatenate((val_face_idx, val_non_face_idx))
    test_indices = np.concatenate((test_face_idx, test_non_face_idx))
    
    print(f"Total: {len(train_indices)} treino, {len(val_indices)} validação, {len(test_indices)} teste.")

    # Cria os Geradores 
    train_generator = FaceDataGenerator(
        data_path_base=data_path_base, window_size=window_size,
        indices=train_indices, batch_size=batch_size, is_training=True
    )
    val_generator = FaceDataGenerator(
        data_path_base=data_path_base, window_size=window_size,
        indices=val_indices, batch_size=batch_size, is_training=False
    )
    test_generator = FaceDataGenerator(
        data_path_base=data_path_base, window_size=window_size,
        indices=test_indices, batch_size=batch_size, is_training=False
    )

    # Constroi e Compila
    # Versão 2
    if learning_rate < 0.0005: 
        model = build_robust_mlp_architecture(input_shape=(input_size,))
    # Versão 1
    else: 
        model = build_mlp_architecture(input_shape=(input_size,))

    model = compile_model(model, learning_rate=learning_rate)
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        model_save_path, monitor='val_loss', save_best_only=True, verbose=1
    )
    
    # Treinamento
    print("\nIniciando treinamento do modelo (com Gerador)...")
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, model_checkpoint]
    )
    print("Treino completo.")

    # Avaliação
    print("\nAvaliando o modelo no conjunto de Teste...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {accuracy:.4f} | Test Loss: {loss:.4f}")

    return model