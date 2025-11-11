import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam


# Constrói a Arquitetura de um modelo MLP
def build_mlp_architecture(input_shape):
    print(f"Construindo modelo MLP com entradas: {input_shape}")

    model = Sequential([
        Input(shape=input_shape, name="input_layer"),
        Dense(128, activation="relu", name="hidden_layer_1"),
        Dense(64, activation="relu", name="hidden_layer_2"),
        Dense(1, activation="sigmoid", name="output_layer")
    ], name="MLP_simples")

    return model

# Modelo para treinamento do modelo 2 (bootstrap)
def build_robust_mlp_architecture(input_shape):
    print(f"Construindo modelo MLP ROBUSTO (V2) com entradas: {input_shape}")

    model = Sequential([
        Input(shape=input_shape, name="input_layer"),
        
        Dense(256, activation="relu", name="hidden_layer_1"),
        Dropout(0.4, name="dropout_1"), # Desliga 40% dos neurônios
        
        Dense(128, activation="relu", name="hidden_layer_2"),
        Dropout(0.4, name="dropout_2"), # Desliga 40% dos neurônios
        
        Dense(1, activation="sigmoid", name="output_layer")
    ], name="MLP_Robusto_V2")

    return model

# Compila um modelo Keras, preparando-o para o treinamento
def compile_model(model_to_compile, learning_rate):
    print(f"Compilando modelo com taxa de aprendizado: {learning_rate}")

    model_to_compile.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss='binary_crossentropy',                 
        metrics=['accuracy']
    )

    return model_to_compile