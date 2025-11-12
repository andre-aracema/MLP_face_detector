import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import AdamW


# Constrói a Arquitetura de um modelo MLP
def build_mlp_architecture(input_shape):
    print(f"Construindo modelo MLP com entradas: {input_shape}")

    model = Sequential([
        Input(shape=input_shape, name="input_layer"),
        Dense(128, kernel_initializer='he_normal', name="hidden_layer_1"),
        BatchNormalization(),
        Activation('relu'),

        Dense(64, kernel_initializer='he_normal', name="hidden_layer_2"),
        BatchNormalization(),
        Activation('relu'),

        Dense(1, activation="sigmoid", name="output_layer")
    ], name="MLP_simples")

    return model

# Modelo para treinamento do modelo 2 (bootstrap)
def build_robust_mlp_architecture(input_shape):
    print(f"Construindo modelo MLP ROBUSTO (V2) com entradas: {input_shape}")

    model = Sequential([
        Input(shape=input_shape, name="input_layer"),
        
        Dense(256, kernel_initializer='he_normal', name="hidden_layer_1"),
        BatchNormalization(name="bn_1"),
        Activation('relu', name="relu_1"),
        Dropout(0.4, name="dropout_1"), # Desliga 40% dos neurônios
        
        Dense(128, kernel_initializer='he_normal', name="hidden_layer_2"),
        BatchNormalization(name="bn_2"),
        Activation('relu', name="relu_2"),
        Dropout(0.4, name="dropout_2"), # Desliga 40% dos neurônios
        
        Dense(1, activation="sigmoid", name="output_layer")
    ], name="MLP_Robusto_V2")

    return model

# Compila um modelo Keras, preparando-o para o treinamento
def compile_model(model_to_compile, learning_rate, weight_decay=1e-4):
    print(f"Compilando modelo com taxa de aprendizado: {learning_rate}, Weight Decay: {weight_decay}")

    model_to_compile.compile(
        optimizer=AdamW(
            learning_rate=learning_rate, 
            weight_decay=weight_decay,
            clipnorm=1.0 
        ), 
        loss='binary_crossentropy',                 
        metrics=['accuracy']
    )

    return model_to_compile