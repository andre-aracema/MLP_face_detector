from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def construir_mlp(input_shape=(1024,)):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model