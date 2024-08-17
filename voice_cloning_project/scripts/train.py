import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar datos
data = pd.read_csv('path_to_your_data.csv')  # Reemplaza con la ruta de tu archivo CSV

# Supongamos que tu archivo CSV tiene características en columnas y una columna 'target' como etiqueta
X = data.drop(columns=['target'])  # Ajusta según tu archivo CSV
y = data['target']  # Ajusta según tu archivo CSV

# Dividir los datos en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(learning_rate=0.001, dropout_rate=0.2):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Ajusta input_shape
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')  # Cambia según el tipo de problema
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',  # Cambia según el tipo de problema
                  metrics=['accuracy'])
    
    return model

# Parámetros de entrenamiento
learning_rate = 0.0001
dropout_rate = 0.2
epochs = 50  # Incrementar el número de épocas
batch_size = 32  # Ajusta el tamaño del batch

# Construir el modelo
model = build_model(learning_rate=learning_rate, dropout_rate=dropout_rate)

# Callback para detener el entrenamiento temprano si no hay mejora
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento
history = model.fit(X_train, y_train,  # Usa tus datos de entrenamiento
                    validation_data=(X_val, y_val),  # Usa tus datos de validación
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping])

# Graficar la pérdida y la precisión
plt.figure(figsize=(12, 6))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Gráfico de precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
