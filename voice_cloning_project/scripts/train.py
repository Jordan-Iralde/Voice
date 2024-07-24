import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Función para cargar los datos (modifica según tu caso)
def load_data():
    # Implementar la lógica para cargar tus datos
    x_train = np.random.rand(100, 20)  # Ejemplo de datos
    y_train = np.random.randint(0, 2, size=(100, 1))  # Ejemplo de etiquetas
    x_val = np.random.rand(20, 20)    # Datos de validación
    y_val = np.random.randint(0, 2, size=(20, 1))    # Etiquetas de validación
    return x_train, y_train, x_val, y_val

# Cargar los datos
x_train, y_train, x_val, y_val = load_data()

# Construir el modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.5),  # Agregar Dropout
    Dense(64, activation='relu'),
    Dropout(0.5),  # Agregar Dropout
    Dense(1, activation='sigmoid')
])

# Compilar el modelo con una tasa de aprendizaje ajustada
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Ajustar la tasa de aprendizaje
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Configurar early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Paciencia para early stopping
    restore_best_weights=True
)

# Configurar el guardado del mejor modelo
model_checkpoint = ModelCheckpoint(
    'models/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False
)

# Entrenar el modelo
history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    batch_size=32  # Ajustar tamaño del lote si es necesario
)

# Evaluar el modelo
loss, accuracy = model.evaluate(x_val, y_val)
print(f"Loss: {loss}, Accuracy: {accuracy}")
