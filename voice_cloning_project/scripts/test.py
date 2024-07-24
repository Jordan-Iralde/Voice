from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo
model = load_model('C:/Users/yo/Documents/GitHub/Voice/models/best_model.keras')

# Función para evaluar el modelo con datos de prueba
def test_model(x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

# Datos de prueba (ejemplo)
x_test = np.random.rand(100, 20)  # Reemplaza con tus datos de prueba reales
y_test = np.random.randint(0, 10, 100)  # Reemplaza con tus etiquetas de prueba reales

# Evaluar el modelo
test_model(x_test, y_test)
