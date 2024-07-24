from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
import soundfile as sf
import os

app = Flask(__name__)

# Ruta donde se guardarán los archivos de audio
AUDIO_FOLDER = 'static/audio'
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Cargar el modelo
model = load_model('C:/Users/yo/Documents/GitHub/Voice/models/best_model.keras')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text', '')

    # Convertir texto a características de audio (esto depende de tu modelo)
    x_input = np.random.rand(1, 20)  # Reemplaza con tus características reales

    # Generar audio con el modelo
    try:
        audio_features = model.predict(x_input)

        # Asegúrate de que audio_features tenga una longitud adecuada
        print(f"Audio features shape: {audio_features.shape}")

        # Ajusta el reshape según sea necesario
        audio = audio_features.reshape(-1)

        # Imprimir algunos detalles del audio generado para depuración
        print(f"Audio length: {len(audio)} samples")
        print(f"Audio max value: {np.max(audio)}")

        # Guardar el audio en un archivo
        audio_path = os.path.join(AUDIO_FOLDER, 'output.wav')
        sf.write(audio_path, audio, 22050)  # Ajusta la frecuencia de muestreo si es necesario

        return jsonify({'message': 'Voice synthesized', 'file': 'output.wav'})
    except Exception as e:
        print("Error durante la síntesis:", e)
        return jsonify({'message': 'Error during synthesis', 'error': str(e)})

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
