import librosa
import os
import numpy as np

def preprocess_audio(file_path, save_path):
    y, sr = librosa.load(file_path, sr=None)  # Cargar el archivo de audio
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # Generar el espectrograma Mel
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convertir a escala logarítmica
    np.save(save_path, log_mel_spec)  # Guardar el espectrograma Mel como archivo .npy

data_dir = r"voice_cloning_project\data\raw"
processed_dir = r"voice_cloning_project\data\processed"

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

for file_name in os.listdir(data_dir):
    if file_name.endswith(".wav"):
        file_path = os.path.join(data_dir, file_name)
        save_path = os.path.join(processed_dir, file_name.replace('.wav', '.npy'))
        preprocess_audio(file_path, save_path)



