from pydub import AudioSegment
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np

def select_audio_file():
    """Permite al usuario seleccionar un archivo de audio."""
    Tk().withdraw()
    return askopenfilename(title="Selecciona un archivo de audio", filetypes=[("Archivos de Audio", "*.wav;*.mp3")])

def convert_audio_to_wav(audio_path, wav_path):
    """Convierte un archivo de audio a formato WAV usando pydub."""
    try:
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")
    except Exception as e:
        print(f"Error al convertir el audio: {e}")

def process_audio(audio_path):
    """Procesa el archivo de audio para extraer características."""
    try:
        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples())
        # Aquí puedes agregar la extracción de características
        print(f"Audio procesado con {len(samples)} muestras.")
    except Exception as e:
        print(f"Error al procesar el audio: {e}")

def main():
    input_audio_path = select_audio_file()
    if input_audio_path:
        wav_path = "processed_audio.wav"
        convert_audio_to_wav(input_audio_path, wav_path)
        process_audio(wav_path)
        print("Procesamiento completado.")
    else:
        print("No se seleccionó ningún archivo.")

if __name__ == "__main__":
    main()
