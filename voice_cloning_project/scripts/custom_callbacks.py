import tensorflow as tf

class TrainingState:
    def __init__(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

training_state = TrainingState()

class CustomCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if training_state.paused:
            self.model.stop_training = True

# Pausar entrenamiento
def pause_training():
    training_state.pause()

# Reanudar entrenamiento
def resume_training():
    training_state.resume()
