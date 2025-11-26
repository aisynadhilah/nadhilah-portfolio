# === prediction.py ===
import numpy as np
import os
import tensorflow as tf
from dwt_mfcc import process_signal_to_mfcc
from scipy.signal import resample
import librosa

def load_and_preprocess_signal(filepath, fs_target=11025):
    # Load sinyal dari file TXT
    signal = np.loadtxt(filepath)

    # Resample jika perlu
    fs_original = 11111
    if fs_original != fs_target:
        signal = librosa.resample(signal.astype(float), orig_sr=fs_original, target_sr=fs_target)

    # Zero-mean normalization (agar sama dengan laptop)
    signal = signal - np.mean(signal)

    return signal, fs_target

def predict_single_model(signal_data, fs, model_path):
    # === Ekstraksi MFCC dari sinyal ===
    mfcc = process_signal_to_mfcc(signal_data, fs)
    if mfcc is None:
        raise ValueError("Gagal mengekstrak MFCC dari sinyal")

    # === Load model ===
    model = tf.keras.models.load_model(model_path, compile=False)
    time_steps, n_features = model.input_shape[1], model.input_shape[2]
    total_required = time_steps * n_features

    fitur_flat = mfcc.flatten()
    if len(fitur_flat) < total_required:
        fitur_flat = np.pad(fitur_flat, (0, total_required - len(fitur_flat)), mode='constant')
    else:
        fitur_flat = fitur_flat[:total_required]

    X_input = fitur_flat.reshape(1, time_steps, n_features).astype('float32')
    prob = model.predict(X_input, verbose=0)[0][0]
    label = "Normal" if prob < 0.61 else "PPOK"

    tf.keras.backend.clear_session()
    return prob, label, mfcc