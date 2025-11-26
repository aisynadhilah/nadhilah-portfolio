import numpy as np
import librosa
import math

# =========================================================
#                1. RESAMPLING
# =========================================================

def resample_signal(signal, original_fs, target_fs=11025):
    """
    Resample sinyal lung sound ke target sampling rate (default 11025 Hz).
    """
    resampled_signal = librosa.resample(
        signal, 
        orig_sr=original_fs, 
        target_sr=target_fs
    )
    return resampled_signal, target_fs


# =========================================================
#            2. DWT LEVEL-3 (Daubechies 4)
# =========================================================

# Daubechies 4 Coefficients
h = np.array([
    (1 + math.sqrt(3)) / (4 * math.sqrt(2)),
    (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
    (3 - math.sqrt(3)) / (4 * math.sqrt(2)),
    (1 - math.sqrt(3)) / (4 * math.sqrt(2))
])

# High-pass filter g (DWT detail)
g = np.array([(-1) ** i * h[::-1][i] for i in range(len(h))])

# Inverse filters (reconstruction)
ih = [
    (1 - math.sqrt(3)) / (4 * math.sqrt(2)),
    (3 - math.sqrt(3)) / (4 * math.sqrt(2)),
    (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
    (1 + math.sqrt(3)) / (4 * math.sqrt(2))
]

ig = [(-1) ** i * ih[::-1][i] for i in range(4)]


def dekomposisi(signal):
    """
    Melakukan single-level DWT decomposition menggunakan Daubechies 4.
    Return:
        a (approximation),
        d (detail)
    """
    n = len(signal)
    a, d = [], []

    for i in range(0, n - len(h) + 1, 2):
        a_val = sum(h[j] * signal[i + j] for j in range(4))
        d_val = sum(g[j] * signal[i + j] for j in range(4))
        a.append(a_val)
        d.append(d_val)

    return a, d


def dwt_level3(signal):
    """
    DWT level-3: menghasilkan a3 dan d3.
    """
    a1, _ = dekomposisi(signal)
    a2, _ = dekomposisi(a1)
    a3, d3 = dekomposisi(a2)

    return np.array(a3), np.array(d3)


# =========================================================
#              3. REKONSTRUKSI SINYAL
# =========================================================

def reconstruct_signal(a, d):
    """
    Rekonstruksi sinyal menggunakan inverse DWT Daubechies-4.
    """
    signal = [0.0] * (len(a) * 2 + 2)

    for i in range(len(a)):
        for j in range(4):
            idx = 2 * i + j
            if idx < len(signal):
                signal[idx] += ih[j] * a[i] + ig[j] * d[i]

    return np.array(signal)


# =========================================================
#              MAIN PIPELINE (Opsional)
# =========================================================

def preprocess_pipeline(raw_signal, raw_fs):
    """
    Melakukan seluruh preprocessing:
        1. Resample ke 11025 Hz
        2. DWT Level-3
        3. Rekonstruksi sinyal

    Return:
        reconstructed_signal, resampled_signal, resampled_fs
    """
    resampled_signal, fs = resample_signal(raw_signal, raw_fs)
    a3, d3 = dwt_level3(resampled_signal)
    reconstructed = reconstruct_signal(a3, d3)

    return reconstructed, resampled_signal, fs
