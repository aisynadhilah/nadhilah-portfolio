# === dwt_mfcc.py ===
import numpy as np
import math
from scipy.signal import resample
import librosa

# === DWT & Rekonstruksi ===
# Daubechies 4 Coefficients
h = np.array([
    (1 + math.sqrt(3)) / (4 * math.sqrt(2)),
    (3 + math.sqrt(3)) / (4 * math.sqrt(2)),
    (3 - math.sqrt(3)) / (4 * math.sqrt(2)),
    (1 - math.sqrt(3)) / (4 * math.sqrt(2))
])
g = np.array([(-1)**i * h[::-1][i] for i in range(len(h))])

ih = [0]*4
ih[0] = (1 - math.sqrt(3)) / (4 * math.sqrt(2))
ih[1] = (3 - math.sqrt(3)) / (4 * math.sqrt(2))
ih[2] = (3 + math.sqrt(3)) / (4 * math.sqrt(2))
ih[3] = (1 + math.sqrt(3)) / (4 * math.sqrt(2))
ig = [(-1)**i * ih[::-1][i] for i in range(4)]

def dekomposisi(sinyal):
    n = len(sinyal)
    a, d = [], []
    for i in range(0, n - len(h) + 1, 2):
        a_val = sum(h[j] * sinyal[i + j] for j in range(4))
        d_val = sum(g[j] * sinyal[i + j] for j in range(4))
        a.append(a_val)
        d.append(d_val)
    return a, d

def dwt_level3(sinyal):
    a1, _ = dekomposisi(sinyal)
    a2, _ = dekomposisi(a1)
    a3, d3 = dekomposisi(a2)
    return a3, d3

def rekonstruksi(a, d):
    sinyal = [0.0] * (len(a) * 2 + 2)
    for i in range(len(a)):
        for j in range(4):
            idx = 2 * i + j
            if idx < len(sinyal):
                sinyal[idx] += ih[j] * a[i] + ig[j] * d[i]
    return sinyal

# === MFCC Ekstraksi ===
def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def compute_spectrogram(signal, fs, frame_duration=0.030, hop_duration=0.015, n_fft=1024):
    frame_size = int(frame_duration * fs)
    hop_size = int(hop_duration * fs)
    num_frames = (len(signal) - frame_size) // hop_size + 1
    frames = np.array([signal[i * hop_size : i * hop_size + frame_size] for i in range(num_frames)], dtype=np.float32)
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_size) / (frame_size - 1))
    frames *= hamming
    return np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

def compute_spectrogram_by_chunk(signal, fs, chunk_duration=3.0):
    chunk_size = int(chunk_duration * fs)
    spectrograms = []
    for start in range(0, len(signal), chunk_size):
        chunk = signal[start:start + chunk_size]
        if len(chunk) < int(0.030 * fs):
            continue
        spec = compute_spectrogram(chunk, fs)
        spectrograms.append(spec)
    return spectrograms

def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
def mel_to_hz(m): return 700 * (10**(m / 2595) - 1)

def get_mel_filterbank(n_filters=64, fft_size=1024, fs=11025):
    f_max = fs / 2
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(f_max), n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / fs).astype(int)
    filterbank = np.zeros((n_filters, fft_size // 2 + 1))
    for m in range(1, n_filters + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m-1], bin_points[m], bin_points[m+1]
        for k in range(f_m_minus, f_m):
            filterbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            filterbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return filterbank

def apply_log_filterbank_power(power_spectrum, filterbank):
    energies = np.dot(power_spectrum, filterbank.T)
    return np.log10(energies + 1e-10)

def apply_dct(log_mel_energy, num_ceps=13):
    num_frames, num_filters = log_mel_energy.shape
    mfccs = np.zeros((num_frames, num_ceps))
    for n in range(num_ceps):
        for t in range(num_frames):
            for m in range(num_filters):
                mfccs[t, n] += log_mel_energy[t, m] * np.cos(np.pi * n * (m + 0.5) / num_filters)
    return mfccs

# === Proses lengkap dari file audio ke MFCC ===
def process_signal_to_mfcc(signal, fs=11025):
    # 1. Zero-mean normalization
    signal = signal - np.mean(signal)

    # 2. DWT Level 3
    a3, d3 = dwt_level3(signal)
    reconstructed = np.array(rekonstruksi(a3, d3), dtype=np.float32)

    # 3. Pre-emphasis
    emphasized = pre_emphasis(reconstructed).astype(np.float32)  

    # 4. Spectrogram (per chunk)
    spectrogram_chunks = compute_spectrogram_by_chunk(emphasized, fs)
    if not spectrogram_chunks:
        return None

    spectrograms = np.vstack(spectrogram_chunks)

    # 5. MFCC extraction
    filterbank = get_mel_filterbank(fs=fs)
    log_mel = apply_log_filterbank_power(spectrograms, filterbank)
    mfcc = apply_dct(log_mel)

    return mfcc.astype(np.float32)