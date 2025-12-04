# Biomedical Signal Analysis - ECG dan EEG Data Signal

Analisis sinyal Electrocardiogram (ECG) dan Electroencephalogram (EEG) dari dua subjek mahasiswa. Proyek ini dilakukan sebagai bagian dari mata kuliah Analisis Sinyal Non-Stasioner, dengan fokus pada:
1. Akuisisi sinyal EEG & ECG menggunakan perangkat klinis standar
2. Pre-processing dan noise removal
3. Analisis domain waktu & frekuensi
4. Ekstraksi fitur fisiologis dan interpretasi biomedis
5. Perbandingan respons kedua subjek pada dua skenario berbeda
6. Visualisasi menggunakan Python
7. Kesimpulan kondisi subjek berdasarkan analisis data.

## 1. Project Overview

Tujuan utama proyek ini adalah memahami karakteristik sinyal biomedik melalui proses teknik sinyal, yaitu:
### 🔹 ECG Analysis
  - Identifikasi komponen P-QRS-T
  - Perhitungan heart rate (HR)
  - Pendeteksian peak R
  - HRV (Heart Rate Variability)
  - Time Domain Analysis
  - Frequency Domain Analysis
  - Non-Stationary Analysis
  - Perbandingan kondisi Subjek 1 dan Subjek 2

### 🔹 EEG Analysis
  - Filtering untuk menghilangkan artefak
  - Analisis gelombang otak:
      - Delta (0.5–4 Hz)
      - Theta (4–7 Hz)
      - Alpha (8–13 Hz)
      - Beta (13–30 Hz)
  - Perhitungan band power tiap channel
  - Perbandingan kondisi Subjek 1 dan Subjek 2

## 2. Methods
### Preprocessing
1. Normalisasi dan resampling
2. Bandpass filter:
   - ECG: Discrete Wavelet Transform Level 3
   - EEG: 1–50 Hz
3. Artifact removal
4. Smoothing (moving average)

### Time-domain Analysis
1. ECG: deteksi R-peak, interval RR, heart rate
2. EEG: statistik sinyal (mean, variance, amplitude)

### Frequency-domain Analysis
1. Fast Fourier Transform (FFT)
2. Autonomic Balance Diagram
3. Power Spectral Density (PSD)
4. Perhitungan band-power EEG menggunakan integrasi spektrum

## 3. Results Summary
### 🔹 ECG
1. Kedua subjek memiliki bentuk gelombang ECG yang normal, termasuk pola kompleks QRS.
2. Heart rate dihitung menggunakan deteksi R-peak.
3. Subjek 1 memiliki heart rate lebih tinggi, mengindikasikan aktivitas fisiologis atau respons stres yang lebih besar meskipun kondisi tubuh secara umum baik dan seimbang.
4. Subjek 2 memiliki heart rate yang sangat tinggi serta nilai VLF yang besar, menunjukkan kemungkinan kelelahan atau stres, namun tetap memiliki kemampuan adaptasi yang baik dan sistem saraf otonom yang seimbang.

### 🔹 EEG
1. Kedua subjek berada di rentang gelombang Beta, yang mengindikasikan aktivitas mental aktif selama proses perekaman.
2. Subjek 1 memiliki nilai mean power frequency (MPF) yang lebih tinggi pada kedua area dibandingkan Subjek 2, menunjukkan tingkat kewaspadaan, fokus, atau aktivitas mental yang lebih besar.
3. Perbedaan distribusi band-power memberikan gambaran variasi respons aktivitas otak antar subjek meskipun keduanya berada dalam kondisi sistem saraf otonom yang normal.

## 4. Key Skills Demonstrated
- Data Analysis
- Biomedical signal processing (ECG & EEG)
- Filtering & preprocessing
- Python scientific computing
- Data visualization (Matplotlib)
- Interpretasi fisiologis sinyal otak dan jantung
- Laporan teknis & analitis

## 5. Author
Rihhadatul Aisy Nadhilah
- LinkedIn: https://www.linkedin.com/in/rihhadatulaisynadhilah/
- Email: ranadhilah17@gmail.com
