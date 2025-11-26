# MODEL - COPD Classification Models (5-Fold BiLSTM)

Folder ini berisi seluruh model terlatih yang digunakan dalam proyek Bidirectional LSTM for COPD Detection Using MFCC Features Extraction. Model disimpan untuk keperluan inference, evaluasi, dan deployment (termasuk implementasi pada Raspberry Pi). Setiap file mewakili satu model yang dilatih menggunakan subset data berbeda berdasarkan 5-Fold Stratified Cross-Validation.

## Arsitektur Model
Semua fold menggunakan arsitektur yang sama
- Input: MFCC sequence (time_steps × n_features)
- 1× BiLSTM layer (64 units)
- Batch Normalization
- Dense(48, ReLU)
- Dropout(0.2)
- Output: Sigmoid (binary classification: Normal vs PPOK)

Model dilatih menggunakan:
- Binary Cross-Entropy dengan label smoothing = 0.05
- Optimizer: Adam, lr = 5e-5
- Augmentasi: noise injection
- Dataset seimbang (Balanced Ratio 1.2 : 1)
