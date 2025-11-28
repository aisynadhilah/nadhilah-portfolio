# Design of Digital Stethoscope for Chronic Obstructive Pulmonary Disease (COPD) Detection Based on Mel-Frequency Cepstral Coefficient (MFCC) Feature Extraction
Keyword: Signal processing: lung sound signal • Feature Extraction • Deep Learning: Bidirectional LSTM

## Project Overview
This project was developed as part of my undergraduate final thesis.
This project aims to detect Chronic Obstructive Pulmonary Disease (COPD) based on lung sound recordings using a Bidirectional LSTM (BiLSTM) model. The pipeline integrates Discrete Wavelet Transform (DWT) and MFCC feature extraction before feeding the signal into a deep learning classifier.

## Methodology
1. Pre-processing
- Resampling lung sound data to 11025 Hz
- Applying Discrete Wavelet Transform (DWT) level-3
- Reconstructing the signal after DWT filtering

2. Feature Extraction
- Extracting MFCC features from the reconstructed signal
  - Pre-Emphasis with a=0,97
  - Signal framing within a range of 30 ms and 50% overlap
  - Hamming Window
  - Fast Fourier Transform (FFT)
  - Mel Filterbank
  - Log Operation
  - Discrete Cosine Transform (DCT)
- Saving MFCC arrays as .npy for efficient processing

3. Model Architecture
- 1× Bidirectional LSTM layer (64 units)
- Batch Normalization
- Dense (48 units, ReLU)
- Dropout (0.2)
- Output layer: Sigmoid
- Optimizer: Adam (lr = 5e-5)
- Loss: Binary Crossentropy with label_smoothing=0.05

4. Training Setup
- 5-Fold Cross Validation
- Dataset balancing (ratio 1.2:1 for Normal:PPOK)
- Noise augmentation applied before training and validation
- Stored models in .keras format

5. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Probability Histogram
- Fold-wise test evaluation (step-by-step testing)

## Project Structure

BidirectionalLSTM_PPOK-detection/

│── README.md                        # Project documentation (this file)

│── Pre-Processing.py                # Pre-processing code

│── DWT_MFCC.py                      # DWT + MFCC feature extraction

│── Model_Training_Evaluation.py     # Full BiLSTM training pipeline

│── Prediction.py                    # Real-time prediction script (Raspberry Pi)

│── GUI/                             # Graphical User Interface - Touchscreen

│── MODEL/                           # Saved 5-fold .keras models

│── RESULTS/                         # Evaluation metrics

│── DATASET/                         # dataset

## Key Results
- Achieved 91.06% training accuracy, 89.05% validation accuracy, and 89.34% test accuracy.
- Model demonstrated balanced performance across both classes with 89% sensitivity and 89% specificity.
- Results indicate that the BiLSTM model can detect COPD with minimal misclassification using MFCC-based lung sound features.

## Future Improvements
- Expand the dataset by including subjects across wider age ranges and individuals with confirmed COPD history.
- Extend the system to classify other respiratory diseases using lung sound signals.
- Integrate the model into a more advanced monitoring interface supported by a centralized database for long-term patient tracking.
- Deploy the system on real-time embedded hardware (Raspberry Pi) to process data directly from digital stethoscopes.
