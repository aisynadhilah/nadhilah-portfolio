# Feature Selection & Classification Model Evaluation
### Final Project — Bayesian, Naïve Bayes, Logistic Regression, and Decision Tree

Project ini berfokus pada pemilihan fitur yang relevan serta evaluasi performa 4 model klasifikasi berbeda. Tujuan utamanya adalah memahami bagaimana masing-masing algoritma bekerja pada dataset yang digunakan, membandingkan akurasi, serta menganalisis faktor-faktor yang memengaruhi performa model.

## Rumusan Masalah
Final project ini dirancang untuk menjawab dua hal utama:
1. Bagaimana melakukan pemilihan fitur yang signifikan terhadap target kelas? Termasuk proses identifikasi fitur yang paling berpengaruh melalui analisis statistik dan eksplorasi pola pada data.

2. Bagaimana menentukan dan menganalisis performa empat model klasifikasi berikut:
   - Bayesian Classification
   - Naïve Bayes Classifier
   - Logistic Regression
   - Decision Tree

Evaluasi dilakukan menggunakan metrik performa seperti akurasi, sehingga dapat diketahui model mana yang paling efektif dalam melakukan klasifikasi pada dataset.

## Hasil Evaluasi Model

Performa masing-masing model adalah sebagai berikut:
| Model                   | Akurasi    |
| ----------------------- | ---------- |
| Bayesian Classification | **81.25%** |
| Naïve Bayes Classifier  | **72.92%** |
| Logistic Regression     | **93.75%** |
| Decision Tree           | **81.25%** |

## Kesimpulan
1. Setiap model menunjukkan performa yang berbeda, dengan akurasi tertinggi pada Logistic Regression (93.75%) dan terendah pada Naïve Bayes (72.92%).
2. Logistic Regression menjadi model terbaik.
Model ini mampu bekerja sangat baik pada data yang bersifat linier atau mendekati linieritas. Hubungan antar fitur yang cukup jelas membuat LR mudah menangkap pola distribusi dan menghasilkan prediksi yang stabil.

3. Naïve Bayes memiliki akurasi terendah.
Hal ini berkaitan dengan asumsi independensi antar fitur. Jika terdapat korelasi kuat antar fitur pada dataset, performa Naïve Bayes akan menurun karena model tidak dirancang untuk menangani dependensi tersebut.
