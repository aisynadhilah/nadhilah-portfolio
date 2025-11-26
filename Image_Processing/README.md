# Skin Lesion Segmentation — Acne & Rosacea Detection
### Image Processing Pipeline for Segmenting and Identifying Acne & Rosacea Regions

Repository ini berisi implementasi lengkap segmentasi citra penyakit kulit jenis Acne dan Rosacea menggunakan metode classical image processing. Pipeline fokus pada pemisahan area kulit yang terinfeksi berdasarkan tekstur, intensitas, dan karakteristik warna, untuk kemudian dihitung luas area lesi yang terdeteksi.

## Tujuan Project
- Melakukan segmentasi pada citra wajah yang mengandung Acne atau Rosacea.
- Mengekstraksi objek-objek yang dianggap sebagai lesi kulit.
- Mengukur luas area segmentasi dan jumlah objek yang teridentifikasi.
- Menyediakan alur proses yang mudah dikembangkan ke tahap klasifikasi berbasis ML/CV.

## Metodologi
1. Input Citra
  - Mendukung format .jpg, .png, .jpeg
  - Citra dibaca dan ditampilkan sebagai referensi awal
2. Histogram Citra
  - Visualisasi distribusi intensitas piksel
  - Membantu memahami kontras & tingkat kecerahan awal citra
3. Konversi ke Grayscale
  - Menyederhanakan data dengan hanya mempertahankan informasi luminance
  - Tahap penting sebelum operasi thresholding dan filtering
4. Adaptive Histogram Equalization (AHE)
  - Meningkatkan kontras lokal pada permukaan kulit
  - Membuat lesi acne/rosacea lebih menonjol dari kulit normal
5. Median Filtering
  - Mengurangi noise (salt & pepper, tekstur kasar) tanpa merusak tepi objek
  - Memperhalus area permukaan kulit
6. Otsu Thresholding
  - Mengubah citra grayscale menjadi biner
  - Secara otomatis menentukan ambang terbaik berdasarkan distribusi intensitas
7. Binary Image
  - Representasi hitam-putih area kulit (background) dan area lesi (foreground)
8. Remove Small Objects
  - Menghapus objek kecil yang bukan lesi (noise, pori-pori, artefak cahaya)
  - Menggunakan connected component + ukuran minimum tertentu
9. Fill Small Holes
  - Mengisi lubang-lubang kecil dalam objek yang sudah tersegmentasi
  - Membuat bentuk area lesi lebih solid dan akurat
10. Labeling Citra
  - Memberi identitas pada setiap objek yang tersegmentasi
  - Menentukan jumlah objek (lesi) pada wajah
11. Area Segmentasi
  - Visualisasi informasi geometri setiap objek terdeteksi

## Catatan
1. Pipeline ini berbasis traditional image processing, bukan deep learning.
2. Akurasi segmentasi sangat dipengaruhi kualitas pencahayaan dan noise pada citra.
3. Project ini dapat menjadi dasar untuk klasifikasi ML atau CNN di tahap berikutnya.
