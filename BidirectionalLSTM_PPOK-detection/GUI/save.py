import os
import shutil
from datetime import datetime

BASE_SAVE_DIR = "/home/lala/Tugas Akhir/AkuisisiData/Detection"

def simpan_semua(data_file, plot_path, mfcc_path, hasil_deteksi, label, prob):
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # Buat folder berdasarkan waktu + label deteksi
    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_nama = f"{label}_{waktu}"
    folder_tujuan = os.path.join(BASE_SAVE_DIR, folder_nama)
    os.makedirs(folder_tujuan, exist_ok=True)

    # Salin file
    shutil.copy(data_file, os.path.join(folder_tujuan, os.path.basename(data_file)))
    shutil.copy(plot_path, os.path.join(folder_tujuan, "plot_sinyal.png"))
    shutil.copy(mfcc_path, os.path.join(folder_tujuan, "mfcc_spectrogram.png"))

    # Simpan hasil deteksi
    with open(os.path.join(folder_tujuan, "hasil_deteksi.txt"), "w") as f:
        f.write(f"Label: {label}\n")
        f.write(f"Probabilitas PPOK: {prob:.4f}\n")

    print(f"[✓] Semua data disimpan di: {folder_tujuan}")
