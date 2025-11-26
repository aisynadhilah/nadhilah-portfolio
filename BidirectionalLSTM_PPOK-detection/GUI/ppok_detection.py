import time
import psutil
import os, sys
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QMessageBox, QGridLayout, QGroupBox, QLineEdit, QStackedWidget, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QFontDatabase, QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from akuisisi import akuisisi_data
from plot import plot_signal
from prediction import load_and_preprocess_signal, predict_single_model
from save import simpan_semua

MODEL_DIR = "/home/lala/Tugas Akhir/GUI"
FONT_PATH = "/home/lala/Tugas Akhir/GUI/Arvo-Regular.ttf" 

def style_tombol(btn):
        btn.setStyleSheet("""
            QPushButton {
                background-color: #5a827e;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4e736f;
            }
        """)
class PPOKApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #f0f0f0; color: #333;")
        self.data_path = None
        self.mfcc_array = None
        self.probabilitas = None
        self.label_prediksi = None
        self.fs = 11111
        self.initUI()

    def initUI(self):
        if os.path.exists(FONT_PATH):
            font_id = QFontDatabase.addApplicationFont(FONT_PATH)
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                self.setFont(QFont(families[0], 10))

        # === LAYOUT UTAMA: 3 KOLOM ===
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 10, 15, 10)
        main_layout.setSpacing(10)

        # === SIDEBAR ===
        sidebar = QVBoxLayout()
        logo_layout = QHBoxLayout()
        for logo in ["logo1.png", "logo2.png"]:
            path = os.path.join(MODEL_DIR, logo)
            if os.path.exists(path):
                lbl = QLabel()
                lbl.setPixmap(QPixmap(path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                logo_layout.addWidget(lbl)
        #sidebar.addLayout(logo_layout)
        logo_widget = QWidget()
        logo_widget.setLayout(logo_layout)
        logo_widget.setContentsMargins(0, 0, 0, 10)  # Tambah margin bawah (10px)
        sidebar.addWidget(logo_widget)

        title = QLabel("PPOK DETECTION")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        sidebar.addWidget(title)
        
        sidebar.addSpacing(5)

        sidebar.addWidget(QLabel("Filename:"))
        #self.filename_input = QLineEdit()
        self.filename_input = QLineEdit("data_1")  
        sidebar.addWidget(self.filename_input)

        tombol = [
            ("START ACQUISITION", self.start_akuisisi),
            ("LOAD DATA", self.load_data),
            ("SHOW SIGNAL", self.tampilkan_sinyal),
            ("DETECTION", self.deteksi_ppok),
            ("SAVE DATA", self.simpan_data),
            ("RESET", self.reset_gui),
            ("HELP", self.help_popup)
        ]
        for txt, f in tombol:
            btn = QPushButton(txt)
            style_tombol(btn)
            btn.clicked.connect(f)
            sidebar.addWidget(btn)
        sidebar.addStretch()
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setFixedWidth(250)
        main_layout.addWidget(sidebar_widget)

        # === TENGAH: SIGNAL PLOT ===
        center_layout = QVBoxLayout()

        self.signal_group = QGroupBox("1. SIGNAL PLOT")
        self.signal_layout = QVBoxLayout()
        self.signal_group.setLayout(self.signal_layout)
        self.signal_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.signal_group.setMinimumHeight(200)
        self.signal_group.setMaximumHeight(220)
        #self.signal_group.setMinimumHeight(200)  # Kecilkan tingginya
        center_layout.addWidget(self.signal_group)

        # === BAWAH: MFCC + RIGHT SIDE ===
        bottom_split_layout = QHBoxLayout()

        # === MFCC SPECTROGRAM ===
        self.mfcc_group = QGroupBox("4. MFCC SPECTROGRAM")
        self.mfcc_layout = QVBoxLayout()
        self.mfcc_group.setLayout(self.mfcc_layout)
        self.mfcc_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #self.mfcc_group.setMinimumHeight(250)  # Tinggi ditambahin
        self.mfcc_group.setMinimumHeight(250)
        self.mfcc_group.setMaximumHeight(280)
        bottom_split_layout.addWidget(self.mfcc_group, 2)

        # === KANAN DARI MFCC: PLAY WAV + RESULT ===
        right_mfcc_side = QVBoxLayout()

        # --- PLAY WAV ---
        self.wav_group = QGroupBox("3. PLAY WAV")
        self.wav_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.wav_group.setMaximumHeight(100)
        self.play_button = QPushButton("PLAY")
        style_tombol(self.play_button)
        self.play_button.clicked.connect(self.play_wav)
        self.wav_group.setLayout(QVBoxLayout())
        self.wav_group.layout().addWidget(self.play_button)
        right_mfcc_side.addWidget(self.wav_group)

        # --- DETECTION RESULT ---
        self.result_group = QGroupBox("5. DETECTION RESULT")
        self.result_layout = QVBoxLayout()
        self.result_group.setMaximumHeight(180)
        self.result_label = QLabel("-")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.result_layout.addWidget(self.result_label)
        self.result_group.setLayout(self.result_layout)
        self.result_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_mfcc_side.addWidget(self.result_group)

        # Gabungkan ke kanan MFCC
        right_mfcc_widget = QWidget()
        right_mfcc_widget.setLayout(right_mfcc_side)
        right_mfcc_widget.setFixedWidth(270)
        bottom_split_layout.addWidget(right_mfcc_widget, 1)

        # Tambahkan bagian bawah ke center_layout
        center_layout.addLayout(bottom_split_layout)

        # === TOMBOL START & CLOSE (bawah) ===
        bottom_buttons = QHBoxLayout()
        self.start_button = QPushButton("START")
        self.start_button.setFixedSize(200, 40)
        style_tombol(self.start_button)
        self.start_button.clicked.connect(self.run_semua)
        bottom_buttons.addWidget(self.start_button)

        self.close_button = QPushButton("CLOSE")
        self.close_button.setFixedSize(200, 40)
        style_tombol(self.close_button)
        self.close_button.clicked.connect(QApplication.quit)
        bottom_buttons.addWidget(self.close_button)

        center_layout.addStretch()
        center_layout.addLayout(bottom_buttons)
        bottom_buttons.setContentsMargins(0, 10, 0, 0)
        bottom_buttons.setSpacing(20)

        # Tambahkan ke main_layout agar resize tetap oke
        main_layout.addLayout(center_layout, stretch=3)

    # === FUNCTION ===
    def start_akuisisi(self):
        filename = self.filename_input.text().strip()
        if not filename:
            QMessageBox.warning(self, "Perhatian", "Nama file tidak boleh kosong.")
            return

        try:
            filepath = akuisisi_data(filename)
            self.data_path = filepath
            QMessageBox.information(self, "Sukses", f"Data berhasil disimpan di:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal melakukan akuisisi:\n{e}")

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih File Data",
            "/home/lala/Tugas Akhir/AkuisisiData/Data",
            "Text Files (*.txt)"
        )
        if file_path:
            self.data_path = file_path
            QMessageBox.information(self, "Sukses", f"File berhasil dimuat:\n{file_path}")

    def tampilkan_sinyal(self):
        if not self.data_path:
            QMessageBox.warning(self, "Perhatian", "Belum ada data untuk ditampilkan.")
            return
        try:
            for i in reversed(range(self.signal_layout.count())):
                widget = self.signal_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            fig = plot_signal(self.data_path, return_fig=True)
            canvas = FigureCanvas(fig)
            self.signal_layout.addWidget(canvas)
            #self.hitung_jumlah_data()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal menampilkan sinyal:\n{e}")

    def deteksi_ppok(self):
        if not self.data_path:
            QMessageBox.warning(self, "Perhatian", "Belum ada data untuk dideteksi.")
            return

        try:
            from prediction import load_and_preprocess_signal, predict_single_model
            
            print("[INFO] Memuat dan memproses sinyal...")
            
            signal_data, fs = load_and_preprocess_signal(self.data_path)
            
            process = psutil.Process(os.getpid())
            # RAM sebelum inferensi
            ram_before = process.memory_info().rss / 1024 / 1024

            print("[INFO] Mulai inferensi...")
            start_time = time.time()
            
            prob, label, mfcc = predict_single_model(signal_data, fs, os.path.join(MODEL_DIR, "Model_Fold1.keras"))

            end_time = time.time()
            inference_time = end_time - start_time
            # RAM sesudah inferensi
            ram_after = process.memory_info().rss / 1024 / 1024

            print(f"[INFO] Waktu inferensi: {inference_time:.4f} detik")
            print(f"[INFO] RAM sebelum inferensi: {ram_before:.2f} MB")
            print(f"[INFO] RAM sesudah inferensi: {ram_after:.2f} MB")
            print(f"[INFO] RAM yang bertambah: {ram_after - ram_before:.2f} MB")
            print(f"[INFO] Label prediksi: {label} (Probabilitas: {prob:.4f})")
            
            self.probabilitas = prob
            self.label_prediksi = label
            self.mfcc_array = mfcc

            for layout in [self.result_layout, self.mfcc_layout]:
                for i in reversed(range(layout.count())):
                    widget = layout.itemAt(i).widget()
                    if widget:
                        widget.setParent(None)

            result_label = QLabel(f"{label}\n({prob:.4f})")
            result_label.setAlignment(Qt.AlignCenter)
            result_label.setStyleSheet("font-size: 26px; font-weight: bold; color: #2c3e50;")
            self.result_layout.addWidget(result_label)

            self.tampilkan_mfcc(mfcc)
            print("Probabilitas:", prob)
            print("Label:", label)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal melakukan deteksi:\n{e}")


    def tampilkan_mfcc(self, mfcc):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        # Hapus MFCC sebelumnya
        for i in reversed(range(self.mfcc_layout.count())):
            widget = self.mfcc_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        fig = Figure(figsize=(5, 3))
        ax = fig.add_subplot(111)

        im = ax.imshow(mfcc.T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title("MFCC Spectrogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MFCC Coefficient")

        fig.colorbar(im, ax=ax)
        canvas = FigureCanvas(fig)
        self.mfcc_layout.addWidget(canvas)

    def simpan_data(self):
        if not all([self.data_path, self.mfcc_array is not None, self.label_prediksi, self.probabilitas]):
            QMessageBox.warning(self, "Perhatian", "Data belum lengkap untuk disimpan.")
            return

        try:
            nama_file = self.filename_input.text().strip()
            if not nama_file:
                QMessageBox.warning(self, "Perhatian", "Nama file belum diisi.")
                return

            save_dir = os.path.join("/home/lala/Tugas Akhir/AkuisisiData/Detection", nama_file)
            os.makedirs(save_dir, exist_ok=True)
            
            # Simpan gambar plot sinyal
            fig_signal = plot_signal(self.data_path, return_fig=True)
            plot_path = "/tmp/plot_sinyal.png"
            fig_signal.savefig(plot_path)
            fig_signal.clf()

            # Simpan gambar MFCC
            from matplotlib.figure import Figure
            fig_mfcc = Figure(figsize=(5, 4))
            ax = fig_mfcc.add_subplot(111)
            im = ax.imshow(self.mfcc_array.T, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title("MFCC Spectrogram")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Coefficient")
            fig_mfcc.colorbar(im, ax=ax)

            mfcc_path = "/tmp/mfcc_spectrogram.png"
            fig_mfcc.savefig(mfcc_path)
            fig_mfcc.clf()

            # Panggil fungsi simpan
            from save import simpan_semua
            simpan_semua(self.data_path, plot_path, mfcc_path, None, self.label_prediksi, self.probabilitas)
            QMessageBox.information(self, "Sukses", "Semua data berhasil disimpan.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal menyimpan data:\n{e}")

    def reset_gui(self):
        for layout in [self.signal_layout, self.result_layout, self.mfcc_layout]:
            for i in reversed(range(layout.count())):
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

    def help_popup(self):
        QMessageBox.information(self, "Bantuan", 
            "1. Start Acquisition untuk merekam suara\n"
            "2. Show Signal untuk melihat plot sinyal\n"
            "3. Detection untuk klasifikasi\n"
            "4. Save Data untuk simpan ke folder\n"
            "5. Reset untuk menghapus tampilan\n\n"
            "Jika ada kendala lain, silahkan hubungi peneliti\n"
            "Lala (+62 813 7276 0128)")
        
    def run_semua(self):
        if not self.filename_input.text().strip():
            QMessageBox.warning(self, "Perhatian", "Isi nama file terlebih dahulu.")
            return
        self.start_akuisisi()
        if not self.data_path:
            return
        self.tampilkan_sinyal()
        self.hitung_jumlah_data()
        self.deteksi_ppok()
    
    def close(self):
        QApplication.quit()

    def hitung_jumlah_data(self):
        try:
            data = np.loadtxt(self.data_path)
            self.jumlah_label.setText(f"{len(data)} sample")
        except:
            self.jumlah_label.setText("Error")

    def play_wav(self):
        if not self.data_path:
            QMessageBox.warning(self, "Perhatian", "Belum ada file untuk diputar.")
            return
        try:
            data = np.loadtxt(self.data_path)
            data = data - np.mean(data)  # baseline
            data = data / np.max(np.abs(data))  # normalisasi ke -1 sampai 1
            sd.play(data, self.fs)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Gagal memainkan audio:\n{e}")


# === COVER PAGE ===
class CoverPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        logo_layout = QHBoxLayout()
        kiri = QLabel()
        kanan = QLabel()
        if os.path.exists(f"{MODEL_DIR}/logo1.png"):
            kiri.setPixmap(QPixmap(f"{MODEL_DIR}/logo1.png").scaled(90, 90, Qt.KeepAspectRatio))
        if os.path.exists(f"{MODEL_DIR}/logo2.png"):
            kanan.setPixmap(QPixmap(f"{MODEL_DIR}/logo2.png").scaled(90, 90, Qt.KeepAspectRatio))
        logo_layout.addWidget(kiri)
        logo_layout.addStretch()
        logo_layout.addWidget(kanan)
        layout.addLayout(logo_layout)

        self.label = QLabel(
            "DESIGN OF DIGITAL STETHOSCOPE FOR<br>"
            "CHRONIC OBSTRUCTIVE PULMONARY DISEASE (COPD) DETECTION<br>"
            "BASED ON MEL-FREQUENCY CEPSTRAL COEFFICIENT (MFCC)<br>"
            "FEATURE EXTRACTION"
        )
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-weight: bold; font-size: 22px; margin: 20px;")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.identitas = QLabel(
            "Rihhadatul Aisy Nadhilah\n5023211020\n\n"
            "Supervisor:\n1. Dr. Rachmad Setiawan, S.T.,M.T.\n2. Nada Fitriyetul Hikmah, S.T.,M.T.\n\n"
            "Biomedical Engineering Department\n"
            "Faculty of Intelligent Electrical Technology and Informatics\n"
            "Sepuluh Nopember Technology Institute\n"
            "2025"
        )
        self.identitas.setAlignment(Qt.AlignCenter)
        self.identitas.setFont(QFont("Arvo", 12))
        layout.addWidget(self.identitas)


        btn = QPushButton("START")
        btn.setStyleSheet("padding: 10px 20px; font-weight: bold; background-color: #5a827e; color: white;")
        style_tombol(btn)
        btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        layout.addWidget(btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        
    def resizeEvent(self, event):
        width = self.width()

        # Hitung font size berdasarkan lebar window
        title_font_size = max(16, width // 40)
        info_font_size = max(10, width // 80)

        # Perbarui font pada label judul
        self.label.setStyleSheet(
            f"font-weight: bold; font-size: {title_font_size}px; margin: 20px;"
        )

        # Perbarui font pada label identitas
        self.identitas.setFont(QFont("Arvo", info_font_size))

        super().resizeEvent(event)


# === MAIN WINDOW ===
class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        
        # === SET GLOBAL FONT ===
        if os.path.exists(FONT_PATH):
            font_id = QFontDatabase.addApplicationFont(FONT_PATH)
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                font = QFont(families[0], 10)
                QApplication.setFont(font)
                
        self.cover = CoverPage(self)
        self.main_gui = PPOKApp()
        self.addWidget(self.cover)
        self.addWidget(self.main_gui)
        self.setCurrentIndex(0)
        #self.showMaximized()
        self.setFixedSize(1024, 600)
        #self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("PPOK Detection")
    window.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    window.showFullScreen()
    window.resize(1024, 600)
    window.move(0, 0)
    #window.show()
    sys.exit(app.exec_())
