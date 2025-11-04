import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from google import genai
from google.genai import types

# $env:GEMINI_API_KEY="AIzaSyCpQRGo0yWjxo-Bo282yJdEQFp51IEvP8M"

# Impor Modul PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, 
    QComboBox, QTabWidget, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
# FIX: QIntValidator ditambahkan ke PySide6.QtGui
from PySide6.QtGui import QFont, QColor, QBrush, QLinearGradient, QPalette, QIntValidator 

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# ======================================================================
# BAGIAN 1: DATA DAN PELATIHAN MODEL
# ======================================================================

data_indonesia = {
    'Makanan': ['Nasi Goreng Kampung', 'Sate Ayam Madura', 'Soto Ayam Lamongan', 'Gado-Gado', 'Rendang Sapi', 'Pempek Kapal Selam', 'Bakso Urat', 'Indomie Goreng', 'Kue Cubit', 'Es Cendol', 'Rawon Surabaya', 'Sayur Asem', 'Pepes Ikan Mas', 'Tahu Gejrot', 'Martabak Manis', 'Martabak Telor', 'Kerupuk Udang', 'Kopi Hitam Aceh', 'Teh Tarik', 'Bika Ambon', 'Klepon', 'Cireng', 'Seblak Pedas', 'Kripik Singkong Balado', 'Pisang Goreng Keju', 'Sambal Terasi', 'Asinan Betawi', 'Lumpia Semarang', 'Nasi Uduk', 'Ayam Geprek', 'Kue Lumpur', 'Putu Ayu', 'Es Campur', 'Kari Ayam', 'Mie Ayam Jamur', 'Kupat Tahu Magelang', 'Wingko Babat', 'Gulai Kambing', 'Cilok Bumbu Kacang', 'Piscok Lumer', 'Coto Makassar', 'Rujak Buah', 'Es Doger', 'Bubur Ayam', 'Onde-Onde', 'Dodol Garut', 'Karedok', 'Wedang Jahe', 'Manisan Pala', 'Kue Lapis'],
    'Asin (Na+)': [45, 30, 40, 20, 35, 30, 50, 40, 5, 2, 35, 10, 25, 15, 5, 45, 35, 5, 10, 5, 1, 25, 15, 10, 8, 50, 15, 30, 35, 40, 5, 2, 5, 30, 35, 30, 5, 40, 25, 2, 30, 5, 3, 30, 2, 5, 10, 5, 5, 5],
    'Asam (H+)': [15, 5, 10, 30, 5, 40, 5, 10, 5, 1, 5, 40, 20, 50, 5, 10, 0, 10, 0, 5, 0, 5, 5, 5, 1, 5, 45, 5, 10, 5, 5, 2, 10, 5, 5, 15, 2, 5, 5, 1, 10, 60, 5, 5, 1, 1, 35, 1, 40, 5],
    'Manis (Gula)': [20, 40, 10, 20, 15, 10, 5, 15, 60, 70, 5, 15, 5, 10, 80, 10, 5, 5, 30, 50, 65, 5, 5, 5, 55, 1, 20, 15, 10, 5, 60, 75, 60, 15, 10, 5, 65, 10, 15, 70, 10, 40, 75, 10, 55, 60, 20, 5, 50, 50],
    'Pahit (Alkaloid)': [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
    'Umami (Glutamat)': [55, 40, 50, 45, 60, 40, 70, 65, 0, 0, 65, 20, 40, 10, 0, 50, 30, 0, 0, 0, 0, 15, 25, 10, 0, 15, 10, 40, 50, 55, 0, 0, 0, 55, 50, 40, 0, 50, 20, 0, 50, 5, 0, 45, 0, 0, 25, 0, 0, 0]
}

kategori_rasa_indonesia = [
    'Asin-Umami Dominan', 'Manis-Umami Seimbang', 'Umami Gurih Kuat', 'Asam-Umami Kaya', 'Umami Rempah Kuat',
    'Asam-Asin Kuat', 'Umami Gurih Kuat', 'Umami Gurih Kuat', 'Manis Dominan', 'Manis Dominan',
    'Umami Rempah Kuat', 'Asam Segar', 'Umami Rempah Kuat', 'Asam-Pedas Kuat', 'Manis Dominan',
    'Asin-Umami Kuat', 'Umami Gurih Ringan', 'Pahit Dominan', 'Manis-Susu', 'Manis Dominan',
    'Manis Dominan', 'Asin-Gurih', 'Pedas Gurih Kuat', 'Asin-Manis Pedas', 'Manis Dominan',
    'Asin-Pedas Kuat', 'Asam Segar', 'Umami Gurih', 'Umami Gurih Kuat', 'Asin-Pedas Kuat',
    'Manis Dominan', 'Manis Dominan', 'Manis Dominan', 'Umami Rempah Kuat', 'Umami Gurih',
    'Asam-Asin Seimbang', 'Manis Dominan', 'Umami Rempah Kuat', 'Asin-Gurih Kacang', 'Manis Dominan',
    'Umami Rempah Kuat', 'Asam-Pedas Kuat', 'Manis Dominan', 'Umami Gurih', 'Manis Dominan',
    'Manis Dominan', 'Asam Segar', 'Hangat Rempah', 'Asam-Manis Kuat', 'Manis Dominan'
]

df = pd.DataFrame(data_indonesia)
df['Kategori_Rasa'] = kategori_rasa_indonesia
df.set_index('Makanan', inplace=True)
kolom_rasa = ['Asin (Na+)', 'Asam (H+)','Manis (Gula)', 'Pahit (Alkaloid)', 'Umami (Glutamat)']

X = df[kolom_rasa]
y = df['Kategori_Rasa']
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
model_knn = KNeighborsClassifier(n_neighbors=5) 
model_knn.fit(X_scaled, y)

# ======================================================================
# BAGIAN 2: WORKER THREAD UNTUK PANGGILAN GEMINI 
# ======================================================================

class GeminiWorker(QThread):
    finished = Signal(str) 

    def __init__(self, nama_sampel, profil_persen):
        super().__init__()
        self.nama_sampel = nama_sampel
        self.profil_persen = profil_persen

    def run(self):
        if 'GEMINI_API_KEY' not in os.environ:
            self.finished.emit("ERROR API: Kunci GEMINI_API_KEY tidak ditemukan. Harap setel variabel lingkungan.")
            return

        try:
            client = genai.Client()
            data_points = "\n".join([f"- {rasa.split(' ')[0]}: {persen:.1f}%" 
                                    for rasa, persen in self.profil_persen.items()])

            # PROMPT UNTUK OUTPUT TERSTRUKTUR (Sesuai permintaan)
            prompt_text = f"""
            Anda adalah kritikus makanan AI untuk Quality Control (QC) restoran. 
            Tugas Anda adalah menganalisis profil rasa dari data sensor yang diberikan.
            Berikan output dengan struktur yang KETAT:

            Analisisnya : [Analisis Overview Maksimal 2 Kalimat]
            Rekomendasi Industri Restoran : [Rekomendasi Praktis Maksimal 3 Kalimat]

            Data Intensitas Rasa Relatif (Dinormalisasi) untuk Makanan {self.nama_sampel}:
            {data_points}
            """

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_text,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            self.finished.emit(response.text)
            
        except Exception as e:
            self.finished.emit(f"ERROR API: Gagal terhubung/proses Gemini. Detail: {e}")


# ======================================================================
# BAGIAN 3: CLASS UTAMA APLIKASI PYQT/PYSIDE
# ======================================================================

class AITastingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Mencicipi (E-Tongue) Simulator")
        self.setGeometry(100, 100, 1000, 750) 
        
        self.gemini_thread = None
        self.input_fields = {} 
        self.data_labels = {} 
        self._set_style()
        self._setup_ui()
        
        QTimer.singleShot(100, lambda: self._update_standard_data(0))


    def _set_style(self):
        
        font = QFont("Segoe UI", 10)
        self.setFont(font)
        
        gradient_style = """
            qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #4285F4, 
                stop:0.33 #EA4335,
                stop:0.66 #FBBC05,
                stop:1 #34A853
            )
        """

        style = f"""
            QWidget {{
                background-color: #f5f5f5; 
                font-family: Segoe UI;
                color: #333333; 
            }}
            QLabel#Header {{
                font-size: 20pt;
                font-weight: bold;
            }}
            QComboBox, QLineEdit, QTabWidget, QFrame, QTextEdit {{
                background-color: white;
                border-radius: 8px; 
                border: 1px solid #ddd;
                padding: 5px;
                color: #333333;
            }}
            QPushButton {{
                background: {gradient_style};
                color: white;
                border: none;
                padding: 10px 12px;
                border-radius: 6px; 
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3367D6, 
                    stop:0.33 #D6392D,
                    stop:0.66 #E4AA00,
                    stop:1 #0F9D58
                );
            }}
            QFrame#KandunganFrame {{
                background-color: #F8F8F8;
                border: 1px solid #C6DBEF;
            }}
            QLabel#KeteranganGemini {{
                color: #616161; 
                font-size: 9pt;
                font-style: italic;
            }}
            QLabel#RasaLabel {{
                color: #4285F4;
            }}
        """
        self.setStyleSheet(style)

    def _setup_ui(self):
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 20, 40, 20)
        
        # --- HEADER (Judul Program) ---
        header_label = QLabel("AI Mencicipi: E-Tongue Simulator")
        header_label.setObjectName("Header")
        header_label.setAlignment(Qt.AlignCenter)
        
        # Gradient Programmatis untuk Text Header
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 300, 0)
        gradient.setColorAt(0.0, QColor("#4285F4"))
        gradient.setColorAt(0.33, QColor("#EA4335"))
        gradient.setColorAt(0.66, QColor("#FBBC05"))
        gradient.setColorAt(1.0, QColor("#34A853"))
        palette.setBrush(QPalette.WindowText, QBrush(gradient))
        header_label.setPalette(palette)
        
        main_layout.addWidget(header_label)
        main_layout.addSpacing(20)

        # --- INPUT & STANDAR FRAME ---
        top_input_widget = QWidget()
        top_layout = QHBoxLayout(top_input_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tab_widget = QTabWidget()
        self._setup_input_tabs()
        top_layout.addWidget(self.tab_widget, 2) 

        self.kandungan_frame = QFrame()
        self.kandungan_frame.setObjectName("KandunganFrame")
        self.kandungan_layout = QVBoxLayout(self.kandungan_frame)
        self.kandungan_layout.addWidget(QLabel("<b>Kandungan Rasa Standar (QC)</b>", alignment=Qt.AlignCenter))
        self.kandungan_layout.addWidget(QLabel("<i>Data sensor ideal (Tidak dapat diubah)</i>", alignment=Qt.AlignCenter))
        self.kandungan_layout.addSpacing(10)
        
        for i, rasa in enumerate(kolom_rasa):
            label = QLabel(f"• {rasa}: --")
            label.setObjectName("RasaLabel")
            self.data_labels[rasa] = label
            self.kandungan_layout.addWidget(label)
        
        self.kandungan_layout.addStretch(1)

        top_layout.addWidget(self.kandungan_frame, 1) 
        main_layout.addWidget(top_input_widget)
        
        # Tombol Mulai Analisis (Di Bawah Input)
        self.analyze_button = QPushButton("Mulai Analisis")
        self.analyze_button.setMinimumHeight(45)
        self.analyze_button.clicked.connect(self._start_analysis)
        main_layout.addWidget(self.analyze_button, alignment=Qt.AlignCenter)
        self.analyze_button.setStyleSheet(self.styleSheet() + "QPushButton {max-width: 300px;}") 
        
        main_layout.addSpacing(30)
        
        # --- OUTPUT FRAME (Di Bawah: Grafik dan Feedback) ---
        output_widget = QWidget()
        output_layout = QHBoxLayout(output_widget)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax.axis('off')
        self.canvas.draw() 
        
        output_layout.addWidget(self.canvas, 1)
        
        self.feedback_box = QTextEdit()
        self.feedback_box.setReadOnly(True)
        self.feedback_box.setFont(QFont("Segoe UI", 10))
        output_layout.addWidget(self.feedback_box, 1)
        
        main_layout.addWidget(output_widget)
        
        # --- Footer Keterangan Gemini ---
        gemini_info = QLabel("Program ini menggunakan Gemini AI untuk Memberi FeedBack")
        gemini_info.setObjectName("KeteranganGemini")
        gemini_info.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(gemini_info)

    def _setup_input_tabs(self):
        """Membuat tab untuk input manual dan pemilihan standar (menggunakan QLineEdit)."""
        
        manual_tab = QWidget()
        manual_layout = QVBoxLayout(manual_tab)

        grid = QGridLayout()
        grid.addWidget(QLabel("Nama Sampel:"), 0, 0)
        self.nama_entry = QLineEdit("Kuah Sup Daging Batch C")
        grid.addWidget(self.nama_entry, 0, 1, 1, 3)
        
        validator = QIntValidator(0, 100, self) 

        for i, label_text in enumerate(kolom_rasa):
            row = (i // 2) + 1
            col_start = (i % 2) * 2
            
            grid.addWidget(QLabel(label_text), row, col_start)
            
            input_line = QLineEdit()
            input_line.setValidator(validator)
            input_line.setText("40" if 'Asin (Na+)' in label_text or 'Umami (Glutamat)' in label_text else ("5" if 'Pahit (Alkaloid)' in label_text else "20"))
            input_line.setFixedWidth(80) 
            
            self.input_fields[label_text] = input_line
            
            grid.addWidget(input_line, row, col_start + 1)
            
        manual_layout.addLayout(grid)
        manual_layout.addStretch(1)
        self.tab_widget.addTab(manual_tab, "🔬 Input Sampel Baru")

        # TAB 2: PILIH STANDAR QC
        qc_tab = QWidget()
        qc_layout = QVBoxLayout(qc_tab)
        
        qc_layout.addWidget(QLabel("<h4>Pilih Standar Rasa untuk Analisis</h4>"))
        
        self.combo_makanan = QComboBox()
        self.combo_makanan.addItems(df.index.tolist())
        self.combo_makanan.currentIndexChanged.connect(self._update_standard_data)
        qc_layout.addWidget(self.combo_makanan)
        
        qc_layout.addSpacing(10)
        qc_layout.addWidget(QLabel("<i>Nilai dari standar yang dipilih akan mengisi kotak input secara otomatis.</i>"))
        
        qc_layout.addStretch(1)
        self.tab_widget.addTab(qc_tab, "✅ Pilih Standar QC")

    @Slot(int)
    def _update_standard_data(self, index):
        """Memuat data standar ke label read-only dan kotak input."""
        makanan = self.combo_makanan.currentText()
        if not makanan:
            return

        data = df.loc[makanan, kolom_rasa] 
        
        # 1. Update Label Read-Only (Kandungan Frame)
        for rasa in kolom_rasa:
            nilai = data[rasa]
            self.data_labels[rasa].setText(f"• {rasa}: <b>{nilai}</b>")
            
        # 2. Update QLineEdit Input Otomatis
        for rasa, line_edit in self.input_fields.items():
            if rasa in data:
                nilai = int(data[rasa])
                line_edit.setText(str(nilai)) 

        self.nama_entry.setText(f"QC: {makanan}")

    @Slot()
    def _start_analysis(self):
        """Memulai pengumpulan data dan memulai thread Gemini."""
        
        self.analyze_button.setEnabled(False) 
        self.feedback_box.setText("Memanggil Gemini AI... Harap tunggu (Membutuhkan koneksi internet).")
        
        data_sensor_list = []
        try:
            for rasa in kolom_rasa:
                nilai = int(self.input_fields[rasa].text())
                data_sensor_list.append(nilai)
        except ValueError:
             QMessageBox.critical(self, "Error Input", "Semua nilai sensor harus berupa angka (0-100).")
             self.analyze_button.setEnabled(True)
             return

        nama_sampel = self.nama_entry.text()

        if sum(data_sensor_list) == 0:
             QMessageBox.critical(self, "Error", "Input sensor tidak boleh nol semua.")
             self.analyze_button.setEnabled(True)
             self.feedback_box.setText("Harap masukkan nilai sensor (>0).")
             return
             
        self._update_plot(data_sensor_list, nama_sampel)
        
        self.gemini_worker = GeminiWorker(nama_sampel, self._get_profil_persen(data_sensor_list))
        self.gemini_worker.finished.connect(self._display_gemini_result)
        self.gemini_worker.start()

    def _get_profil_persen(self, data_sensor_list):
        """Mengambil data sensor dan mengembalikan Series profil persentase."""
        profil_baru = pd.Series(data_sensor_list, index=kolom_rasa)
        total_intensitas = profil_baru.sum()
        return (profil_baru / total_intensitas) * 100

    def _update_plot(self, data_sensor_list, nama_sampel):
        """Mengupdate grafik Matplotlib dengan data baru."""
        profil_persen = self._get_profil_persen(data_sensor_list)
        
        self.ax.cla()
        colors = ['#EA4335', '#FBBC05', '#34A853', '#4285F4', '#DB4437'] 
        profil_persen.plot(kind='bar', ax=self.ax, color=colors) 
        
        self.ax.set_title(f"Grafik Profil Rasa ({nama_sampel})", fontsize=10)
        self.ax.set_ylabel("Intensitas Relatif (%)", fontsize=8)
        self.ax.tick_params(axis='x', labelsize=7)
        self.fig.tight_layout()
        self.canvas.draw()
    
    @Slot(str)
    def _display_gemini_result(self, result):
        """Menerima hasil dari thread Gemini dan menampilkannya di UI dengan format yang diminta."""
        
        nama_sampel = self.nama_entry.text()
        
        if result.startswith("ERROR"):
            self.feedback_box.setText(f"<span style='color: red;'>{result}</span>")
        else:
            # Mengganti format bold dari Gemini menjadi <b></b> dan newline menjadi <br>
            formatted_result = result.replace("\n", "<br>").replace("**", "<b>").replace("</b>", "</b>")
            
            # MENGAMBIL FRASA UTAMA UNTUK DIBOLD (FIXED)
            formatted_result = formatted_result.replace("Analisisnya :", "<b>Analisisnya :</b>")
            
            formatted_result = formatted_result.replace("Rekomendasi Industri Restoran :", "<br><br><b>Rekomendasi Industri Restoran :</b>")


            final_output = f"""
                <h3 style='color: #4285F4; margin-bottom: 5px;'>Hasil Analisis "{nama_sampel}"</h3>
                <hr style='border: 0; border-top: 1px solid #ddd; margin: 5px 0;'>
                <p>{formatted_result}</p>
            """
            self.feedback_box.setText(final_output)
            
        self.analyze_button.setEnabled(True)


if __name__ == '__main__':
    if 'GEMINI_API_KEY' not in os.environ:
         print("PENTING: Variabel lingkungan GEMINI_API_KEY tidak ditemukan. Harap atur sebelum menjalankan.")
    
    app = QApplication(sys.argv)
    window = AITastingApp()
    window.show()
    sys.exit(app.exec())