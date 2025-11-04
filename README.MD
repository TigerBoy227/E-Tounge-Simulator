🧪 AI Mencicipi: E-Tongue QC Simulator
	Proyek ini adalah simulasi aplikasi desktop Quality Control (QC) Rasa untuk industri restoran. Aplikasi ini menggantikan electronic tongue (E-Tongue) fisik dengan input data sensor digital dan menggunakan Google Gemini AI untuk menghasilkan feedback naratif yang detail dan profesional tentang konsistensi rasa makanan. Aplikasi dibangun menggunakan Python dan framework GUI PySide6 dengan estetika modern.

✨ Fitur Utama
1. Aplikasi Desktop: Dibangun dengan PySide6 untuk kinerja yang stabil dan UI modern.
2. Gemini AI Integration: Menggunakan Gemini AI untuk menghasilkan feedback naratif ala kritikus makanan.
3. Simulasi QC: Pengguna dapat memilih dari 50 Standar Rasa Makanan Indonesia atau memasukkan data sensor sampel baru.
4. Asynchronous Processing: Menggunakan threading (QThread) untuk menjaga UI tetap responsif saat menunggu respons dari Gemini API.

🛠️ Persyaratan Sistem & Instalasi
	Proyek ini membutuhkan Python 3.8+ dan beberapa library utama, serta konfigurasi kunci API.
1. Instalasi Lingkungan
Buka terminal di direktori proyek dan instal semua dependencies:
Bashpip install -r requirements.txt
2. Konfigurasi Kunci API
PENTING: Kunci API Anda harus diatur sebagai variabel lingkungan (GEMINI_API_KEY) di sistem Anda. Jangan pernah mengunggah kunci ini ke GitHub.
Kunci Anda: AIzaSyCpQRGo0yWjxo-Bo282yJdEQFp51IEvP8M
Cara Konfigurasi (Sebelum Menjalankan Program):
SistemPerintah (Gunakan kunci Anda)
Windows PowerShell$env:GEMINI_API_KEY="AIzaSyCpQRGo0yWjxo-Bo282yJdEQFp51IEvP8M"
Linux/macOS (Bash/Zsh)export GEMINI_API_KEY="AIzaSyCpQRGo0yWjxo-Bo282yJdEQFp51IEvP8M"

▶️ Cara Menjalankan Aplikasi
	Setelah konfigurasi API selesai di terminal yang sama:
python E-Tounge Simulator.py
Aplikasi desktop akan terbuka, dan Anda bisa mulai menganalisis rasa!