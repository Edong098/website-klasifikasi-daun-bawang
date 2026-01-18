# 🌱 Klasifikasi Daun Bawang Berbasis Citra Digital

Proyek ini merupakan aplikasi **klasifikasi daun bawang berbasis citra digital** menggunakan metode **Deep Learning (Transfer Learning)** yang diimplementasikan dalam **aplikasi web berbasis Flask**.

Model yang digunakan telah dilatih sebelumnya dan mampu melakukan prediksi kelas daun bawang berdasarkan gambar yang diunggah oleh pengguna.

---

## 📌 Fitur Utama
- Upload gambar daun bawang melalui web
- Preprocessing citra otomatis
- Prediksi kelas daun bawang menggunakan model CNN
- Tampilan web sederhana dan responsif

---

## 🧠 Model Deep Learning
- Framework: **TensorFlow / Keras**
- Input Image Size: **224 x 224**
- Model disimpan dalam format `.h5`
- Model **tidak disimpan di GitHub** untuk menjaga ukuran repository tetap ringan

### 🔗 Link Download Model (.h5)
Silakan unduh model melalui Google Drive berikut:

👉 **Download Model**  
https://drive.google.com/file/d/1XYF78LaAF7Mx1yppn8wDIPHl5yVd-9JT/view?usp=drive_link

Setelah diunduh, letakkan file:
```bash
hybrid_final.h5

📂 Struktur Folder
KLASIFIKASI_DAUN_BAWANG/
│
├── app.py
├── requirements.txt
│
├── static/
│   ├── style.css
│   ├── lottie/
│   └── uploads/        
│
├── templates/
│   └── index.html
│
└── .gitignore
├── hybrid_final.h5

⚙️ Instalasi & Menjalankan Aplikasi
1️⃣ Clone Repository
git clone https://github.com/Edong098/website-klasifikasi-daun-bawang.git
cd KLASIFIKASI_DAUN_BAWANG

2️⃣ Install Dependency
pip install -r requirements.txt

3️⃣ Download Model

Unduh file hybrid_final.h5 dari link Google Drive di atas dan simpan di folder utama project.

4️⃣ Jalankan Aplikasi
python app.py


Buka browser dan akses:
http://127.0.0.1:5000