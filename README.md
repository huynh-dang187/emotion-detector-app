### Emotion Detection App

## Introduction 

Ứng dụng nhận diện cảm xúc gồm **2 giai đoạn phát triển**:

- **Phase 1**: Nhận diện cảm xúc từ văn bản do người dùng nhập.  
- **Phase 2**: Nhận diện cảm xúc qua khuôn mặt từ ảnh hoặc camera.

---

## Features

### 1. Nhận diện cảm xúc văn bản
- **Dữ liệu:** IMDb (50k review).
- **Xử lý:** làm sạch, tokenization, TF-IDF.
- **Thuật toán:** Naive Bayes, Logistic Regression.
- **Kết quả:** Accuracy >80%, phân loại tích cực / tiêu cực.
- **Giao diện:** Nhập văn bản → dự đoán cảm xúc ngay trên Streamlit.
### 2. Nhận diện cảm xúc khuôn mặt
- **Dữ liệu:** Ảnh upload, webcam realtime, chụp camera.
- **Phát hiện khuôn mặt:** OpenCV Haar Cascade.
- **Nhận diện cảm xúc:** DeepFace (CNN pretrained).
- **Hiển thị:** bounding box + label cảm xúc, tuổi, giới tính.
- **Chế độ:** Upload ảnh, webcam realtime, hoặc chụp ảnh.

# Demo

link demo phát hiện cảm xúc qua video : https://drive.google.com/file/d/1X8Uyh4wE-LfgdoXWIrAolHsjLTyb8tUr/view?usp=sharing

link demo phát hiện cảm xúc qua văn bản và hình ảnh (upload) : https://drive.google.com/file/d/1N-A5BEPDuWHLb9WagPrDKYfB0S67XXMy/view?usp=sharing


---

## 🛠️ Installation


### Clone project (kéo project về)
git clone https://github.com/huynh-dang187/emotion-detector-app.git

### Create virtual environment (tạo môi trường ảo ,không bắt buộc nhưng nên) (optional but recommended) 
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

### Install dependencies (cài đặt các gói phụ thuộc)
pip install -r requirements.txt

---


## 🚀 Usage
Run the Streamlit app:

        streamlit run final_app.py

---

## 📄 License
Distributed under the MIT License. See LICENSE for details.