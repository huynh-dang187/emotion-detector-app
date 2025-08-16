import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np

st.set_page_config(page_title="Emotion, Age & Gender Detector", layout="centered")

st.markdown("""
    <h1 style='text-align: center; 
               font-size: 42px; 
               background: -webkit-linear-gradient(45deg, #ff6b6b, #5f27cd, #1dd1a1); 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;'>
        Nhận Diện Cảm Xúc Qua Hình Ảnh
    </h1>
""", unsafe_allow_html=True)

# Upload ảnh
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Đọc ảnh từ file upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Resize chuẩn
    img_resized = cv2.resize(img, (224, 224))

    # Phân tích bằng DeepFace
    with st.spinner("Đang phân tích..."):
        result_list = DeepFace.analyze(
            img_resized,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False,
            detector_backend='mtcnn'
        )
        result = result_list[0]

    # Thêm CSS tùy chỉnh
    st.markdown("""
        <style>
        .result-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .result-title {
            font-size: 22px;
            font-weight: bold;
            color: #333333;
        }
        .result-value {
            font-size: 28px;
            font-weight: bold;
            color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)

    # Chia kết quả thành 3 cột
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="result-box">
                <div class="result-title">Emotion</div>
                <div class="result-value">{result['dominant_emotion'].title()}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="result-box">
                <div class="result-title">Age</div>
                <div class="result-value">{result['age']}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="result-box">
                <div class="result-title">Gender</div>
                <div class="result-value">{result['dominant_gender'].title()}</div>
            </div>
        """, unsafe_allow_html=True)
