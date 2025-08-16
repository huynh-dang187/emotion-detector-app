# app.py
import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ==============================
# Load model + vectorizer
# ==============================
model = joblib.load("../Save_Model copy/sentiment_model.pkl")
vectorizer = joblib.load("../Save_Model copy/tfidf_vectorizer.pkl")

# Load stopwords 1 lần thôi
STOP_WORDS = set(stopwords.words('english'))

def preprocess(text):
    """Tiền xử lý văn bản"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)

# ==============================
# Sidebar
# ==============================
st.sidebar.title("Menu")
app_mode = st.sidebar.radio(
    "Chọn chế độ:",
    ("Nhận diện cảm xúc qua văn bản", "Nhận diện cảm xúc qua hình ảnh")
)

# ==============================
# Ứng dụng 1: Text Sentiment
# ==============================
if app_mode == "Nhận diện cảm xúc qua văn bản":
    st.title("📝 Ứng dụng Nhận diện Cảm xúc qua Văn bản")

    if "history" not in st.session_state:
        st.session_state.history = []

    sentence = st.text_input("Nhập câu của bạn:")

    if st.button("Phân loại cảm xúc"):
        if sentence.strip():
            processed_text = preprocess(sentence)
            vector_input = vectorizer.transform([processed_text])
            prediction = model.predict(vector_input)[0]

            if prediction == "positive":
                st.success("Positive 😍")
            elif prediction == "negative":
                st.error("Negative 😡")
            else:
                st.info("Neutral 😐")

            st.session_state.history.append((sentence, prediction))
        else:
            st.warning("⚠️ Vui lòng nhập câu trước khi phân loại!")

    if st.session_state.history:
        st.subheader("📖 Lịch sử")
        df_history = pd.DataFrame(st.session_state.history, columns=["Câu", "Kết quả"])
        st.table(df_history)

        st.bar_chart(df_history["Kết quả"].value_counts())

# ==============================
# Ứng dụng 2: Image Emotion
# ==============================
elif app_mode == "Nhận diện cảm xúc qua hình ảnh":
    st.markdown("""
        <h1 style='text-align: center; 
                   font-size: 42px; 
                   background: -webkit-linear-gradient(45deg, #ff6b6b, #5f27cd, #1dd1a1); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent;'>
            👤 Nhận diện Cảm xúc qua Hình ảnh
        </h1>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)

        with st.spinner("Đang phân tích..."):
            result_list = DeepFace.analyze(
                img,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False,
                detector_backend='mtcnn'
            )
            # Fix trường hợp trả về dict
            result = result_list[0] if isinstance(result_list, list) else result_list

        # CSS
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
