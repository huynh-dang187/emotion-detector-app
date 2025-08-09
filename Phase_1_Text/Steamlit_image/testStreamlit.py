import streamlit as st
import pandas as pd
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Đọc file CSS với UTF-8
with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Ứng dụng Nhận diện Cảm xúc")
# --------------------------------------------------------
# |             ỨNG DỤNG PHÂN LOẠI CẢM XÚC                |
# --------------------------------------------------------
# | Mục đích: Xác định cảm xúc của câu văn bạn nhập vào   |
# --------------------------------------------------------
# | Nhập câu của bạn: [ Textbox ]                         |
# |                                                       |
# | [ Nút: Phân loại cảm xúc ]                            |
# --------------------------------------------------------
# | Kết quả:                                           |
# |    - Nhãn dự đoán: Positive / Negative / Neutral      |
# |    - Xác suất: 85%                                    |
# --------------------------------------------------------
# | Thêm: (tùy chọn)                                      |
# |   - Lịch sử các câu đã nhập                           |
# |   - Biểu đồ cột tỉ lệ dự đoán                         |
# --------------------------------------------------------
model = joblib.load("../Save_Model/sentiment_model.pkl")
vectorizer = joblib.load("../Save_Model/tfidf_vectorizer.pkl")

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)




st.header("Mục đích: Xác định cảm xúc của câu văn bạn nhập vào")

# --- Khởi tạo lịch sử ---
if "history" not in st.session_state:
    st.session_state.history = []  # [(text, label)]
    
sentence = st.text_input("Nhập câu của bạn:")

if st.button("Phân loại cảm xúc"):
    if sentence.strip():
        # 1. Tiền xử lý
        processed_text = preprocess(sentence)

        # 2. Vector hóa
        vector_input = vectorizer.transform([processed_text])

        # 3. Dự đoán
        prediction = model.predict(vector_input)[0]

        # 4. Hiển thị kết quả
        if prediction == "positive":
            st.success("Positive 😍")
        elif prediction == "negative":
            st.error("Negative 😡")
        else:
            st.info("Neutral 😐")  # nếu có train trung tính

        # 5. Lưu lịch sử
        st.session_state.history.append((sentence, prediction))
    else:
        st.warning("Vui lòng nhập câu trước khi phân loại!")
        
#  Hiển thị lịch sử 
if st.session_state.history:
    st.subheader(" Lịch sử các câu đã nhập")
    df_history = pd.DataFrame(st.session_state.history, columns=["Câu", "Kết quả"])
    st.table(df_history)

# Vẽ biểu đồ 
    st.subheader(" Tỉ lệ dự đoán")
    
    counts = df_history["Kết quả"].value_counts()
    st.bar_chart(counts)
    
