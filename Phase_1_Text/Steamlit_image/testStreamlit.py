import streamlit as st
import pandas as pd
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --------------------------------------------------------
# |      😀 ỨNG DỤNG PHÂN LOẠI CẢM XÚC                   |
# --------------------------------------------------------
# | Mục đích: Xác định cảm xúc của câu văn bạn nhập vào   |
# --------------------------------------------------------
# | Nhập câu của bạn: [ Textbox ]                         |
# |                                                       |
# | [ Nút: Phân loại cảm xúc ]                            |
# --------------------------------------------------------
# | 📌 Kết quả:                                           |
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


st.title("ỨNG DỤNG PHÂN LOẠI CẢM XÚC")

st.header("Mục đích: Xác định cảm xúc của câu văn bạn nhập vào")
user_input = st.text_input("Nhập câu của bạn: ")

#Xử lí 
if st.button("Phân loại cảm xúc"):
  if user_input.strip() != "":
        # 4. Tiền xử lý
        processed_text = preprocess(user_input)

        # 5. Vector hóa
        vector_input = vectorizer.transform([processed_text])

        # 6. Dự đoán
        prediction = model.predict(vector_input)[0]

        if prediction == "positive":
            st.success("Positive 😍")
        elif prediction == "negative":
            st.error("Negative 😡")
        else:
            st.info("Neutral 😐") 
else:
        st.warning("Vui lòng nhập câu trước khi phân loại!")
    

