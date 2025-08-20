# app.py
import cv2
import av
import time
import string
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from deepface import DeepFace

# Text/NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# WebRTC
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration


# ==============================
# Setup trang + CSS
# ==============================
st.set_page_config(page_title="Emotion Detector (All-in-One)", page_icon="😊", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"]  {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial;
}
.app-card {
  background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
  border: 1px solid #eeeeee;
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.05);
  padding: 22px;
  margin: 10px 0 20px 0;
}
.app-title {
  font-size: 32px;
  font-weight: 800;
  letter-spacing: 0.2px;
  background: linear-gradient(90deg, #111827, #4b5563);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 6px;
}
.app-subtitle {
  color: #6b7280;
  font-size: 15px;
  margin-bottom: 18px;
}
.emotion-badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 9999px;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
  font-weight: 700;
  letter-spacing: 0.3px;
}
.video-frame {
  border: 2px solid #f3f4f6;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
section[data-testid="stSidebar"] > div {
  background: #0b1220;
}
section[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.markdown('<div class="app-title">Emotion Detector · All-in-One</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Text • Image • Webcam (WebRTC) • Camera Input</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ==============================
# Sidebar: Menu + Cấu hình
# ==============================
with st.sidebar:
    st.header("Menu chức năng")
    app_mode = st.radio(
        "Chọn chế độ:",
        (
            "💬 Văn bản",
            "🖼 Ảnh upload",
            "🎥 Webcam realtime (WebRTC)",
            "📸 Chụp ảnh thủ công"
        ),
        index=0
    )

    st.divider()
    st.subheader("⚙️ Phát hiện khuôn mặt (Haar)")
    scaleFactor = st.slider("scaleFactor", 1.05, 1.5, 1.10, 0.01)
    minNeighbors = st.slider("minNeighbors", 3, 10, 5, 1)
    minSize_val = st.slider("minSize (px)", 20, 120, 30, 2)

    st.subheader("🤖 DeepFace")
    analyze_every = st.slider("Phân tích mỗi N frame", 1, 15, 3, 1)
    


# ==============================
# Chuẩn bị tài nguyên
# ==============================
# NLTK: đảm bảo có dữ liệu
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    _ = nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

STOP_WORDS = set(stopwords.words('english'))

@st.cache_resource(show_spinner=False)
def load_text_models():
    try:
        model = joblib.load("../Save_Model copy/sentiment_model.pkl")
        vectorizer = joblib.load("../Save_Model copy/tfidf_vectorizer.pkl")
        return model, vectorizer, None
    except Exception as e:
        return None, None, e

def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ==============================
# Utils
# ==============================
def draw_box_and_label(img, x, y, w, h, label):
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 85, 85), 2)
    cv2.putText(img, str(label), (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 85, 85), 2, cv2.LINE_AA)
    return img


# ==============================
# 1) Văn bản (Text Sentiment)
# ==============================
if app_mode == "💬 Văn bản":
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("📝 Nhận diện cảm xúc qua Văn bản")

    model, vectorizer, load_err = load_text_models()
    if load_err is not None or model is None or vectorizer is None:
        st.warning("⚠️ Không thể load model/vectorizer. Kiểm tra đường dẫn `../Save_Model copy/...`")
    else:
        if "history" not in st.session_state:
            st.session_state.history = []

        sentence = st.text_input("Nhập câu cần phân tích:")

        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Phân loại cảm xúc"):
                if sentence.strip():
                    processed = preprocess(sentence)
                    vec = vectorizer.transform([processed])
                    pred = model.predict(vec)[0]

                    if pred == "positive":
                        st.success("Kết quả: Positive ")
                    elif pred == "negative":
                        st.error("Kết quả: Negative ")
                    else:
                        st.info("Kết quả: Neutral ")

                    st.session_state.history.append((sentence, pred))
                else:
                    st.warning("Vui lòng nhập câu trước khi phân loại!")

        with colB:
            if st.button("Xóa lịch sử"):
                st.session_state.history = []

        if st.session_state.history:
            st.subheader("📖 Lịch sử")
            df_hist = pd.DataFrame(st.session_state.history, columns=["Câu", "Kết quả"])
            st.table(df_hist)
            st.bar_chart(df_hist["Kết quả"].value_counts())

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================
# 2) Ảnh upload (DeepFace)
# ==============================
elif app_mode == "🖼 Ảnh upload":
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("👤 Nhận diện cảm xúc qua Ảnh")

    uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, channels="BGR", caption="Ảnh đã upload", use_container_width=True)

        with st.spinner("Đang phân tích..."):
            try:
                res_list = DeepFace.analyze(
                    img,
                    actions=['emotion', 'age', 'gender'],
                    
                )
                result = res_list[0] if isinstance(res_list, list) else res_list
            except Exception as e:
                st.error(f"Lỗi phân tích: {e}")
                st.stop()

        # Hiển thị kết quả
        col1, col2, col3 = st.columns(3)
        col1.metric("Emotion", str(result.get('dominant_emotion', 'unknown')).title())
        col2.metric("Age", result.get('age', '—'))
        col3.metric("Gender", str(result.get('dominant_gender', 'unknown')).title())

        # Vẽ khung khuôn mặt nếu có box
        if "region" in result and isinstance(result["region"], dict):
            x, y = result["region"].get("x", 0), result["region"].get("y", 0)
            w, h = result["region"].get("w", 0), result["region"].get("h", 0)
            if w and h:
                img_box = img.copy()
                draw_box_and_label(img_box, x, y, w, h, result.get('dominant_emotion', 'unknown'))
                st.markdown('<div class="video-frame" style="margin-top:10px;">', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================
# 3) Webcam realtime (WebRTC)
# ==============================
elif app_mode == "🎥 Webcam realtime (WebRTC)":
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("🎥 Realtime Emotion Detection (WebRTC)")

    st.caption("Mẹo: tăng `Phân tích mỗi N frame` để mượt hơn; giảm để chính xác hơn.")

    class EmotionTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0
            self.last_emotion = "detecting..."

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=(minSize_val, minSize_val)
            )

            self.frame_count += 1
            do_analyze = (self.frame_count % analyze_every == 0)

            for (x, y, w, h) in faces:
                roi = img[y:y+h, x:x+w]

                # Resize nhỏ để speed up DeepFace
                try:
                    roi_small = cv2.resize(roi, (224, 224))
                except Exception:
                    roi_small = roi

                if do_analyze:
                    try:
                        res = DeepFace.analyze(
                            roi_small,
                            actions=["emotion"],
                            
                        )
                        dom = res[0].get("dominant_emotion", "neutral") if isinstance(res, list) \
                              else res.get("dominant_emotion", "neutral")
                        self.last_emotion = dom
                    except Exception:
                        self.last_emotion = "unknown"

                draw_box_and_label(img, x, y, w, h, self.last_emotion)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    st.markdown('<div class="video-frame">', unsafe_allow_html=True)
    webrtc_streamer(
        key="emotion-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=EmotionTransformer,
        async_processing=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================
# 4) Chụp ảnh thủ công (camera_input)
# ==============================
elif app_mode == "📸 Chụp ảnh thủ công":
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("📸 Chụp ảnh và phân tích")

    img_file = st.camera_input("Chụp khuôn mặt của bạn", label_visibility="collapsed")

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=(minSize_val, minSize_val)
        )

        emotions_found = []
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            try:
                # Resize giống chế độ realtime để tăng ổn định
                roi_resized = cv2.resize(roi, (224,224))
            except Exception:
                roi_resized = roi

            try:
                res = DeepFace.analyze(
                    roi_resized,
                    actions=["emotion"],
                    
                )
                if isinstance(res, list):
                    dom = res[0].get("dominant_emotion", "neutral")
                else:
                    dom = res.get("dominant_emotion", "neutral")
            except Exception:
                dom = "neutral"

            label = dom
            emotions_found.append(dom)
            frame = draw_box_and_label(frame, x, y, w, h, label)

        st.markdown("##### Kết quả")
        if emotions_found:
            dom_show = max(set(emotions_found), key=emotions_found.count)
            st.markdown(
                f'<span class="emotion-badge">Dominant: {dom_show}</span>',
                unsafe_allow_html=True
            )
        else:
            st.info("Không phát hiện được khuôn mặt. Hãy chụp lại cận hơn, đủ sáng nhé.")

        st.markdown('<div class="video-frame" style="margin-top:10px;">', unsafe_allow_html=True)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================
# Footer
# ==============================
st.markdown("""
<div class="app-card" style="text-align:center;">
  <small>Built with <b>Streamlit</b>, <b>OpenCV</b> & <b>DeepFace</b> · 
  Với Webcam: dùng WebRTC. Nếu gặp sự cố trình duyệt, nhấn <code>r</code> để reload.</small>
</div>
""", unsafe_allow_html=True)
