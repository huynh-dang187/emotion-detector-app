import cv2
import av
import time
import numpy as np
import streamlit as st
from deepface import DeepFace

# WebRTC cho webcam realtime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="wide")

# ========== CSS đẹp qua Markdown ==========
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

# ========== Header ==========
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.markdown('<div class="app-title">Real-time Emotion Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">OpenCV + DeepFace · Webcam realtime (WebRTC) hoặc chụp ảnh thủ công</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ========== Sidebar ==========
with st.sidebar:
    st.header("⚙️ Cấu hình")
    mode = st.radio("Chế độ", ["Webcam realtime (WebRTC)", "Chụp ảnh (camera_input)"], index=0)
    st.caption("Nếu không thấy video, thử chế độ chụp ảnh.")
    st.divider()

    st.subheader("Phát hiện khuôn mặt")
    scaleFactor = st.slider("scaleFactor", 1.05, 1.5, 1.1, 0.01)
    minNeighbors = st.slider("minNeighbors", 3, 10, 5, 1)
    minSize_val = st.slider("minSize (px)", 20, 120, 30, 2)

    st.subheader("DeepFace")
    analyze_every = st.slider("Phân tích mỗi N frame", 1, 10, 3, 1)
    enforce_detection = st.checkbox("enforce_detection", value=False)
    st.caption("Tắt enforce_detection để tránh lỗi khi không thấy rõ gương mặt.")

# ========== Tải Cascade ==========
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ========== Hàm tiện ích ==========
EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "sad": "😢", "surprise": "😲", "neutral": "😐"
}

def draw_box_and_label(img, x, y, w, h, label):
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 85, 85), 2)
    cv2.putText(img, label, (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 85, 85), 2, cv2.LINE_AA)
    return img

# ========== Class xử lý video ==========
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.last_emotion = "detecting..."
        self.last_time = time.time()

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

            try:
                face_small = cv2.resize(roi, (224, 224))
            except Exception:
                face_small = roi

            if do_analyze:
                try:
                    res = DeepFace.analyze(
                        face_small,
                        actions=["emotion"],
                        enforce_detection=enforce_detection
                    )
                    if isinstance(res, list):
                        dom = res[0].get("dominant_emotion", "neutral")
                    else:
                        dom = res.get("dominant_emotion", "neutral")

                    self.last_emotion = dom
                except Exception:
                    self.last_emotion = "unknown"

            img = draw_box_and_label(img, x, y, w, h, self.last_emotion)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ========== Chế độ 1: Webcam realtime ==========
if mode == "Webcam realtime (WebRTC)":
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 🎥 Webcam")
    st.markdown('<div class="video-frame">', unsafe_allow_html=True)

    rtc_configuration = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

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

# ========== Chế độ 2: Chụp ảnh thủ công ==========
else:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 📸 Chụp ảnh")
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
                res = DeepFace.analyze(
                    roi, actions=["emotion"], enforce_detection=enforce_detection
                )
                if isinstance(res, list):
                    dom = res[0].get("dominant_emotion", "neutral")
                else:
                    dom = res.get("dominant_emotion", "neutral")
            except Exception:
                dom = "neutral"

            label = f"{EMOJI.get(dom, '🙂')} {dom}"
            emotions_found.append(dom)
            frame = draw_box_and_label(frame, x, y, w, h, label)

        st.markdown("##### Kết quả")
        if emotions_found:
            dom_show = max(set(emotions_found), key=emotions_found.count)
            st.markdown(f'<span class="emotion-badge">Dominant: {EMOJI.get(dom_show,"🙂")} {dom_show}</span>', unsafe_allow_html=True)
        else:
            st.info("Không phát hiện được khuôn mặt. Hãy chụp lại cận hơn, đủ sáng nhé.")

        st.markdown('<div class="video-frame" style="margin-top:10px;">', unsafe_allow_html=True)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ========== Footer ==========
st.markdown("""
<div class="app-card" style="text-align:center;">
  <small>Built with <b>Streamlit</b>, <b>OpenCV</b> & <b>DeepFace</b>. Nhấn <code>r</code> để reload nếu gặp sự cố trình duyệt.</small>
</div>
""", unsafe_allow_html=True)
