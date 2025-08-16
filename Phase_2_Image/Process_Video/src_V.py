import cv2
from deepface import DeepFace
from collections import deque, Counter

# 1️⃣ Load Haar Cascade để detect mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2️⃣ Khởi tạo webcam
cap = cv2.VideoCapture(0)

# 3️⃣ Queue để lưu cảm xúc 10 frame gần nhất
frame_window = 10
emotion_queue = deque(maxlen=frame_window)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # nếu không lấy được frame, thoát vòng lặp

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # chuyển sang grayscale để detect nhanh
    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,   # tăng độ nhạy
    minNeighbors=3       # giảm số neighbors để detect nhiều hơn
)


    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]  # cắt mặt ra
            #  ✅ Kiểm tra kích thước mặt để tránh lỗi DeepFace
            if face_img.shape[0] < 30 or face_img.shape[1] < 30:
                continue  # bỏ qua frame quá nhỏ

            try:
                # 4️⃣ Phân tích cảm xúc với DeepFace
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result['dominant_emotion']
                emotion_queue.append(dominant_emotion)

                # 5️⃣ Tính cảm xúc trung bình của 10 frame gần nhất
                if len(emotion_queue) == frame_window:
                    most_common_emotion = Counter(emotion_queue).most_common(1)[0][0]
                else:
                    most_common_emotion = dominant_emotion

                # 6️⃣ Vẽ bounding box và text emotion lên khuôn mặt
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
                cv2.putText(frame, most_common_emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            except Exception as e:
                # 7️⃣ Nếu DeepFace lỗi, in ra console để debug, không vẽ “Error” trên frame
                print("DeepFace error:", e)
                continue

    else:
        # 8️⃣ Nếu không detect được mặt, hiển thị thông báo trên frame
        cv2.putText(frame, "    ", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # 9️⃣ Hiển thị frame webcam
    cv2.imshow('Webcam Emotion Detector', frame)

    #  🔟 Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 11️⃣ Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
