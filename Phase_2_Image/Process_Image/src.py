import cv2
from deepface import DeepFace

# Load bộ phát hiện khuôn mặt Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Bắt đầu thu hình từ webcam
cap = cv2.VideoCapture(0)

while True:
    # Ghi lại từng khung hình
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được frame từ camera/video!")
        break

    # Chuyển frame sang grayscale để phát hiện khuôn mặt
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt trong frame
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # Cắt vùng khuôn mặt từ frame gốc (màu)
        face_roi = frame[y:y + h, x:x + w]

        # Phân tích cảm xúc với DeepFace
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        emotion = result[0]['dominant_emotion']

        # Vẽ khung + hiển thị cảm xúc
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Hiển thị video (cả khi chưa nhận diện được mặt)
    cv2.imshow('Real-time Emotion Detection', frame)

    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ sau khi thoát
cap.release()
cv2.destroyAllWindows()
