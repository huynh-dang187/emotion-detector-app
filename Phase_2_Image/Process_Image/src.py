from deepface import DeepFace
import cv2

# Load ảnh
img_path = "Image_to_test/img1.jpg"
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Không tìm thấy ảnh hoặc đường dẫn sai!")

# Resize chuẩn
img = cv2.resize(img, (224, 224))

# Dự đoán cảm xúc
result_list = DeepFace.analyze(img, actions=['emotion',"age","gender"], enforce_detection=False, detector_backend='mtcnn')

# Lấy phần tử đầu tiên của list
result = result_list[0]

# In ra dominant emotion
print("Dominant emotion:", result['dominant_emotion'])

print("Age:", result['age'])

print("Gender:", result['dominant_gender'])
