#### ->>>>>>>> Phase 1 <<<<<<<-####
# IMDb Sentiment Analysis

## 1. Mục tiêu
Xây dựng mô hình phân loại cảm xúc (tích cực / tiêu cực) từ đánh giá phim trên IMDb.

## 2. Quy trình xử lý
1. Load dữ liệu từ file CSV
2. Làm sạch văn bản: lowercase, loại bỏ dấu câu
3. Tokenize bằng NLTK
4. Loại bỏ stopwords
5. Vector hóa bằng TF-IDF
6. Chia dữ liệu train/test (80/20)
7. Huấn luyện mô hình Naive Bayes (MultinomialNB)
8. Đánh giá bằng accuracy và confusion matrix

## 3. Kết quả
- Accuracy: **0.8699**
- Confusion Matrix: [[4390, 571],
                    [ 730, 4309]]

## 4. Thư viện sử dụng
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- joblib


#### ->>>>>>>> Phase 2 <<<<<<<<-####

## 1. Mục tiêu
Phân loại cảm xúc (tích cực / tiêu cực) của người dùng qua Ảnh và Video
## 2. Quy trình xử lý
-
-
-
## 3. Kết quả
-
-
-
## 4. Thư viện sử dụng
-matplotlib 
-numpy
-opencv


## 5.Kế hoạch   
 # Kế hoạch đầu tiên (Nhận diện cảm xúc qua Text do người dùng nhập (29/7 - 8/8 ))
 # Kế hoạch tiếp theo cho ngày (9/8/2025)
- Tích hợp mô hình vào ứng dụng Streamlit
- Cho phép người dùng nhập câu và nhận kết quả cảm xúc    
- Tích hợp thêm lịch sử nhập và biểu đồ tỉ lệ             
 # Kế hoạch tiếp theo cho ngày (10/8/2025)
 -Tìm hiểu thêm về OpenCV 
 # Kế hoạch tiếp theo cho ngày (14/8/2025)
 -Dùng OpenCV đọc file ảnh và xử lí ảnh (mở,đọc,chỉnh kích thước ảnh ...)
    + Học thêm về BGR và RGB (H,W,C)
 -Dùng OpenCV mở video 
    + Học cv.cvtColor() để chuyển ảnh sang grayscale.
 # Kế hoạch tiếp theo cho ngày (15/8/2025)
    + Tải file Haar Cascade và thử phát hiện khuôn mặt từ webcam


 