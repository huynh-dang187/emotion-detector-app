import preprocess
import pandas as pd
import string 
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix




# Đọc dữ liệu
df = pd.read_csv('IMDB Dataset.csv')
# Làm sạch text
df['clean_text'] = df['review'].str.lower().str.translate(str.maketrans('', '', string.punctuation))
# Áp dụng hàm preprocess cho từng dòng
df['processed'] = df['clean_text'].apply(preprocess.preprocess)

# # Test hàm cho 1 câu riêng
# text = "This is a great movie i saw"

# print(preprocess.preprocess(text))

vectorizer = TfidfVectorizer() # khởi tạo TF-idf 

X = vectorizer.fit_transform(df['processed'])

from sklearn.model_selection import train_test_split

y = df['sentiment']  # hoặc 0/1 nếu đã chuyển sang số
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# 8. Lưu model & vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print(" Model và vectorizer đã được lưu thành công!")