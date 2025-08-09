import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

df = pd.read_csv('IMDB Dataset.csv')
stop_words = set(stopwords.words('english'))

import string


def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)