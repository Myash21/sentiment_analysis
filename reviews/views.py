# views.py
from django.shortcuts import render
from django.http import HttpResponse
import pickle
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove special characters and digits
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)

    # Remove additional spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

    return text

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Load the pickle files
with open('reviews/ml_models/count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('reviews/ml_models/naive_bayes_model.pkl', 'rb') as f:
    nb_bow = pickle.load(f)

def preprocess_and_predict(review):
    review_clean = clean_text(review)
    review_clean = lemmatize_text(review_clean)
    review_vectorized = cv.transform([review_clean]).toarray()
    prediction = nb_bow.predict(review_vectorized)
    return prediction

def index(request):
    if request.method == 'POST':
        review = request.POST['review']
        prediction = preprocess_and_predict(review)
        prediction_text = 'Positive' if prediction[0] == 1 else 'Negative'
        return render(request, 'reviews/index.html', {'prediction': prediction_text, 'review': review})
    return render(request, 'reviews/index.html')
