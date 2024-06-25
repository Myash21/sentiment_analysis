from django.shortcuts import render
import os
import pickle
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
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

# Define the path to the models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'reviews', 'ml_models')

# Load the pickle files
with open(os.path.join(MODELS_DIR, 'count_vectorizer.pkl'), 'rb') as f:
    cv = pickle.load(f)

with open(os.path.join(MODELS_DIR, 'naive_bayes_model.pkl'), 'rb') as f:
    nb_bow = pickle.load(f)

def preprocess_and_predict(review):
    review_clean = clean_text(review)
    review_clean = lemmatize_text(review_clean)
    review_vectorized = cv.transform([review_clean]).toarray()
    prediction = nb_bow.predict(review_vectorized)
    return prediction

def home(request):
    prediction = None
    if request.method == 'POST':
        review = request.POST.get('review')
        prediction = preprocess_and_predict(review)
        prediction = "Positive" if int(prediction[0]) == 1 else "Negative"
    return render(request, 'reviews/index.html', {'prediction': prediction})
