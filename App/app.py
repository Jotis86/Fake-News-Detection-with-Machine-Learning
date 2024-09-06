import pandas as pd
import streamlit as st
import string
import nltk
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load the CSV file
print("Loading the CSV file...")
df = pd.read_csv("C:\\Users\\juane\\OneDrive\\Escritorio\\Datos\\features_labels.csv")
print("CSV file loaded.")

# Check the DataFrame columns
print("DataFrame columns:", df.columns)

# Separate features and labels
X = df.drop(columns=['label']).values
y = df['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
print("Training the RandomForest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("RandomForest model trained.")

def clean_word(word: str) -> str:
    """Remove punctuation and lowercase a word"""
    return re.sub(f'[{string.punctuation}]', '', word.lower().strip())

def clean_text(text: str) -> list[str]:
    """Remove stop words and punctuation from a whole text."""
    return [clean_word(word) for word in text.split() if clean_word(word) not in stop_words]

st.title("Fake News Detector")
st.subheader("Detecting fake news with machine learning")

text_to_predict = st.text_area("Enter the news to check if it is fake or not.")
button = st.button("Analyze")

if button:
    st.info("Cleaning text...")
    text_to_predict_clean = clean_text(text_to_predict)
    st.info("Vectorizing text...")
    # Here you can add your own logic to vectorize the input text
    # For example, you could use a pre-trained model or a different vectorization technique
    text_to_predict_vectorized = np.zeros((1, X_train.shape[1]))  # Placeholder for vectorization
    st.info("Classifying text...")
    is_real = rf_model.predict(text_to_predict_vectorized)[0]

    if is_real:
        st.success("Is real!")
    else:
        st.error("Is fake!")
