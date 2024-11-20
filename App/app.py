import pandas as pd
import streamlit as st
import string
import nltk
import re
import numpy as np
import joblib

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load the pre-trained RandomForest model
rf_model = joblib.load('Notebook_Machine Learning/random_forest_model.pkl')

def clean_word(word: str) -> str:
    """Remove punctuation and lowercase a word"""
    return re.sub(f'[{string.punctuation}]', '', word.lower().strip())

def clean_text(text: str) -> list[str]:
    """Remove stop words and punctuation from a whole text."""
    return [clean_word(word) for word in text.split() if clean_word(word) not in stop_words]

# Set up the sidebar with radio buttons
st.sidebar.title("Menú")
st.sidebar.image("C:\\Users\\juane\\OneDrive\\Escritorio\\Proyectos Python\\Project Fake News\\images\\Cat.png", use_column_width=True)
menu_option = st.sidebar.radio("Selecciona una opción", ["Objetivos del Proyecto", "Analizador de Texto"])

# Display the image at the top
st.image("C:\\Users\\juane\\OneDrive\\Escritorio\\Proyectos Python\\Project Fake News\\images\\imagen.jpg", use_column_width=True)

# Display content based on the menu selection
if menu_option == "Objetivos del Proyecto":
    st.title("Objetivos del Proyecto")
    st.write("""
    - Desarrollar un modelo de machine learning que pueda clasificar con precisión los artículos de noticias como falsos o reales.
    - Ayudar a combatir la desinformación y promover fuentes de información confiables.

    ## Metodología del Proyecto
    1. **Carga de Datos**: Se cargan los conjuntos de datos de noticias falsas y reales.
    2. **Limpieza de Datos**: Se eliminan las palabras vacías y la puntuación de los textos.
    3. **Vectorización de Textos**: Se utilizan técnicas de vectorización como Word2Vec para convertir los textos en vectores numéricos.
    4. **Entrenamiento de Modelos**: Se entrenan varios modelos de machine learning, incluyendo Random Forest, Decision Tree y K-Nearest Neighbors.
    5. **Evaluación de Modelos**: Se evalúan los modelos utilizando métricas como precisión, recall, F1-score y matriz de confusión.
    6. **Visualización de Resultados**: Se crean gráficos para visualizar el rendimiento de los modelos.
    7. **Desarrollo de la Aplicación Web**: Se utiliza Streamlit para crear una interfaz de usuario que permita la clasificación de noticias en tiempo real.
    """)
elif menu_option == "Analizador de Texto":
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
        text_to_predict_vectorized = np.zeros((1, rf_model.n_features_in_))  # Placeholder for vectorization
        st.info("Classifying text...")
        is_real = rf_model.predict(text_to_predict_vectorized)[0]

        if is_real:
            st.success("Is real!")
        else:
            st.error("Is fake!")
