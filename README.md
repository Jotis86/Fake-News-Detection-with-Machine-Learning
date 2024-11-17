# 📰 Fake News Detection 📰

![Portada](images/imagen.jpg)

## 🎯 Objectives
The goal of this project is to develop a machine learning model that can accurately classify news articles as either fake or real. This helps in combating misinformation and promoting reliable information sources. 🌐

## ⚙️ Functionality
- Load and preprocess news datasets. 📂
- Clean and vectorize text data. 🧹
- Train multiple machine learning models. 🤖
- Evaluate and compare model performance. 📊
- Visualize results and metrics. 📈
- Provide a user interface for real-time news classification. 🖥️

## 🛠️ Tools Used
- **Python** 🐍
- **Pandas** for data manipulation 📊
- **NLTK** for text processing 📝
- **Gensim** for word embeddings 🌐
- **Scikit-learn** for machine learning models 🤖
- **Matplotlib** and **Seaborn** for data visualization 📈
- **Streamlit** for creating the web app 🌐

## 🛠️ Development Process
1. **Data Loading**: Load fake and real news datasets. 📂
2. **Data Cleaning**: Remove stop words and punctuation. 🧹
3. **Text Vectorization**: Use Word2Vec to convert text to vectors. 🌐
4. **Model Training**: Train Random Forest, Decision Tree, and K-Nearest Neighbors models. 🤖
5. **Model Evaluation**: Assess models using accuracy, precision, recall, and F1-score. 📊
6. **Visualization**: Create bar plots to visualize model performance. 📈
7. **Web App Development**: Use Streamlit to create a user-friendly interface for real-time news classification. 🖥️

## 📊 Results
The models were evaluated on various metrics, and the results were visualized using bar plots. The Random Forest model showed the highest accuracy and F1-score. 🌟

## 📈 Visualizations
To better understand the performance of each model, we created several bar plots using Matplotlib and Seaborn. These visualizations include:

- **Accuracy**: This plot shows the accuracy of each model, indicating how often the model correctly classifies news articles. 📊
- **Precision**: This plot illustrates the precision of each model, showing the proportion of true positive results among all positive results predicted by the model. 🎯
- **Recall**: This plot displays the recall of each model, representing the proportion of true positive results among all actual positive cases. 🔍
- **F1-Score**: This plot combines precision and recall into a single metric, providing a balanced measure of the model's performance. ⚖️

These visualizations help in comparing the models and understanding their strengths and weaknesses in classifying fake and real news. 📈

## 🤖 Models and Metrics
- **Random Forest** 🌲
  - Accuracy: 0.9597 📊
  - Precision: 0.9619 🎯
  - Recall: 0.9524 🔍
  - F1-Score: 0.9572 ⚖️
- **Decision Tree** 🌳
  - Accuracy: 0.9105 📊
  - Precision: 0.9229 🎯
  - Recall: 0.8846 🔍
  - F1-Score: 0.9033 ⚖️
- **K-Nearest Neighbors** 👥
  - Accuracy: 0.9409 📊
  - Precision: 0.9252 🎯
  - Recall: 0.9520 🔍
  - F1-Score: 0.9384 ⚖️

## 🗂️ Project Structure
- App
- Data
- Notebook Cleaning
- Notebook Machine Learning

## 🌐 Web App
The project includes a Streamlit web application that allows users to input news articles and classify them as fake or real in real-time. The app performs the following steps:

1. **Text Input**: Users can enter the news text they want to classify. 📝
2. **Text Cleaning**: The app cleans the input text by removing stop words and punctuation. 🧹
3. **Text Vectorization**: The cleaned text is vectorized using a pre-trained model. 🌐
4. **Classification**: The vectorized text is classified using the trained Random Forest model. 🤖
5. **Result Display**: The app displays whether the news is real or fake. 🖥️

## 📬 Contact
For any questions or suggestions, feel free to reach out! 📧

- **Name**: Juan Duran Bon
- **Email**: jotaduranbon@gmail.com



