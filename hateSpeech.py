import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the TF-IDF vectorizer
with open('C:/Users/durga prasad/Desktop/project/myenv/vector.pkl', 'rb') as file:
    vector = pickle.load(file)

# Load the trained model
with open('C:/Users/durga prasad/Desktop/project/myenv/naive_bayes.pkl', 'rb') as file:
    naive_bayes_model = pickle.load(file)

def predict_category(headline):
    vec = vector.transform([headline]).toarray()
    prediction = naive_bayes_model.predict(vec)[0]
    categories = {0: 'Hate', 1: 'No Hate'}
    return categories[prediction]

# Streamlit app
st.title('Hate/NoHate Prediction')

# Text input for user to enter headline
headline_input = st.text_input('Enter speech')

# Button to predict category
if st.button('Predict'):
    prediction = predict_category(headline_input)
    st.write(f'Prediction: {prediction}')
