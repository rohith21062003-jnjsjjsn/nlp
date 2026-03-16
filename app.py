import streamlit as st
import pickle

model = pickle.load(open("sentiment_model.pkl","rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl","rb"))

st.title("Sentiment Analysis App")

text = st.text_area("Enter Review")

if st.button("Predict"):
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    st.write("Sentiment:", result[0])
