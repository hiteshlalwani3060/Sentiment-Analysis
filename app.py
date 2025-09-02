import pickle
import numpy as np
import streamlit as st
from gensim.models import Word2Vec, KeyedVectors

@st.cache_resource
def load_word2vec_model():
    model_for_word_to_vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True,limit=500000)
    return model_for_word_to_vec


@st.cache_resource
def load_model():
    with open('Review_sentiment.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
        
def preprocess_text(text ,model_for_word_to_vec):
    tokens = text.split()
    vec = [model_for_word_to_vec[word] for word in tokens if word in model_for_word_to_vec]
    return np.mean(vec, axis=0).reshape(1,-1) if vec else np.zeros((1,300))

model_for_word_to_vec = load_word2vec_model()
model= load_model()

# main app
st.title('Sentiment Analysis')
user_input = st.text_area("Enter your review:")
if st.button('Predict Sentiment'):
    x = preprocess_text(user_input, model_for_word_to_vec)
    pred= model.predict(x)[0]
    sent_map = {0:"Negative", 1:"Neutral", 2:"Positive"}
    st.write(f"Predicted Sentiment: **{sent_map[pred]}**")