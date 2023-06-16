import streamlit as st
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download('punkt')
st.title(":blue[Twitter sentiment analysis](😄😞😐🤔)")
st.subheader('Single tweet classification :')
st.write('  * Positive sentiment  :😄')
st.write('  * negative sentiment  :😞')
st.write('  * neutral sentiment   :😐' )
st.write('  *   irrelevant sentiment:🤔')
tweet_input=st.text_input("enter tweet :")
model=pickle.load(open('twitter_logist_clf.pickle', "rb"))
vectorizer=pickle.load(open('bow_counts.pickle',"rb"))
if st.button('analyse'):
    st.write("prediction:")
    bow=vectorizer.transform([tweet_input])
    sentiment=model.predict(bow)
    if sentiment=='Negative':
        st.write('Negative')
        emoji="😞" 
        st.markdown(f"<span style='font-size: 60px'>{emoji}</span>", unsafe_allow_html=True)
    elif sentiment=='Positive':
        st.write('Positive')
        emoji="😀"
        st.markdown(f"<span style='font-size: 60px'>{emoji}</span>", unsafe_allow_html=True)
    elif sentiment=='Neutral':
        st.write('Neutral')
        emoji="😐"
        st.markdown(f"<span style='font-size: 60px'>{emoji}</span>", unsafe_allow_html=True)
    else:
        st.write('Irrelevant')
        emoji="🤔"
        st.markdown(f"<span style='font-size: 60px'>{emoji}</span>", unsafe_allow_html=True)
