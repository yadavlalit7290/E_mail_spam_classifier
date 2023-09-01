import pickle
import pandas as pd
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
import nltk
import string
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob

ps = PorterStemmer()
# nltk.download('punkt')  # Download the 'punkt' dataset if not already downloaded

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))           
    return " ".join(y)



pipe = pickle.load(open('model.pkl','rb'))


st.title('Sms Spam Classifier')
a1 = st.text_area('Enter your message')
a = TextBlob(a1).correct()  # Apply spell correction using TextBlob
a = transform_text(str(a))  
if st.button('Predict'):
    if a1 == '':
        st.write('Kuch toh likhle , mai kit te bta du')
    else:
        pred = pipe.predict([a])[0]
        if pred == 0:
            st.success('It is not Spam 😀')
        else:
            st.error('It is Spam 😥😨')