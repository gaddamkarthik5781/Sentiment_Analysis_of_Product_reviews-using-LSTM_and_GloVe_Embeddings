import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the trained model
model = load_model("LSTM_Model-for-Sentimental_Analysis.keras")

# Load the tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

max_comment_length = 1500

# Text preprocessing function
def text_preprocessing(text):
    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"     # Emoticons
                                   u"\U0001F300-\U0001F5FF"     # Symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"     # Transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"     # Flags (iOS)
                                   u"\U00002702-\U000027B0"     # Other symbols
                                   u"\U000024C2-\U0001F251"     # Enclosed characters
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    text = text.lower()                                # Convert text to lowercase
    text = re.sub(r'\S+@\S+', '', text)                # Remove email addresses
    text = re.sub(r'\d+', '', text)                    # Remove digits
    text = re.sub(r'[^\w\s]', '', text)                # Remove punctuation
    text = text.strip()                                # Remove leading and trailing whitespace
    text = remove_emoji(text)                          # Remove emojis

    # Tokenization and removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    
    # Handling negations and emphasis
    negation_words = ["not", "no", "never", "didn't", "doesn't", "don't", "aren't", "isn't", "wasn't", "weren't", "can't", "couldn't"]
    negated = False
    result_tokens = []
    
    for idx, token in enumerate(tokens):
        if token in negation_words:
            negated = True
            result_tokens.append(token)  # Including negation word itself
        elif negated and token in stop_words:
            negated = False
        elif negated:
            result_tokens.append("not_" + token)  # Negating the following word
            negated = False
        elif token not in stop_words:
            result_tokens.append(token)
    
    # Lemmatizing the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in result_tokens if token.isalpha()]
    
    text = " ".join(tokens)

    return text

# Already calculated Accuracy of the model
model_accuracy = 0.92

# Streamlit app
st.title("Product Review Sentiment Analysis")

st.write("""
### Enter a product review to analyze its sentiment
""")

review_text = st.text_area("Review Text")

if st.button("Analyze Sentiment"):
    if review_text:
        # Preprocessing the review text
        preprocessed_review = text_preprocessing(review_text)
        
        # Tokenizing and padding the preprocessed review
        tokenized_review = tokenizer.texts_to_sequences([preprocessed_review])
        padded_review = pad_sequences(tokenized_review, maxlen=max_comment_length, padding='post')
        
        # Predicting the sentiment
        prediction = model.predict(padded_review)
        sentiment_score = prediction[0][0]
        
        # Determining sentiment label based on score
        if sentiment_score >= 0.7:
            sentiment = "Positive"
        elif sentiment_score >= 0.3:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"
        
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Sentiment Score:** {sentiment_score:.4f}")
        st.write(f"**Model Accuracy:** {model_accuracy * 100:.2f}%")
    else:
        st.write("Please enter a review text.")
