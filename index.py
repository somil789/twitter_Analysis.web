import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources (if not already downloaded)
nltk.download('vader_lexicon')

# Function to get tweet length
def get_tweet_length(tweet):
    return len(tweet)

# Function to perform sentiment analysis
def analyze_sentiment(tweet):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(tweet)['compound']
    if sentiment_score==0:
        return 'Negative'
    else:
        return 'Positive'
    

# Function to read CSV file and analyze tweets
def analyze_tweets(csv_file):
    df = pd.read_csv(csv_file)
    df['Tweet Length'] = df['tweet'].apply(get_tweet_length)
    df['Sentiment'] = df['tweet'].apply(analyze_sentiment)
    sentiment_counts = df['Sentiment'].value_counts()
    return df, sentiment_counts

# Page Design
st.title("Twitter Sentiment Analysis")
st.write("This web app analyzes the sentiment of tweets from the provided CSV file.")

# Introduction
st.write("""
### Introduction
This web application analyzes the sentiment of tweets from a provided CSV file. 
""")

# About
st.write("""
### About
This web application is developed using Streamlit, NLTK, Pandas, and Matplotlib. It performs sentiment analysis on tweets from a CSV file and displays the results graphically.
""")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

# Perform analysis if file is uploaded
if uploaded_file is not None:
    st.write("### Analysis Results")
    df, sentiment_counts = analyze_tweets(uploaded_file)

    # Display DataFrame
    st.write("#### DataFrame:")
    st.write(df)

    # Display Sentiment Counts
    st.write("#### Sentiment Counts:")
    st.write(sentiment_counts)

    # Plotting
st.write("#### Sentiment Analysis Graph:")
fig, ax = plt.subplots(figsize=(10, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'], ax=ax)
ax.set_title('Sentiment Analysis of Tweets')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Number of Tweets')
ax.tick_params(axis='x', rotation=0)
st.pyplot(fig)

