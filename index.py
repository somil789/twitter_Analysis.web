import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import cleantext


# Download NLTK resources (if not already downloaded)
nltk.download('vader_lexicon')
nltk.download('punkt')

# Function to get tweet length
def get_tweet_length(tweet):
    return len(tweet)

# Function to perform sentiment analysis
def analyze_sentiment(tweet):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(tweet)['compound']
    if sentiment_score > 0:
        return 'Positive'
    elif sentiment_score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to read CSV file and analyze tweets
def analyze_tweets(df):
    df['Tweet Length'] = df['tweet'].apply(get_tweet_length)
    df['Sentiment'] = df['tweet'].apply(analyze_sentiment)
    sentiment_counts = df['Sentiment'].value_counts()
    return df, sentiment_counts




# Page Design

# st.title("Twitter Analysis")
st.markdown('<div style="display: flex; align-items: center;"><img src="https://abs.twimg.com/responsive-web/client-web/icon-ios.b1fc7275.png" alt="Twitter Logo" width=100><h1 style="margin-left: 20px;">Twitter Analysis</h1></div>', unsafe_allow_html=True)
# st.markdown
st.write("This web app analyzes the sentiment of tweets from the provided CSV file.")

st.write("""
#### Group Info:-
         """)

st.write("""
         1) Somil Sharma
         2) Vishwa Deepak Giri
         """)


# Introduction
st.write("""
### Introduction
Twitter analysis refers to the process of examining data from the social media platform Twitter to gain insights, trends, sentiment, or other valuable information. Twitter is a rich source of real-time data with millions of users generating tweets on a wide range of topics every day. Analyzing this data can provide valuable insights for businesses, researchers, journalists, and individuals. 
""")
st.write("""
         
         Twitter analysis analysing:



Upload datasets.
Analyze tweet dimensions.
Count sentiments, labels.
Calculate tweet lengths.
Compare sentiments.
Distribute tweet data.
Analyze statistics.
Identify common words.
Analyze tweet relationships.
Examine tweet lengths.
List common words.
Analyze hashtags.
 """)






# About
st.write("""
### About
This web application is developed using Streamlit, NLTK, Pandas, Matplotlib, and WordCloud. It performs sentiment analysis on tweets from a CSV file and displays various statistics and visualizations.
""")



st.header('Sentiment Analyis')
with st.expander('Analyse Text/ Tweets'):
    text = st.text_input('text here:')
    if text:
        blob= TextBlob(text)
st.write('Polarity:', round(blob.sentiment.polarity,2))
st.write('Subjectivity:', round(blob.sentiment.subjectivity,2))




# File Upload
uploaded_file = st.file_uploader("Upload Twitter CSV File", type=['csv'])

# Perform analysis if file is uploaded
if uploaded_file is not None:
    st.write("### Analysis Results")
    df = pd.read_csv(uploaded_file)
    df, sentiment_counts = analyze_tweets(df)

    # Display DataFrame information
    st.write("#### Dataset Information:")
    st.write(f"Number of Rows: {df.shape[0]}")
    st.write(f"Number of Columns: {df.shape[1]}")

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

    
# Distribution of Tweet Lengths
    st.write("#### Distribution of Tweet Lengths:")
    st.bar_chart(df['Tweet Length'].value_counts())

    # Most Common Words
    st.write("#### Most Common Words:")
    words = ' '.join(df['tweet']).lower().split()
    word_counts = Counter(words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    st.image(wordcloud.to_array())

    # Comparison between positive and negative tweets
    st.write("#### Comparison between Positive and Negative Tweets:")
    positive_count = sentiment_counts.get('Positive', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    total_count = df.shape[0]
    st.write(f"Percentage of Positive Tweets: {positive_count/total_count * 100:.2f}%")
    st.write(f"Percentage of Negative Tweets: {negative_count/total_count * 100:.2f}%")

    # Mean and standard deviation of tweet lengths for positive tweets
    positive_tweets = df[df['Sentiment'] == 'Positive']
    mean_length_positive = positive_tweets['Tweet Length'].mean()
    std_dev_positive = positive_tweets['Tweet Length'].std()
    st.write("#### Mean and Standard Deviation of Tweet Lengths for Positive Tweets:")
    st.write(f"Mean Length: {mean_length_positive:.2f}")
    st.write(f"Standard Deviation: {std_dev_positive:.2f}")

    import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

# Function to extract hashtags from tweets
def extract_hashtags(tweet):
    hashtags = re.findall(r'#(\w+)', tweet)
    return hashtags

# Function to analyze tweets and find top 10 hashtags
def analyze_tweets(df):
    hashtags_list = df['tweet'].apply(extract_hashtags)
    hashtags_flat = [tag for sublist in hashtags_list for tag in sublist]
    hashtags_count = Counter(hashtags_flat)
    top_10_hashtags = hashtags_count.most_common(10)
    return top_10_hashtags

# Page Design
st.title("Top 10 Trending Hashtags")

# File Upload
uploaded_file = st.file_uploader("Upload Twitter CSV File", type=['csv'], key="uploader")


# Perform analysis if file is uploaded
if uploaded_file is not None:
    st.write("### Analysis Results")
    df = pd.read_csv(uploaded_file)
    top_10_hashtags = analyze_tweets(df)

    # Display top 10 hashtags
    st.write("#### Top 10 Trending Hashtags:")
    for i, (hashtag, count) in enumerate(top_10_hashtags, start=1):
        st.write(f"{i}. #{hashtag}: {count} mentions")

    # Plotting
    st.write("#### Top 10 Trending Hashtags Graph:")
    fig, ax = plt.subplots(figsize=(10, 6))
    hashtags_df = pd.DataFrame(top_10_hashtags, columns=['Hashtag', 'Count'])
    hashtags_df.plot(kind='bar', x='Hashtag', y='Count', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title('Top 10 Trending Hashtags')
    ax.set_xlabel('Hashtag')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    st.pyplot(fig)


import re

# Function to extract hashtags from tweets
def extract_hashtags(tweet):
    hashtags = re.findall(r'#(\w+)', tweet)
    return hashtags

# Extract hashtags from all tweets
df['Hashtags'] = df['tweet'].apply(extract_hashtags)

# Flatten the list of hashtags
all_hashtags = [tag for sublist in df['Hashtags'] for tag in sublist]

# Function to filter hashtags by sentiment
def filter_hashtags_by_sentiment(df, sentiment):
    filtered_hashtags = [tag for tags in df[df['Sentiment'] == sentiment]['Hashtags'] for tag in tags]
    return filtered_hashtags

# Get top hashtags for positive, negative, and neutral sentiments
top_positive_hashtags = Counter(filter_hashtags_by_sentiment(df, 'Positive')).most_common(10)
top_negative_hashtags = Counter(filter_hashtags_by_sentiment(df, 'Negative')).most_common(10)
top_neutral_hashtags = Counter(filter_hashtags_by_sentiment(df, 'Neutral')).most_common(10)

# Plotting top hashtags for each sentiment
def plot_top_hashtags(top_hashtags, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    df_top_hashtags = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
    df_top_hashtags.plot(kind='bar', x='Hashtag', y='Count', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Hashtag')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    st.pyplot(fig)

# Plot top hashtags for positive sentiment
st.write("#### Top Hashtags for Positive Sentiment:")
plot_top_hashtags(top_positive_hashtags, "Top Hashtags for Positive Sentiment")

# Plot top hashtags for negative sentiment
st.write("#### Top Hashtags for Negative Sentiment:")
plot_top_hashtags(top_negative_hashtags, "Top Hashtags for Negative Sentiment")

# Plot top hashtags for neutral sentiment
st.write("#### Top Hashtags for Neutral Sentiment:")
plot_top_hashtags(top_neutral_hashtags, "Top Hashtags for Neutral Sentiment")




