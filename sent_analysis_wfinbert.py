import requests
import pandas as pd

# Replace the demo next to apikey with your own from the Alpha Vantage website

# Fetch news sentiment data from Alpha Vantage API
url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=BABA&time_from=20240801T0000&limit=200&apikey=293CSLERR4IAEZYY'
r = requests.get(url)
data = r.json()

# Convert the JSON data to a DataFrame
df = pd.DataFrame(data)

# Extract the 'feed' column which contains news articles
series = df['feed']

# Print the number of articles and the first and last few articles
print(len(series))
print(series.head())
print("\n\n=======================================\n")
print(series.tail())
print("=======================================\n")

# Extract headlines from the news articles
headlines = []
for i in series:
    headlines.append(i['title'])

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import eodhd
from eodhd import APIClient
import pandas as pd
import requests
import numpy as np
import warnings

# Load the FinBERT model and tokenizer for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Create a sentiment analysis pipeline
sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device='cpu')

# Analyze the sentiment of each headline
results = sentiment_pipeline(headlines, top_k=None)
    
# Store sentiment scores for each headline
sent_scores = []

for result in results:
    score_row = []  
    for score in result:  
        score_row.append(score['score'])  
    sent_scores.append(score_row) 

# Define a softmax function to convert scores to probabilities
sample = sent_scores[0]

def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum(axis=0)

# Convert sentiment scores to probabilities
scores = np.array(sample)

prob = []
for score in sent_scores:
    probabilities = softmax(score)
    prob.append(probabilities)
    
# Organize sentiment scores into a structured format
m_arr = []
for i in results:
    s_arr = []
    re_dict = {'positive': 0, 'neutral': 0, 'negative': 0}
    for dict_p in i:
        if dict_p['label'] == 'positive':
            re_dict['positive'] = dict_p['score']

    for dict_nu in i:
        if dict_nu['label'] == 'neutral':
            re_dict['neutral'] = dict_nu['score']

    for dict_n in i:
        if dict_n['label'] == 'negative':
            re_dict['negative'] = dict_n['score']
        
    s_arr.append(re_dict)
    m_arr.append(s_arr)

# Flatten the structured sentiment scores for further analysis
final = []
for result in m_arr:
    for score in result:  
        score_row = list(score.values())
        final.append(score_row) 

# Calculate and print the average probability of positive sentiment
first_values = [sublist[0] for sublist in final]
average_first_value_p = sum(first_values) / len(first_values)
    
print(f"\nThe probability of the market going up is: {average_first_value_p}")

# Calculate and print the average probability of neutral sentiment
second_values = [sublist[1] for sublist in final]
average_first_value_nu = sum(second_values) / len(second_values)
    
print(f"\nThe probability of the market being neutral is: {average_first_value_nu}")
    
# Calculate and print the average probability of negative sentiment
third_values = [sublist[2] for sublist in final]
average_first_value_n = sum(third_values) / len(third_values)
    
print(f"\nThe probability of the market going down is: {average_first_value_n}")
