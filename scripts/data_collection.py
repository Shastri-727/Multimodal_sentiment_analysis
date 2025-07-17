import tweepy
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Authenticate with the Twitter API v2
client = tweepy.Client(BEARER_TOKEN)

# Define the search query
# -is:retweet excludes retweets, lang:en gets English tweets
query = "OpenAI -is:retweet lang:en"

# Use the recent search endpoint
response = client.search_recent_tweets(query=query, max_results=100, tweet_fields=["created_at", "text", "lang"])

tweets = response.data

# Create a DataFrame
if tweets:
    df = pd.DataFrame([tweet.data for tweet in tweets])
    # Save to a CSV file in the raw data directory
    df.to_csv("data/raw/recent_tweets.csv", index=False)
    print("Successfully collected and saved tweets.")
else:
    print("No tweets found for the query.")