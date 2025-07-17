import pandas as pd
import re

def clean_text(text):
    """A simple function to clean tweet text."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@[a-zA-Z0-9_]+", "", text)  # Remove mentions
    text = re.sub(r"#", "", text)  # Remove hashtags
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text

# Load the raw data
df = pd.read_csv("data/raw/recent_tweets.csv")

# Apply the cleaning function
df['cleaned_text'] = df['text'].apply(clean_text)

# Simple data validation
assert 'cleaned_text' in df.columns
assert df['cleaned_text'].isnull().sum() == 0

# Save the processed data
df.to_csv("data/processed/cleaned_tweets.csv", index=False)

print("Data preprocessing and validation complete.")