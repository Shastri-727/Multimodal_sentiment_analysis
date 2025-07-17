from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

# Initialize the FastAPI app
app = FastAPI()

# Load the sentiment analysis pipeline
# Ensure the model is saved in the './models/fine-tuned-bert' directory
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="./models/fine-tuned-bert")
except Exception as e:
    # If the model isn't found, create a placeholder pipeline
    print(f"Could not load local model: {e}. Using a default model.")
    sentiment_pipeline = pipeline("sentiment-analysis")


class TextToAnalyze(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running."}


@app.post("/analyze")
def analyze_sentiment(data: TextToAnalyze):
    """
    Analyzes the sentiment of a given text.
    """
    result = sentiment_pipeline(data.text)
    return result[0]

# To run this app, use the command:
# uvicorn app:app --reload