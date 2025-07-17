import pytest
from transformers import pipeline

@pytest.fixture
def sentiment_pipeline():
    """Load the fine-tuned model for testing."""
    return pipeline("sentiment-analysis", model="./models/fine-tuned-bert")

def test_positive_sentiment(sentiment_pipeline):
    """Test for correctly identifying positive sentiment."""
    result = sentiment_pipeline("I love the new update, it's fantastic!")
    assert result[0]['label'].startswith('LABEL_1') # Assuming LABEL_1 is positive

def test_negative_sentiment(sentiment_pipeline):
    """Test for correctly identifying negative sentiment."""
    result = sentiment_pipeline("This is the worst service I have ever received.")
    assert result[0]['label'].startswith('LABEL_0') # Assuming LABEL_0 is negative