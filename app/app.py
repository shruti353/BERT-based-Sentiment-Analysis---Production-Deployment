from fastapi import FastAPI
from schema import SentimentRequest
from src.inference import predict

app = FastAPI(title="BERT Sentiment API")

@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    label = predict(request.text)
    sentiment = "positive" if label == 1 else "negative"
    return {"sentiment": sentiment}
