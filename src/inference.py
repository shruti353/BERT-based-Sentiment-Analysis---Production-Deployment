import torch
from transformers import BertForSequenceClassification
from tokenizer import tokenizer
from config import MODEL_NAME

model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()
