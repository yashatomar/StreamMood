from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle
import os

app = FastAPI(title="StreamMood API", description="Real-time Sentiment Analysis API")

model = None
tokenizer = None
le = None

@app.on_event("startup")
def load_model():
    global model, tokenizer, le
    print("Loading DistilBERT model into API...")
    
    # Path relative to the root folder
    model_path = "models/distilbert" 
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    with open("models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
        
    print("Model loaded successfully!")

class TextBatch(BaseModel):
    messages: List[str]

@app.post("/analyze")
def analyze_sentiment(payload: TextBatch):
    if model is None:
        return {"error": "Model not loaded yet"}
        
    results = []
    for text in payload.messages:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).item()
            confidence = torch.softmax(outputs.logits, dim=1).max().item()
        
        sentiment = le.inverse_transform([preds])[0]
        
        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4)
        })
        
    return {"results": results}