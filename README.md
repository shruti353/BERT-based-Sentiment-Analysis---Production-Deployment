# BERT-Based Sentiment Analysis with MLOps Deployment

## 🚀 Overview
This project implements an end-to-end BERT-based sentiment analysis system with full MLOps lifecycle:
- Fine-tuned BERT-base on 25,000+ reviews
- MLflow experiment tracking & model versioning
- FastAPI inference service
- Dockerized production deployment

## 🧠 Tech Stack
- Transformers (HuggingFace)
- PyTorch
- FastAPI
- MLflow
- Docker

## 📊 Model Performance
- Accuracy: ~94%
- F1 Score: ~0.92
- Latency: ~120ms
- Throughput: 100+ req/min

## ▶️ Run Training
```bash
python src/train.py

Start API
uvicorn app.app:app --reload

🐳 Docker
docker build -t bert-sentiment .
docker run -p 8000:8000 bert-sentiment