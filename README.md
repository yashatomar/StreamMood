# StreamMood

# 🎙️ StreamMood: Real-Time Chat Sentiment Analysis

StreamMood is an end-to-end Machine Learning and Full-Stack application designed to analyze the sentiment of live chat streams in real-time. This project uses a simulated live stream of the NASA Artemis II dataset to predict the mood of the audience (Positive, Negative, or Neutral).

## 🚀 Current Progress: 35% Complete (Milestone 1)

### ✅ What is Completed
*   **Phase 1: Environment Setup:** Modular folder structure and virtual environment.
*   **Phase 2: Data Pipeline:** Ingested the raw NASA Artemis II dataset (242K rows), cleaned URLs/mentions, handled timezone formats, and split data into training and replay streams using `pandas`.
*   **Phase 3: Baseline ML Model:** Trained a TF-IDF + Logistic Regression model to establish an accuracy benchmark.
*   **Phase 4: Transformer Model (DistilBERT):** Fine-tuned `distilbert-base-uncased` on the chat data using a Google Colab T4 GPU for high-performance sentiment classification.
*   **Phase 5 (Part 1): Backend API:** Built a FastAPI server that loads the trained DistilBERT model and provides a `/analyze` endpoint for real-time inference.

### 🛠️ Tech Stack
*   **Data Processing:** Python, Pandas, Scikit-Learn
*   **Machine Learning:** PyTorch, Hugging Face Transformers (DistilBERT)
*   **Backend:** FastAPI, Uvicorn
*   **Frontend (Upcoming):** Streamlit

### 📂 Project Structure
```text
StreamMood/
├── api/                  # (Removed - main.py is in root for easier execution)
├── app/                  # Future Streamlit frontend
├── data/
│   ├── raw/              # Raw NASA Artemis II dataset
│   └── processed/        # Cleaned training and replay CSVs
├── models/               # Saved ML models (Baseline & DistilBERT)
├── pipeline/             # Future Producer-Consumer streaming scripts
├── main.py               # FastAPI backend (The "Waiter")
├── preprocess.py         # Data cleaning script
├── train_baseline.py     # Baseline model training script
├── train_distilbert.py   # Transformer model training script
└── requirements.txt      # Python dependencies
