import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os

# 1. Load Data
print("Loading cleaned training data...")
df = pd.read_csv("data/processed/labeled_training_data.csv")
df = df.dropna(subset=['message', 'sentiment'])

# 2. Encode Labels (Convert Text labels to Numbers: 0, 1, 2)
print("Encoding labels...")
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])
num_labels = len(le.classes_)
print(f"Labels mapped: {list(le.classes_)}")

# 3. Train/Test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['message'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 4. Load Tokenizer and Model
print("Loading DistilBERT model (this may take a moment)...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

# 5. Create Custom Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 6. Setup Device (Use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training will run on: {device}")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# 7. Training Loop
print("Starting fine-tuning...")
epochs = 2 # Kept low so it doesn't take hours on a CPU
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

# 8. Evaluation
print("Evaluating model...")
model.eval()
val_preds = []
val_true = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_true.extend(labels.cpu().numpy())

print(f"DistilBERT Weighted F1-Score: {f1_score(val_true, val_preds, average='weighted'):.4f}")

# 9. Save Model and Tokenizer
save_path = "models/distilbert"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Save the Label Encoder (We need this for the API later)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"\nDistilBERT model successfully saved to {save_path}!")