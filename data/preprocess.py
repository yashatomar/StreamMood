import pandas as pd
import re
import os

# 1. Define Paths
RAW_DATA_PATH = "data/raw/train-00000-of-00001.parquet" # <--- CHANGE THIS IF YOUR FILENAME IS DIFFERENT
PROCESSED_DIR = "data/processed"

print(f"Loading raw data from {RAW_DATA_PATH}...")

# 2. Load Data
# We use read_parquet because the Hugging Face dataset is in Parquet format
df = pd.read_parquet(RAW_DATA_PATH)

print(f"Original shape: {df.shape}")

# 3. Clean the Text
def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove URLs (http/https)
    text = re.sub(r'http\S+', '', str(text))
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove extra whitespaces left behind
    text = ' '.join(text.split())
    return text

print("Cleaning text...")
df['message'] = df['message'].apply(clean_text)

# 4. Handle Timestamps
print("Processing timestamps...")
# The dataset uses 'published_at' instead of 'timestamp'
df['published_at'] = pd.to_datetime(df['published_at'], format='mixed', utc=True)
# Sort chronologically (oldest to newest)
df = df.sort_values(by='published_at')

# 5. Drop Missing Labels
# Twitch messages in this dataset do not have sentiment labels (they are NaN/Null).
# We MUST drop them so the AI only learns from labeled data.
print("Dropping unlabeled messages...")
df = df.dropna(subset=['sentiment'])
print(f"Shape after dropping missing labels: {df.shape}")

# 6. Split the Data
# We will use 70% for training the AI, and 30% for replaying as a live stream
print("Splitting data into training and replay streams...")
split_index = int(len(df) * 0.7)
train_df = df.iloc[:split_index]
replay_df = df.iloc[split_index:]

# 7. Save to CSV
# We save as CSV for easier handling in the next phases
os.makedirs(PROCESSED_DIR, exist_ok=True)

train_path = os.path.join(PROCESSED_DIR, "labeled_training_data.csv")
replay_path = os.path.join(PROCESSED_DIR, "replay_stream.csv")

train_df.to_csv(train_path, index=False)
replay_df.to_csv(replay_path, index=False)

print("\n--- Phase 2 Complete! ---")
print(f"Training data saved to: {train_path} ({len(train_df)} rows)")
print(f"Replay stream saved to: {replay_path} ({len(replay_df)} rows)")