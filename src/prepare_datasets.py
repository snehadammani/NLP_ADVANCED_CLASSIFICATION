# src/prepare_datasets.py

import pandas as pd
from torch.utils.data import DataLoader
from src.dataset_loader import SentenceTransformationDataset

# Load cleaned and split CSVs from previous step
csv_path = r"C:\Users\RAHUL\Documents\Project\NLP_ADVANCE\data\raw\generated_dataset.csv"
df = pd.read_csv(csv_path)

# Lowercase & strip
df['Original'] = df['Original'].astype(str).str.lower().str.strip()
df['Transformed'] = df['Transformed'].astype(str).str.lower().str.strip()

# Split dataset
from sklearn.model_selection import train_test_split

train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
val_relative_size = 0.2 / 0.8
train_df, val_df = train_test_split(train_val_df, test_size=val_relative_size, stratify=train_val_df['Label'], random_state=42)

# Create Dataset objects
tokenizer_name = 'bert-base-uncased'
train_dataset = SentenceTransformationDataset(train_df['Transformed'].tolist(), train_df['Label'].tolist(), tokenizer_name)
val_dataset = SentenceTransformationDataset(val_df['Transformed'].tolist(), val_df['Label'].tolist(), tokenizer_name)
test_dataset = SentenceTransformationDataset(test_df['Transformed'].tolist(), test_df['Label'].tolist(), tokenizer_name)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
