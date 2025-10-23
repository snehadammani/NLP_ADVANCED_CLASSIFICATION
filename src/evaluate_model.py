# src/evaluate_model.py

import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer
from src.dataset_loader import SentenceTransformationDataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Step 1: Load Test Dataset
# -----------------------------
csv_path = r"C:\Users\RAHUL\Documents\Project\NLP_ADVANCE\data\raw\generated_dataset.csv"
df = pd.read_csv(csv_path)

df['Transformed'] = df['Transformed'].astype(str).str.lower().str.strip()
df['Label'] = df['Label'].astype(str).str.strip()

# Split same as training
from sklearn.model_selection import train_test_split
train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

# Dataset and DataLoader
tokenizer_name = 'bert-base-uncased'
test_dataset = SentenceTransformationDataset(test_df['Transformed'].tolist(), test_df['Label'].tolist(), tokenizer_name)
test_loader = DataLoader(test_dataset, batch_size=16)

# -----------------------------
# Step 2: Load Trained Model
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r"C:\Users\RAHUL\Documents\Project\NLP_ADVANCE\model"
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# -----------------------------
# Step 3: Predict and Evaluate
# -----------------------------
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy calculation
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")   # <- Clear accuracy output

# Classification Report
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=test_dataset.label2id.keys()))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))

