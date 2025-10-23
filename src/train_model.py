# src/train_model.py

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# -----------------------------
# Step 1: Load and prepare dataset
# -----------------------------
csv_path = r"C:\Users\RAHUL\Documents\Project\NLP_ADVANCE\data\raw\generated_dataset.csv"
df = pd.read_csv(csv_path)

# Lowercase & strip
df['Original'] = df['Original'].astype(str).str.lower().str.strip()
df['Transformed'] = df['Transformed'].astype(str).str.lower().str.strip()
df['Label'] = df['Label'].astype(str).str.strip()

# Split dataset
train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
val_relative_size = 0.2 / 0.8
train_df, val_df = train_test_split(train_val_df, test_size=val_relative_size, stratify=train_val_df['Label'], random_state=42)

# -----------------------------
# Step 2a: Create Tokenizer
# -----------------------------
tokenizer_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

# -----------------------------
# Step 2b: Dataset Class
# -----------------------------
class SentenceTransformationDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=64):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = {label: i for i, label in enumerate(sorted(set(labels)))}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = self.label2id[self.labels[idx]]

        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
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

# -----------------------------
# Step 2c: Create Datasets & DataLoaders
# -----------------------------
train_dataset = SentenceTransformationDataset(train_df['Transformed'].tolist(), train_df['Label'].tolist(), tokenizer)
val_dataset = SentenceTransformationDataset(val_df['Transformed'].tolist(), val_df['Label'].tolist(), tokenizer)
test_dataset = SentenceTransformationDataset(test_df['Transformed'].tolist(), test_df['Label'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# -----------------------------
# Step 3: Load BERT Model
# -----------------------------
num_labels = len(set(df['Label']))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model.to(device)

# -----------------------------
# Step 4: Optimizer
# -----------------------------
optimizer = AdamW(model.parameters(), lr=2e-5)

# -----------------------------
# Step 5: Training Loop
# -----------------------------
epochs = 3

for epoch in range(epochs):
    # Training
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    val_acc = total_correct / total_samples
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f}")

# -----------------------------
# Step 6: Save Model & Tokenizer
# -----------------------------
output_dir = r"C:\Users\RAHUL\Documents\Project\NLP_ADVANCE\model"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved at {output_dir}")
print("Training complete ✅")
