# src/dataset_loader.py

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SentenceTransformationDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer_name='bert-base-uncased', max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Map labels to integers
        self.label2id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.label_ids = [self.label2id[label] for label in labels]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = self.label_ids[idx]

        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Squeeze to remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item
