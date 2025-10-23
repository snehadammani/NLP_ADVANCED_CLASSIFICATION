import torch
from transformers import BertForSequenceClassification, BertTokenizer
import shap

# Load trained model & tokenizer
model_path = r"C:\Users\RAHUL\Documents\Project\NLP_ADVANCE\model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Example sentence
sentence = "He eats an apple"

# Tokenize
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=64)

# Predict function for SHAP
def f(x):
    tokens = tokenizer.batch_encode_plus(x.tolist(), padding='max_length', truncation=True, max_length=64, return_tensors='pt')
    with torch.no_grad():
        logits = model(tokens['input_ids'], attention_mask=tokens['attention_mask']).logits
    return logits.numpy()

# SHAP explainer
explainer = shap.Explainer(f, tokenizer)
shap_values = explainer([sentence])
shap.plots.text(shap_values[0])
