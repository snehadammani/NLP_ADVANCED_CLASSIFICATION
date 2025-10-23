# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ Load dataset
def load_dataset(csv_path):
    """
    Load dataset from a CSV file.
    Expected columns: 'Original', 'Transformed', 'Label'
    """
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# 2️⃣ Basic cleaning (if needed)
def clean_text(df, text_columns=['Original', 'Transformed']):
    """
    Lowercase text and strip whitespaces
    """
    for col in text_columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
    return df

# 3️⃣ Train/Validation/Test split
def split_dataset(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Stratified split into train, validation, and test sets
    """
    # First split off test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['Label'], random_state=random_state
    )
    # Then split train/validation
    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative_size, stratify=train_val_df['Label'], random_state=random_state
    )
    print(f"Train: {train_df.shape[0]}, Validation: {val_df.shape[0]}, Test: {test_df.shape[0]}")
    return train_df, val_df, test_df

# ✅ Main execution
if __name__ == "__main__":
    csv_path = r"C:\Users\RAHUL\Documents\Project\NLP_ADVANCE\data\raw\generated_dataset.csv"
    df = load_dataset(csv_path)
    df = clean_text(df)
    train_df, val_df, test_df = split_dataset(df)
