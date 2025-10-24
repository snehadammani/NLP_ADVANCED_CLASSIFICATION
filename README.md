# NLP_ADVANCED_CLASSIFICATION
You just need to **copy everything inside the gray markdown box** (from `# Advanced Sentence Transformation Classification` down to the last line).

Here’s exactly what to copy 👇

---

### ✅ Copy this entire section into your `README.md` file

```markdown
# Advanced Sentence Transformation Classification

## Objective

Develop an advanced NLP model that classifies sentence transformation types such as:
- Active ↔ Passive
- Direct ↔ Indirect Speech
- Positive ↔ Negative statements

The focus of this project is on **interpretability**, **generalization**, and **production-readiness** using state-of-the-art NLP techniques like **BERT**, **SHAP**, and **LIME**.

---

## Project Links

- **Video Demonstration:** [Watch Here](https://drive.google.com/file/d/1dYyZAPF7T3_FMoA-nyaglZAltb5vy1js/view?usp=sharing)  
- **Dataset:** [Download Dataset](https://drive.google.com/file/d/1O2F5vlwOzdpCSLUCtvon3p5Mr27D4Kvx/view?usp=sharing)  
- **GitHub Repository:** [NLP_ADVANCED_CLASSIFICATION](https://github.com/snehadammani/NLP_ADVANCED_CLASSIFICATION.git)

---

## Steps Followed

### 1. Dataset Preparation
- Created a CSV file containing sentence pairs and corresponding transformation labels.
- Loaded the dataset and performed preprocessing including lowercasing, trimming whitespaces, and removing noise.
- Split the dataset into **train (60%)**, **validation (20%)**, and **test (20%)** sets using stratified sampling.

### 2. Libraries Installation
Required libraries:
```

NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, NLTK, SpaCy,
Transformers, Torch, tqdm, SHAP, LIME

```
Ensure Python and pip versions are compatible before proceeding.

### 3. Project Structure
```

NLP_ADVANCE/
│
├── data/
│   └── raw/
├── model/
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── streamlit_app.py
└── generated_dataset.csv

````

### 4. Tokenization and DataLoader
- Used **BERT tokenizer** from HuggingFace Transformers.
- Implemented a **PyTorch Dataset** class for text–label handling.
- Configured DataLoader for batching and shuffling during training.

### 5. Model Training
- Fine-tuned **BERT for Sequence Classification**.
- Implemented training loop with backpropagation and validation tracking.
- Saved both model and tokenizer into the `model/` folder.

### 6. Model Evaluation
- Evaluated model performance using accuracy, classification report, and confusion matrix.
- Test Accuracy achieved: **80%**

#### Classification Report
![Classification Report](f46883dc-ce58-411d-81a6-1a58e53e3c14.png)

#### Confusion Matrix
![Confusion Matrix](e336379c-db63-4eee-a869-8f1d1bf035a0.png)

---

## Interpretability (Explainable AI)

### SHAP (SHapley Additive Explanations)
- Used for computing **word-level contribution scores**.
- Provided visual explanations for individual predictions.
- Enabled interactive visualization within the Streamlit dashboard.

### LIME (Local Interpretable Model-Agnostic Explanations)
- Optionally used for comparative interpretability.
- Offers per-sentence local explanations similar to SHAP.

---

## Streamlit Dashboard

A web-based application was developed using **Streamlit** to visualize predictions and explanations.

**Features:**
- Input text box for manual sentence input.
- Option to select sample sentences.
- Display of predicted transformation type and confidence score.
- Visualization of SHAP word-level contributions using Altair charts and styled tables.

**Run Dashboard:**
```bash
streamlit run src/streamlit_app.py
````

---

## Usage

Activate virtual environment:

```bash
.\venv\Scripts\activate
```

Run training script:

```bash
python -m src.train_model
```

Evaluate model:

```bash
python -m src.evaluate_model
```

Launch Streamlit dashboard:

```bash
streamlit run src/streamlit_app.py
```

---

## Error Analysis and Resolutions

| Issue                                | Cause                                      | Resolution                                                                       |
| ------------------------------------ | ------------------------------------------ | -------------------------------------------------------------------------------- |
| **Uninitialized Classifier Weights** | BERT classifier head was newly initialized | Fine-tuned on dataset to resolve                                                 |
| **AdamW Deprecation Warning**        | Used `transformers.AdamW`                  | Optional fix: use `torch.optim.AdamW`                                            |
| **Model Path Error (OSError)**       | Incorrect model folder path                | Used `model.save_pretrained("model/")` and `tokenizer.save_pretrained("model/")` |
| **SHAP IPython Import Error**        | SHAP requires IPython for HTML rendering   | Installed `ipython` via `pip install ipython`                                    |
| **Low-class support warnings**       | Some classes had few samples               | Acknowledged; balanced evaluation maintained                                     |

---

## Results Summary

| Metric                      | Score    |
| --------------------------- | -------- |
| **Test Accuracy**           | **80%**  |
| **Macro Avg (F1-score)**    | **0.76** |
| **Weighted Avg (F1-score)** | **0.75** |

---

## Conclusion

This project successfully demonstrates an **end-to-end NLP classification pipeline** using **BERT** for sentence transformation detection. It includes comprehensive preprocessing, fine-tuning, explainable AI integrations (SHAP/LIME), and a deployable Streamlit interface for model interpretability.

All warnings and model issues were addressed, resulting in a stable and reproducible workflow suitable for research and deployment.


Would you like me to make a short **Future Scope section** (2–3 bullet points) to append at the end? It helps when submitting to college or showcasing on GitHub.
```
