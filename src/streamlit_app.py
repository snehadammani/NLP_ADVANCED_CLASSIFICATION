# src/streamlit_app.py

import streamlit as st
import torch
import shap
import numpy as np
import pandas as pd
import altair as alt
from transformers import BertTokenizer, BertForSequenceClassification

# ------------------------------
# Config / Constants
# ------------------------------
MODEL_PATH = r"C:\Users\RAHUL\Documents\Project\NLP_ADVANCE\model"

labels = [
    "Direct to Indirect Speech",
    "Active to Passive",
    "Negative to Positive",
    "Passive to Active",
    "Positive to Negative",
    "Indirect to Direct Speech",
]

# ------------------------------
# Load resources once and cache
# ------------------------------
@st.cache_resource
def load_resources():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    # Prediction function for SHAP
    def model_predict_safe(inputs):
        if isinstance(inputs, np.ndarray):
            inputs = [" ".join(["word"] * len(inputs[0])) for _ in range(len(inputs))]
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = [str(x) for x in inputs]

        tokenized = tokenizer(inputs, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**tokenized)
            probs = torch.softmax(outputs.logits, dim=1).numpy()
        return probs

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(model_predict_safe, masker)

    return tokenizer, model, explainer, model_predict_safe

# ------------------------------
# Streamlit UI Styling
# ------------------------------
def _style_header():
    st.set_page_config(page_title="Sentence Transformation Classifier", layout="wide", page_icon="🧠")
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;background-color:#1f2937;padding:12px;border-radius:10px'>
      <h1 style='margin:0;color:#f9fafb'>🧠 Sentence Transformer</h1>
      <div style='color:#9ca3af;margin-left:8px'>— Classify sentence transformations with SHAP explanations</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# Main app
# ------------------------------
def main():
    _style_header()

    tokenizer, model, explainer, model_predict_safe = load_resources()

    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.write(
            "Upload a sentence or pick a sample. The model predicts the transformation type and shows word-level SHAP contributions."
        )
        st.markdown("---")
        st.subheader("Samples")
        samples = [
            "She wrote the letter.",
            "The ball was thrown by John.",
            "I do not like green eggs.",
            "He said that he would come.",
        ]
        sample_choice = st.selectbox("Pick a sample", options=["(none)"] + samples)

    # Input form
    with st.form("input_form"):
        sentence = st.text_area("Enter a sentence to classify:", height=80)
        submitted = st.form_submit_button("Analyze")

    if (not sentence or sentence.strip() == "") and sample_choice != "(none)":
        sentence = sample_choice

    if sentence and (submitted or sample_choice != "(none)"):
        try:
            probs = model_predict_safe([sentence])[0]
            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        col_left, col_right = st.columns([2, 3])

        with col_left:
            st.subheader("Prediction")
            st.markdown(f"*Transformation Type:* **{pred_label}**")
            st.metric("Confidence", f"{probs[pred_idx]:.2f}")
            st.markdown("---")
            st.subheader("Raw probabilities")
            prob_df = pd.DataFrame({"label": labels, "probability": probs})
            st.table(prob_df.sort_values("probability", ascending=False).style.format({"probability": "{:.3f}"}))

        # SHAP visualization
        with col_right:
            st.subheader("Word-level SHAP contributions")
            try:
                shap_values = explainer([sentence])[0]
                words = list(shap_values.data)
                values = shap_values.values[:, pred_idx]

                shap_df = pd.DataFrame({"Word": words, "SHAP Value": values})
                shap_df['abs'] = shap_df['SHAP Value'].abs()

                # Altair horizontal bar chart with custom background
                chart = (
                    alt.Chart(shap_df.sort_values('abs', ascending=False).head(40))
                    .mark_bar()
                    .encode(
                        x=alt.X('SHAP Value:Q'),
                        y=alt.Y('Word:N', sort='-x'),
                        color=alt.condition(alt.datum['SHAP Value'] > 0, alt.value('#10b981'), alt.value('#ef4444')),
                        tooltip=['Word', alt.Tooltip('SHAP Value', format='.4f')]
                    )
                    .properties(height=300)
                    .configure_view(
                        strokeOpacity=0,
                        fill="#f3f4f6"  # Light grey background for chart
                    )
                )
                st.altair_chart(chart, use_container_width=True)

                # Styled dataframe
                display_df = shap_df[['Word', 'SHAP Value']].copy()
                display_df['SHAP Value'] = display_df['SHAP Value'].map(lambda x: float(x))
                st.dataframe(display_df.style.format({'SHAP Value': '{:.4f}'}).applymap(
                    lambda v: 'color: #10b981' if v > 0 else 'color: #ef4444', subset=['SHAP Value']
                ))
            except Exception as e:
                st.warning(f"SHAP explanation failed: {e}")


if __name__ == "__main__":
    main()
