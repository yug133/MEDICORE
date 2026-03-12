"""
ICD-10 Coding Module
Owner: Shravani Shinde

This module takes clinical notes as input and predicts the
top-k most likely ICD-10 codes using a fine-tuned BioBERT model.

Steps to complete:
    1. ✅ Skeleton ready
    2. ⬜ Download and preprocess dataset (see data_prep.py)
    3. ⬜ Train BioBERT model (see train.py)
    4. ⬜ Save trained model to /models/icd_model/
    5. ⬜ Replace placeholder predict() with real inference
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────
MODEL_NAME   = "emilyalsentzer/Bio_ClinicalBERT"   # Pretrained base
MODEL_PATH   = os.getenv("ICD_MODEL_PATH", "./models/icd_model")
MAX_LENGTH   = 512
TOP_K        = 5      # Return top 5 ICD codes

# Sample ICD-10 codes (replace with full list from your dataset)
SAMPLE_ICD_CODES = [
    "J18.9",  # Pneumonia
    "I10",    # Hypertension
    "E11.9",  # Type 2 Diabetes
    "F32.1",  # Major depressive disorder
    "K21.0",  # GERD
    "J45.9",  # Asthma
    "N18.3",  # Chronic kidney disease
    "M54.5",  # Low back pain
]


class ICDCodingModule:

    def __init__(self):
        self.tokenizer = None
        self.model     = None
        self.icd_codes = SAMPLE_ICD_CODES
        self.is_loaded = False

    def load_model(self):
        """Load the fine-tuned model. Falls back to base model if not trained yet."""
        print("[ICD Module] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if os.path.exists(MODEL_PATH):
            print(f"[ICD Module] Loading fine-tuned model from {MODEL_PATH}")
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        else:
            print("[ICD Module] ⚠️  Fine-tuned model not found. Using base model (untrained).")
            print("[ICD Module] Run train.py to train the model first.")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=len(self.icd_codes),
                ignore_mismatched_sizes=True
            )

        self.model.eval()
        self.is_loaded = True
        print("[ICD Module] ✅ Model loaded successfully.")

    def predict(self, clinical_note: str) -> dict:
        """
        Predict top-k ICD-10 codes from a clinical note.

        Args:
            clinical_note: Raw clinical text from a doctor

        Returns:
            dict with top_codes list and confidence_scores
        """
        if not self.is_loaded:
            self.load_model()

        # Tokenize
        inputs = self.tokenizer(
            clinical_note,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits  = outputs.logits
            probs   = torch.sigmoid(logits).squeeze().numpy()

        # Get top-k codes
        top_indices = np.argsort(probs)[::-1][:TOP_K]
        results = [
            {
                "icd_code":   self.icd_codes[i],
                "confidence": round(float(probs[i]), 4)
            }
            for i in top_indices
        ]

        return {
            "top_codes": results,
            "model_version": "base_untrained" if not os.path.exists(MODEL_PATH) else "fine_tuned",
            "note": "Train the model on MIMIC-III data for accurate predictions."
        }


# ── Singleton instance used by the API ───────────────
icd_module = ICDCodingModule()
