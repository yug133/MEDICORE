"""
Mental Health Risk Triage Module
Owner: Krishna Kumar

This module analyses clinical notes for hidden psychological risk indicators
and classifies patients into Low / Moderate / High risk levels.

Steps to complete:
    1. ✅ Skeleton ready (rule-based baseline works immediately)
    2. ⬜ Download CLPsych or Dreaddit dataset
    3. ⬜ Fine-tune MentalBERT (see train.py)
    4. ⬜ Save model to /models/mental_model/
    5. ⬜ Replace rule-based fallback with model predictions
"""

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────
MODEL_NAME  = "mental/mental-bert-base-uncased"
MODEL_PATH  = os.getenv("MENTAL_MODEL_PATH", "./models/mental_model")
MAX_LENGTH  = 256

# ── Risk Keyword Dictionary (rule-based baseline) ─────
RISK_KEYWORDS = {
    "high": [
        "suicidal", "suicide", "kill myself", "end my life",
        "self-harm", "cutting", "overdose", "worthless", "hopeless",
        "no reason to live", "want to die", "can't go on"
    ],
    "moderate": [
        "depressed", "depression", "anxious", "anxiety", "panic attack",
        "can't sleep", "insomnia", "crying", "sad all the time",
        "losing interest", "no motivation", "feeling empty", "numb"
    ],
    "low": [
        "stressed", "worried", "overwhelmed", "tired", "fatigue",
        "trouble concentrating", "appetite changes", "mood swings"
    ]
}

RISK_ACTIONS = {
    "High":     "🚨 URGENT: Refer to psychiatrist immediately. Do not leave patient unattended.",
    "Moderate": "⚠️  Schedule mental health follow-up within 1 week. Consider counselling referral.",
    "Low":      "✅ Monitor at next regular appointment. Provide self-care resources."
}


class MentalHealthModule:

    def __init__(self):
        self.tokenizer = None
        self.model     = None
        self.is_loaded = False

    def load_model(self):
        """Load fine-tuned MentalBERT. Falls back to rule-based if model not trained."""
        if os.path.exists(MODEL_PATH):
            print("[Mental Module] Loading fine-tuned MentalBERT...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            self.model.eval()
            print("[Mental Module] ✅ Model loaded.")
        else:
            print("[Mental Module] ⚠️  Trained model not found. Using rule-based fallback.")
            print("[Mental Module] This still works for demo — train model for higher accuracy.")

        self.is_loaded = True

    def _rule_based_triage(self, text: str) -> tuple[str, list[str], float]:
        """
        Baseline rule-based classifier using keyword matching.
        Works immediately without any training. Use this for your demo
        while the ML model is being trained.
        """
        text_lower = text.lower()
        found      = {"high": [], "moderate": [], "low": []}

        for level, keywords in RISK_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    found[level].append(kw)

        # Determine risk level by priority
        if found["high"]:
            return "High", found["high"], 0.92
        elif found["moderate"]:
            return "Moderate", found["moderate"], 0.78
        elif found["low"]:
            return "Low", found["low"], 0.65
        else:
            return "Low", [], 0.55

    def _model_triage(self, text: str) -> tuple[str, float]:
        """ML model-based triage (used after training is complete)."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=1).squeeze()

        label_map   = {0: "Low", 1: "Moderate", 2: "High"}
        pred_index  = probs.argmax().item()
        confidence  = round(probs[pred_index].item(), 4)
        return label_map[pred_index], confidence

    def triage(self, clinical_note: str) -> dict:
        """
        Analyse clinical note and return risk level with recommended action.

        Args:
            clinical_note: Raw clinical text

        Returns:
            dict with risk_level, confidence, triggered_keywords, action
        """
        if not self.is_loaded:
            self.load_model()

        # Use ML model if trained, else fall back to rule-based
        if self.model:
            risk_level, confidence = self._model_triage(clinical_note)
            triggered_keywords     = []
            method                 = "ml_model"
        else:
            risk_level, triggered_keywords, confidence = self._rule_based_triage(clinical_note)
            method = "rule_based"

        return {
            "risk_level":          risk_level,
            "confidence":          confidence,
            "recommended_action":  RISK_ACTIONS[risk_level],
            "triggered_keywords":  triggered_keywords,
            "method_used":         method,
            "note": (
                "Rule-based mode active — train MentalBERT for production use."
                if method == "rule_based"
                else "ML model prediction."
            )
        }


# ── Singleton instance used by the API ───────────────
mental_module = MentalHealthModule()
