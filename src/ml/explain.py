"""
ESCALATION EXPLAINER  (Phase 4, Step 4)
========================================
Wraps the trained escalation model (xgb_escalation.json) + a SHAP
TreeExplainer into the same "load once, fail-open" pattern as
registry.py — nothing else in the codebase should import xgboost/shap
directly for this, they just ask this module for an explanation.

Not called from anywhere live yet (see docs/PHASE_4_PREDICTION_EXPLAINABILITY.md
§10) — Alert rows don't get created until Phase 5. This module exists so
that wiring, when it happens, is a single import + one function call.
"""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

ML_DIR = Path(__file__).resolve().parent.parent.parent / "ml_models"

# Same phrasing used in notebooks/06_escalation_predictor.ipynb Cell 9.
# ensemble_score is deliberately excluded — it's a redundant rescaling of
# votes (ensemble_score = votes / total_available), always ~0 SHAP
# importance, and including it in the sentence would just be noise.
FEATURE_DESCRIPTIONS = {
    "votes": {
        "up":   "an unusually high number of detectors agreed something was wrong",
        "down": "only a few detectors flagged anything unusual",
    },
    "zscore_value": {
        "up":   "the readings were far outside their normal statistical range",
        "down": "the readings stayed close to their normal statistical range",
    },
    "iforest_score": {
        "up":   "the overall combination of metrics looked highly unusual to the pattern-detection model",
        "down": "the overall combination of metrics looked fairly typical",
    },
    "lstm_error": {
        "up":   "the recent sequence of readings didn't match any normal pattern the system has learned",
        "down": "the recent sequence of readings still resembled normal patterns",
    },
}


class EscalationExplainer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        self.model = None
        self.explainer = None
        self.feature_cols = []
        self._load_model()

    def _load_model(self):
        model_path = ML_DIR / "xgb_escalation.json"
        config_path = ML_DIR / "xgb_escalation_config.json"
        if not model_path.exists() or not config_path.exists():
            print("⚠️  Escalation model files not found — escalation prediction DISABLED")
            return

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        with open(config_path) as f:
            config = json.load(f)
        self.feature_cols = config["feature_cols"]

        import shap
        self.explainer = shap.TreeExplainer(self.model)
        print(f"✅ Escalation predictor loaded ({len(self.feature_cols)} features)")

    @property
    def available(self):
        return self.model is not None

    def explain(self, feature_values: dict, top_k: int = 3) -> dict:
        """
        feature_values: dict mapping each of self.feature_cols to a number
                         for ONE reading, e.g.
                         {"votes": 2, "ensemble_score": 0.67,
                          "zscore_value": 3.16, "iforest_score": -0.01,
                          "lstm_error": 0.12}

        Returns a dict shaped exactly like Alert.explanation_text /
        Alert.contributing_features expect:
            {"probability": float, "explanation_text": str,
             "contributing_features": [{"feature", "value", "contribution"}, ...]}

        Fail-open: returns {"available": False} if the model isn't loaded,
        the same pattern registry.py's score_* methods use.
        """
        if not self.available:
            return {"available": False}

        x = np.array([[feature_values[c] for c in self.feature_cols]], dtype=np.float64)

        # A model that's been saved to disk and reloaded (like this one) loses
        # a few internal sklearn details that were present right after .fit()
        # in the notebook. Without them, SHAP can hand back expected_value /
        # shap_values wrapped in an extra array layer instead of plain numbers
        # or a plain (n_features,) row — even for a single-row, binary-only
        # prediction. Unwrap everything down to plain Python floats/1-D arrays
        # before doing any arithmetic or string formatting with them.
        shap_out = self.explainer.shap_values(x)
        if isinstance(shap_out, list):          # some shap versions: [class0_array, class1_array]
            shap_out = shap_out[-1]
        shap_row = np.asarray(shap_out).reshape(-1)   # flatten down to (n_features,)

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = np.asarray(base_value).reshape(-1)[-1]
        base_value = float(base_value)

        raw_margin = base_value + float(shap_row.sum())
        probability = float(1 / (1 + np.exp(-raw_margin)))

        contributions = [
            (name, feature_values[name], float(contribution))
            for name, contribution in zip(self.feature_cols, shap_row)
            if name != "ensemble_score"
        ]
        contributions.sort(key=lambda c: abs(c[2]), reverse=True)
        top = contributions[:top_k]

        reasons = [
            FEATURE_DESCRIPTIONS[name]["up" if contribution > 0 else "down"]
            for name, value, contribution in top
            if name in FEATURE_DESCRIPTIONS
        ]

        verdict = "likely to become severe" if probability >= 0.5 else "unlikely to become severe"
        sentence = f"This incident looks {verdict} ({probability:.0%} confidence)"
        if reasons:
            sentence += f", mainly because {reasons[0]}"
            if len(reasons) > 1:
                sentence += f", and {reasons[1]}"
            if len(reasons) > 2:
                sentence += f". A smaller factor: {reasons[2]}"
        sentence += "."

        return {
            "available": True,
            "probability": round(float(probability), 4),
            "explanation_text": sentence,
            "contributing_features": [
                {"feature": name, "value": round(float(value), 4), "contribution": round(contribution, 4)}
                for name, value, contribution in top
            ],
        }


# module-level singleton — same lazy-load-once pattern as registry.py
explainer = EscalationExplainer()


if __name__ == "__main__":
    # quick manual test: python -m src.ml.explain
    if explainer.available:
        result = explainer.explain({
            "votes": 2, "ensemble_score": 0.667,
            "zscore_value": 3.157, "iforest_score": -0.013,
            "lstm_error": 0.124,
        })
        print(result["explanation_text"])
        print(result["contributing_features"])
    else:
        print("Escalation model not available — train it first with "
              "notebooks/06_escalation_predictor.ipynb")