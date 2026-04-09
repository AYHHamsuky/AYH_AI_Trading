"""
XGBoost signal classifier — BUY (2) / HOLD (1) / SELL (0)
Includes walk-forward retraining, feature importance, and SHAP analysis.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from config import XGBOOST_CONFIG as XC, MODEL_DIR

logger = logging.getLogger(__name__)

FEATURE_COLS_EXCLUDE = {"open","high","low","close","volume","label","future_return"}


class XGBoostSignalModel:
    """
    Three-class classifier: SELL=0, HOLD=1, BUY=2.
    Uses monotone constraints to prevent overfitting on RSI/MACD.
    """

    def __init__(self, symbol: str = "EURUSD", timeframe: str = "H1"):
        self.symbol    = symbol
        self.timeframe = timeframe
        self.model     = None
        self.scaler    = StandardScaler()
        self.feature_cols: list = []
        self._bar_count = 0
        self._model_path = os.path.join(MODEL_DIR, f"xgb_{symbol}_{timeframe}.pkl")

    # ── Training ──────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        """Full train on labelled feature DataFrame. Returns eval metrics."""
        import xgboost as xgb

        X, y = self._prepare(df)
        if len(X) < 200:
            raise ValueError(f"Too few samples ({len(X)}) to train")

        # Walk-forward cross-validation
        tscv   = TimeSeriesSplit(n_splits=5)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            mdl = self._build_model()
            mdl.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            preds = mdl.predict(X_val)
            acc   = accuracy_score(y_val, preds)
            scores.append(acc)
            if verbose:
                logger.info("Fold %d accuracy: %.4f", fold+1, acc)

        # Final model on all data
        self.model = self._build_model()
        # Use 80% for train, 20% for early stopping
        split = int(len(X) * 0.8)
        self.model.fit(
            X.iloc[:split], y.iloc[:split],
            eval_set=[(X.iloc[split:], y.iloc[split:])],
            verbose=False,
        )

        metrics = {
            "cv_accuracy_mean": np.mean(scores),
            "cv_accuracy_std":  np.std(scores),
        }
        if verbose:
            preds = self.model.predict(X.iloc[split:])
            logger.info("\n%s", classification_report(y.iloc[split:], preds,
                        target_names=XC["classes"]))
        self.save()
        return metrics

    def retrain(self, df: pd.DataFrame) -> None:
        """Lightweight retrain on fresh data (warm start)."""
        if self.model is None:
            self.train(df)
            return
        X, y = self._prepare(df, fit_scaler=False)
        split = int(len(X) * 0.8)
        self.model.fit(
            X.iloc[:split], y.iloc[:split],
            eval_set=[(X.iloc[split:], y.iloc[split:])],
            verbose=False,
            xgb_model=self.model.get_booster(),
        )
        self.save()

    # ── Inference ─────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (class_labels, probabilities).
        class_labels: array of 0/1/2
        probabilities: array of shape (n, 3)
        """
        if self.model is None:
            if not self.load():
                raise RuntimeError("Model not trained — call train() first")
        X, _ = self._prepare(df, fit_scaler=False, add_label=False)
        probs  = self.model.predict_proba(X)
        labels = self.model.predict(X)
        self._bar_count += len(X)
        return labels, probs

    def predict_latest(self, df: pd.DataFrame) -> dict:
        """Predict on the most recent bar only. Triggers retraining if due."""
        labels, probs = self.predict(df.tail(1))
        signal  = int(labels[0])
        prob    = float(probs[0].max())
        return {
            "signal":      signal,
            "signal_name": XC["classes"][signal],
            "confidence":  prob,
            "prob_sell":   float(probs[0][0]),
            "prob_hold":   float(probs[0][1]),
            "prob_buy":    float(probs[0][2]),
        }

    # ── Feature importance ────────────────────────────────────────────────
    def feature_importance(self, top_n: int = 20) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained")
        imp = pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)
        return imp.head(top_n)

    def shap_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return SHAP values for the last `n` rows."""
        try:
            import shap
        except ImportError:
            logger.warning("shap not installed — pip install shap")
            return pd.DataFrame()
        X, _ = self._prepare(df, fit_scaler=False, add_label=False)
        explainer = shap.TreeExplainer(self.model)
        shap_vals  = explainer.shap_values(X)
        # For multi-class, shap_vals is a list; pick BUY class
        vals = shap_vals[2] if isinstance(shap_vals, list) else shap_vals
        return pd.DataFrame(vals, columns=self.feature_cols, index=df.index[-len(X):])

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self) -> None:
        obj = {"model": self.model, "scaler": self.scaler, "feature_cols": self.feature_cols}
        with open(self._model_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("XGBoost model saved → %s", self._model_path)

    def load(self) -> bool:
        if not os.path.exists(self._model_path):
            return False
        with open(self._model_path, "rb") as f:
            obj = pickle.load(f)
        self.model        = obj["model"]
        self.scaler       = obj["scaler"]
        self.feature_cols = obj["feature_cols"]
        logger.info("XGBoost model loaded ← %s", self._model_path)
        return True

    # ── Helpers ───────────────────────────────────────────────────────────
    def _prepare(
        self, df: pd.DataFrame,
        fit_scaler: bool = True,
        add_label: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
        if not self.feature_cols:
            self.feature_cols = cols
        X = df[self.feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
        if fit_scaler:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X), columns=self.feature_cols, index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X), columns=self.feature_cols, index=X.index
            )
        y = df["label"] if (add_label and "label" in df.columns) else None
        return X_scaled, y

    def _build_model(self):
        import xgboost as xgb
        params = {k: v for k, v in XC.items()
                  if k not in ("classes","retrain_every","early_stopping","eval_metric","use_gpu")}
        params["objective"]    = "multi:softprob"
        params["num_class"]    = 3
        params["tree_method"]  = "gpu_hist" if XC["use_gpu"] else "hist"
        params["eval_metric"]  = XC["eval_metric"]
        return xgb.XGBClassifier(
            early_stopping_rounds=XC["early_stopping"],
            **params
        )
