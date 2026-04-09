"""
LSTM price forecaster — predicts next N-bar return direction + magnitude.
Uses bidirectional LSTM + attention mechanism.
"""

import os
import logging
import numpy as np
import pandas as pd
import pickle
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from config import LSTM_CONFIG as LC, MODEL_DIR

logger = logging.getLogger(__name__)

FEATURE_COLS_EXCLUDE = {"label", "future_return"}


class LSTMForecaster:
    """
    Sequence model: predicts next `forecast_steps` bar returns.
    Converts to directional signal (BUY/HOLD/SELL) via threshold.
    """

    def __init__(self, symbol: str = "EURUSD", timeframe: str = "H1"):
        self.symbol    = symbol
        self.timeframe = timeframe
        self.model     = None
        self.scaler    = MinMaxScaler(feature_range=(-1, 1))
        self.feature_cols: list = []
        self._model_path = os.path.join(MODEL_DIR, f"lstm_{symbol}_{timeframe}.pkl")
        self._weights_path = os.path.join(MODEL_DIR, f"lstm_{symbol}_{timeframe}.weights.h5")

    # ── Training ──────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, verbose: int = 0) -> dict:
        """Build and train LSTM on feature DataFrame."""
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        X, y = self._build_sequences(df, fit_scaler=True)
        if len(X) < 100:
            raise ValueError(f"Too few sequences ({len(X)}) to train LSTM")

        split    = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        self.model = self._build_keras_model(X.shape[2])
        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=LC["patience"],
                restore_best_weights=True, verbose=verbose
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]
        history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=LC["epochs"],
            batch_size=LC["batch_size"],
            callbacks=cb,
            verbose=verbose,
        )
        self.save()
        final_val_loss = min(history.history["val_loss"])
        logger.info("LSTM trained | val_loss: %.6f", final_val_loss)
        return {"val_loss": final_val_loss}

    # ── Inference ─────────────────────────────────────────────────────────
    def predict_return(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predicted_return, confidence).
        predicted_return: scalar expected % return over forecast_steps bars
        """
        if self.model is None:
            if not self.load():
                raise RuntimeError("LSTM not trained — call train() first")
        X, _ = self._build_sequences(df, fit_scaler=False)
        if len(X) == 0:
            return np.array([0.0]), np.array([0.5])
        raw = self.model.predict(X[-1:], verbose=0)
        return raw.flatten(), np.abs(raw.flatten())

    def predict_signal(self, df: pd.DataFrame, threshold: float = 0.003) -> dict:
        """Convert LSTM forecast into BUY/SELL/HOLD signal."""
        pred_ret, _ = self.predict_return(df)
        r = float(pred_ret[0])
        if r > threshold:
            signal, name = 2, "BUY"
        elif r < -threshold:
            signal, name = 0, "SELL"
        else:
            signal, name = 1, "HOLD"
        # Confidence proportional to predicted magnitude
        confidence = min(abs(r) / (threshold * 3), 1.0)
        return {
            "signal":         signal,
            "signal_name":    name,
            "predicted_return": r,
            "confidence":     confidence,
        }

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self) -> None:
        meta = {"scaler": self.scaler, "feature_cols": self.feature_cols}
        with open(self._model_path, "wb") as f:
            pickle.dump(meta, f)
        if self.model:
            self.model.save_weights(self._weights_path)
        logger.info("LSTM saved → %s", self._model_path)

    def load(self) -> bool:
        if not os.path.exists(self._model_path):
            return False
        with open(self._model_path, "rb") as f:
            meta = pickle.load(f)
        self.scaler       = meta["scaler"]
        self.feature_cols = meta["feature_cols"]
        if os.path.exists(self._weights_path) and self.feature_cols:
            self.model = self._build_keras_model(len(self.feature_cols))
            self.model.load_weights(self._weights_path)
        logger.info("LSTM loaded ← %s", self._model_path)
        return True

    # ── Helpers ───────────────────────────────────────────────────────────
    def _build_sequences(
        self, df: pd.DataFrame, fit_scaler: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
        if not self.feature_cols:
            self.feature_cols = cols
        data = df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        if fit_scaler:
            data = self.scaler.fit_transform(data)
        else:
            data = self.scaler.transform(data)

        seq_len = LC["sequence_len"]
        horizon = LC["forecast_steps"]
        X, y = [], []
        for i in range(seq_len, len(data) - horizon):
            X.append(data[i - seq_len:i])
            if "future_return" in df.columns:
                y.append(df["future_return"].iloc[i])
        if not X:
            return np.array([]), np.array([])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32) if y else None
        return X, y

    def _build_keras_model(self, n_features: int):
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        inp = layers.Input(shape=(LC["sequence_len"], n_features))

        # Bidirectional LSTM stack
        x = layers.Bidirectional(
            layers.LSTM(LC["hidden_units"][0], return_sequences=True)
        )(inp)
        x = layers.Dropout(LC["dropout"])(x)
        x = layers.Bidirectional(
            layers.LSTM(LC["hidden_units"][1], return_sequences=True)
        )(x)
        x = layers.Dropout(LC["dropout"])(x)

        # Self-attention over time steps
        attn = layers.Dense(1, activation="tanh")(x)
        attn = layers.Softmax(axis=1)(attn)
        x    = layers.Multiply()([x, attn])
        x    = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

        # Dense head
        x   = layers.Dense(LC["dense_units"], activation="relu")(x)
        x   = layers.Dropout(LC["dropout"])(x)
        out = layers.Dense(1, activation="linear")(x)   # predicted return

        model = Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LC["learning_rate"]),
            loss="huber",
            metrics=["mae"],
        )
        return model
