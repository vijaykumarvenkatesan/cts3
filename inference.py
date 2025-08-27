# inference.py
import os
import joblib
import pickle
import numpy as np
import pandas as pd

# try to reuse DBconnect blob loader if available
try:
    import DBconnect as dbconnect
    _has_dbconnect = True
except Exception:
    dbconnect = None
    _has_dbconnect = False

class ModelBundle:
    def __init__(self, model_path=None, encoders_path=None, metadata_path=None, model_obj=None, encoders_obj=None, metadata_obj=None):
        """
        Initialize bundle either from paths (local) or directly from objects.
        """
        if model_obj is not None:
            self.model = model_obj
        else:
            self.model = self._load_any(model_path)

        if encoders_obj is not None:
            self.encoders = encoders_obj
        else:
            self.encoders = self._load_any(encoders_path) if encoders_path else None

        if metadata_obj is not None:
            self.metadata = metadata_obj
        else:
            self.metadata = self._load_any(metadata_path) if metadata_path else None

    def _load_any(self, path):
        if not path or not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    @classmethod
    def load_from_azure(cls):
        """Load model, encoders, metadata from Azure Blob Storage using DBconnect helper.

        Returns a ModelBundle instance or None on failure.
        """
        if not _has_dbconnect:
            return None
        model, encoders, selected_features, optimal_threshold = dbconnect.load_model_components_from_azure()
        if model is None:
            return None
        # Compose metadata similar to your local metadata layout so preprocess_for_model can read it
        metadata = {
            'model_info': {
                'features_used': selected_features,
                'optimal_threshold': optimal_threshold
            }
        }
        return cls(model_obj=model, encoders_obj=encoders, metadata_obj=metadata)

    def predict_dataframe(self, X: pd.DataFrame, optimal_threshold: float = None):
        # if threshold not provided, try metadata
        if optimal_threshold is None and isinstance(self.metadata, dict):
            try:
                optimal_threshold = float(self.metadata.get('model_info', {}).get('optimal_threshold', 0.5))
            except Exception:
                optimal_threshold = 0.5
        if optimal_threshold is None:
            optimal_threshold = 0.5

        # predict_proba preferred
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[:, 1]
            std = (proba >= 0.5).astype(int)
            opt = (proba >= float(optimal_threshold)).astype(int)
        else:
            std = self.model.predict(X)
            proba = std.astype(float)
            opt = std
        return proba, std, opt
