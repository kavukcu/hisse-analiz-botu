import os
import joblib
import numpy as np
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    VotingRegressor,
)

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


class AIEngine:

    def __init__(self):

    self.models = {}

    self.model = None

    self.model_file = "ai_model.pkl"

    self.model_dir = "models"

    self.metrics = {}

    self.feature_importance = {}

    self.confidence = 0.0

    os.makedirs(self.model_dir, exist_ok=True)

    def build_model(self):

        models = []

        models.append(
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )

        models.append(
            (
                "gb",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                ),
            )
        )

        models.append(
            (
                "et",
                ExtraTreesRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )

        models.append(
            (
                "ridge",
                Ridge(alpha=1.0),
            )
        )

        models.append(
            (
                "svr",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("svr", SVR(C=5.0, epsilon=0.01)),
                    ]
                ),
            )
        )

        self.model = VotingRegressor(models)

    from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

def fit(self, X, y):

    self.build_model()

    tscv = TimeSeriesSplit(n_splits=5)

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    for train_idx, test_idx in tscv.split(X):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        self.model.fit(X_train, y_train)

        pred = self.model.predict(X_test)

        mae_scores.append(
            mean_absolute_error(y_test, pred)
        )

        rmse_scores.append(
            np.sqrt(
                mean_squared_error(y_test, pred)
            )
        )

        r2_scores.append(
            r2_score(y_test, pred)
        )

    print("=" * 50)

    print("AI MODEL RAPORU")

    print("=" * 50)

    print(f"MAE : {np.mean(mae_scores):.5f}")

    print(f"RMSE: {np.mean(rmse_scores):.5f}")

    print(f"R²  : {np.mean(r2_scores):.5f}")

    self.model.fit(X, y)

    joblib.dump(
        self.model,
        self.model_file,
    )
    self.metrics = {

    "MAE": float(np.mean(mae_scores)),

    "RMSE": float(np.mean(rmse_scores)),

    "R2": float(np.mean(r2_scores))

}
    def predict(self, X):
        def get_metrics(self):

    return self.metrics


def get_confidence(self):

    return self.confidence


def calculate_confidence(self, predictions):

    predictions = np.array(predictions)

    std = np.std(predictions)

    mean = np.mean(np.abs(predictions))

    if mean == 0:

        self.confidence = 0.0

    else:

        self.confidence = max(
            0,
            min(
                100,
                100 - (std / mean) * 100
            )
        )

    return self.confidence

        if self.model is None:

            if os.path.exists(self.model_file):

                self.model = joblib.load(self.model_file)

            else:

                raise Exception("Model eğitilmedi.")

        prediction = self.model.predict(X)

self.calculate_confidence(prediction)

return prediction
    if XGB_AVAILABLE:

    models.append(

        (

            "xgb",

            XGBRegressor(

                n_estimators=300,

                learning_rate=0.03,

                max_depth=6,

                subsample=0.9,

                colsample_bytree=0.9,

                random_state=42,

            ),

        )

    )


if LGBM_AVAILABLE:

    models.append(

        (

            "lgbm",

            LGBMRegressor(

                n_estimators=300,

                learning_rate=0.03,

                random_state=42,

            ),

        )

    )


if CATBOOST_AVAILABLE:

    models.append(

        (

            "cat",

            CatBoostRegressor(

                iterations=300,

                learning_rate=0.03,

                depth=6,

                verbose=False,

            ),

        )

    )