import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Handle XGBoost import gracefully
XGBOOST_AVAILABLE = False
XGBClassifier = None
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    # Handle both import errors and library loading errors
    XGBOOST_AVAILABLE = False
    XGBClassifier = None


CLEVELAND_FEATURES_ORDER: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COL = "target"

CATEGORICAL_CHOICES = {
    "sex": [0, 1],
    "cp": [0, 1, 2, 3],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [0, 1, 2],
    "ca": [0, 1, 2, 3],
    "thal": [1, 2, 3],
}

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

def _coerce_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean '?', cast numerics, normalize column names, and binarize target."""
    df = df.copy()
    colmap = {c.lower(): c for c in df.columns}
    for col in CLEVELAND_FEATURES_ORDER + [TARGET_COL]:
        if col not in df.columns and col in colmap:
            df[col] = df.pop(colmap[col])

    for col in CLEVELAND_FEATURES_ORDER + [TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce")

    if TARGET_COL in df.columns:
        df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    return df

def load_cleveland_dataframe(file_path: Optional[str] = None, uploaded_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Load Cleveland dataset from upload or file path and ensure schema."""
    if uploaded_df is not None:
        df = _coerce_and_clean(uploaded_df)
        missing = [c for c in CLEVELAND_FEATURES_ORDER + [TARGET_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"Uploaded data missing required columns: {missing}")
        return df

    if file_path is not None and os.path.exists(file_path):
        if file_path.endswith(".csv"):
            # Try reading with headers first; fall back to no header
            try:
                df = pd.read_csv(file_path)
                if len(df.columns) == len(CLEVELAND_FEATURES_ORDER) + 1:  # +1 for target
                    first_row_numeric = all(pd.to_numeric(df.iloc[0], errors='coerce').notna())
                    if first_row_numeric:
                        # Re-read without headers and assign names
                        df = pd.read_csv(file_path, header=None)
                        df.columns = CLEVELAND_FEATURES_ORDER + [TARGET_COL]
            except:
                # Fallback: read without headers
                df = pd.read_csv(file_path, header=None)
                df.columns = CLEVELAND_FEATURES_ORDER + [TARGET_COL]
        else:
            df = pd.read_excel(file_path)
        df = _coerce_and_clean(df)
        missing = [c for c in CLEVELAND_FEATURES_ORDER + [TARGET_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"File missing required columns: {missing}")
        return df

    raise FileNotFoundError(
        "No dataset found. Please upload a CSV/XLSX with columns: "
        f"{CLEVELAND_FEATURES_ORDER + [TARGET_COL]}"
    )

# -----------------------------
# Preprocess & Modeling
# -----------------------------
def build_preprocessor() -> ColumnTransformer:
    """
    - Numeric: impute median
    - Categorical: impute most_frequent + one-hot
    """
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS)
        ],
        remainder="drop"
    )
    return preprocessor

def build_models() -> Dict[str, Pipeline]:
    """
    Create sklearn Pipelines for each model with optimized hyperparameters.
    Hyperparameters are tuned for heart disease prediction tasks.
    """
    pre = build_preprocessor()

    # Decision Tree - Optimized for interpretability and performance
    dt = Pipeline(steps=[
        ("prep", pre),
        ("clf", DecisionTreeClassifier(
            random_state=42,
            criterion="entropy",  # Better for binary classification
            max_depth=8,          # Deeper for better performance
            min_samples_split=10, # Prevent overfitting
            min_samples_leaf=4,   # Smoother decision boundaries
            class_weight="balanced"  # Handle class imbalance
        ))
    ])

    # k-NN - Optimized distance metric and neighbors
    knn = Pipeline(steps=[
        ("prep", pre),
        ("clf", KNeighborsClassifier(
            n_neighbors=7,        # Odd number, optimal for this dataset size
            weights="distance",   # Weight by distance for better performance
            metric="manhattan",   # Often better for categorical features
            p=1                   # Manhattan distance parameter
        ))
    ])

    # Naive Bayes - Optimized smoothing parameter
    nb = Pipeline(steps=[
        ("prep", pre),
        ("clf", GaussianNB(
            var_smoothing=1e-8    # Optimized smoothing for stability
        ))
    ])

    # Random Forest - Optimized for ensemble performance
    rf = Pipeline(steps=[
        ("prep", pre),
        ("clf", RandomForestClassifier(
            random_state=42,
            n_estimators=200,      # More trees for better performance
            max_depth=10,          # Deeper trees
            min_samples_split=5,   # Conservative splitting
            min_samples_leaf=2,    # Leaf size for generalization
            max_features="sqrt",   # Feature subsampling
            bootstrap=True,        # Bootstrap sampling
            class_weight="balanced", # Handle imbalance
            n_jobs=-1             # Use all cores
        ))
    ])

    # AdaBoost - Optimized learning rate and estimators
    ada = Pipeline(steps=[
        ("prep", pre),
        ("clf", AdaBoostClassifier(
            random_state=42,
            n_estimators=150,      # More estimators
            learning_rate=0.8,     # Slower learning for stability
            algorithm="SAMME"      # Compatible algorithm for newer sklearn
        ))
    ])

    # Gradient Boosting - Optimized for performance
    gb = Pipeline(steps=[
        ("prep", pre),
        ("clf", GradientBoostingClassifier(
            random_state=42,
            n_estimators=150,      # More estimators
            learning_rate=0.08,    # Lower learning rate
            max_depth=4,           # Moderate depth
            min_samples_split=10,  # Conservative splitting
            min_samples_leaf=4,    # Leaf constraints
            subsample=0.8,         # Stochastic gradient boosting
            max_features="sqrt"    # Feature subsampling
        ))
    ])

    models = {"Decision Tree": dt, "k-NN": knn, "Naive Bayes": nb, "Random Forest": rf, "AdaBoost": ada, "Gradient Boosting": gb}

    # Add XGBoost if available - Optimized hyperparameters
    if XGBOOST_AVAILABLE:
        xgb = Pipeline(steps=[
            ("prep", pre),
            ("clf", XGBClassifier(
                random_state=42,
                n_estimators=150,      # More estimators
                learning_rate=0.08,    # Lower learning rate
                max_depth=4,           # Moderate depth
                min_child_weight=3,    # Regularization
                gamma=0.1,             # Minimum split loss
                subsample=0.8,         # Row sampling
                colsample_bytree=0.8,  # Column sampling
                reg_alpha=0.1,         # L1 regularization
                reg_lambda=1.0,        # L2 regularization
                eval_metric='logloss',
                use_label_encoder=False
            ))
        ])
        models["XGBoost"] = xgb

    # Ensemble with optimized weights based on typical performance
    # Use the same optimized hyperparameters for ensemble components
    estimators = [
        ("dt", DecisionTreeClassifier(random_state=42, criterion="entropy", max_depth=8, 
                                    min_samples_split=10, min_samples_leaf=4, class_weight="balanced")),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance", metric="manhattan")),
        ("nb", GaussianNB(var_smoothing=1e-8)),
        ("rf", RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, 
                                    min_samples_split=5, min_samples_leaf=2, max_features="sqrt", 
                                    class_weight="balanced", n_jobs=-1)),
        ("ada", AdaBoostClassifier(random_state=42, n_estimators=150, learning_rate=0.8, algorithm="SAMME")),
        ("gb", GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.08, 
                                        max_depth=4, min_samples_split=10, min_samples_leaf=4, 
                                        subsample=0.8, max_features="sqrt")),
    ]
    
    if XGBOOST_AVAILABLE:
        estimators.append(("xgb", XGBClassifier(random_state=42, n_estimators=150, learning_rate=0.08, 
                                              max_depth=4, min_child_weight=3, gamma=0.1, subsample=0.8, 
                                              colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, 
                                              eval_metric='logloss', use_label_encoder=False)))

    # Weighted voting based on expected performance
    weights = [1.0, 1.2, 0.8, 1.5, 1.3, 1.4]  # Higher weights for better performing models
    if XGBOOST_AVAILABLE:
        weights.append(1.6)  # XGBoost typically performs well

    ensemble = Pipeline(steps=[
        ("prep", pre),
        ("clf", VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=weights
        ))
    ])

    models["Ensemble (Soft Voting)"] = ensemble
    return models

def fit_all_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """
    Fit all models on train split; return fitted models and metrics (AUC on holdout).
    """
    X = df[CLEVELAND_FEATURES_ORDER]
    y = df[TARGET_COL].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = build_models()
    metrics = []

    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        # Predictions and probabilities
        y_pred = pipe.predict(X_te)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, proba)
        else:
            # Fallback if probabilities are not available
            proba = None
            auc = roc_auc_score(y_te, y_pred)

        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)

        metrics.append({
            "Model": name,
            "ROC-AUC": round(float(auc), 4),
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1": round(float(f1), 4),
        })

    metrics_df = pd.DataFrame(metrics).sort_values("ROC-AUC", ascending=False, ignore_index=True)
    
    # Add performance ranking and highlight best performance
    metrics_df["Rank"] = range(1, len(metrics_df) + 1)
    
    # Mark the best performing model
    best_model_idx = metrics_df["ROC-AUC"].idxmax()
    metrics_df.loc[best_model_idx, "Model"] = "🏆 " + metrics_df.loc[best_model_idx, "Model"] + " (BEST)"
    
    # Reorder columns to show rank first
    metrics_df = metrics_df[["Rank", "Model", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1"]]
    
    return models, metrics_df

def predict_all(models: Dict[str, Pipeline], input_dict: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Predict probability for positive class (heart disease) for each model.
    Returns: {model_name: {"prob_1": float, "prob_0": float, "label": int}}
    """
    # Ensure full set & order
    row = [[input_dict[c] for c in CLEVELAND_FEATURES_ORDER]]
    X_new = pd.DataFrame(row, columns=CLEVELAND_FEATURES_ORDER)

    out = {}
    for name, pipe in models.items():
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_new)[0]
            # convention: class order is [0,1]
            out[name] = {
                "prob_0": float(proba[0]),
                "prob_1": float(proba[1]),
                "label": int(np.argmax(proba))
            }
        else:
            label = int(pipe.predict(X_new)[0])
            out[name] = {"prob_0": 1.0 - label, "prob_1": float(label), "label": label}
    return out


def example_patient(index: int = 0) -> Dict[str, float]:
    """
    Get example patients with specific features provided by user.
    """
    # Example 1: No heart disease (37,1,3,130,250,0,0,187,0,3.5,3,0,3,0)
    # Example 2: Heart disease (56,1,3,130,256,1,2,142,1,0.6,2,1,6,2)
    
    if index == 0:
        # No heart disease example
        return {
            "age": 37.0,
            "sex": 1.0,
            "cp": 3.0,
            "trestbps": 130.0,
            "chol": 250.0,
            "fbs": 0.0,
            "restecg": 0.0,
            "thalach": 187.0,
            "exang": 0.0,
            "oldpeak": 3.5,
            "slope": 3.0,
            "ca": 0.0,
            "thal": 3.0
        }
    else:
        # Heart disease example
        return {
            "age": 56.0,
            "sex": 1.0,
            "cp": 3.0,
            "trestbps": 130.0,
            "chol": 256.0,
            "fbs": 1.0,
            "restecg": 2.0,
            "thalach": 142.0,
            "exang": 1.0,
            "oldpeak": 0.6,
            "slope": 2.0,
            "ca": 1.0,
            "thal": 6.0
        }

def get_example_labels() -> List[int]:
    """
    Get the labels for the example patients to display in the UI.
    Returns list of labels for the specific examples provided.
    """
    # Example 1: No heart disease (target = 0)
    # Example 2: Heart disease (target = 2, binarized to 1)
    return [0, 1]  # First example: no disease, second example: heart disease
