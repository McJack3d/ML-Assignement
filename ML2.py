import numpy as np
import pandas as pd
import inspect
from pathlib import Path
import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor


# ----------------------------
# 1) Custom cleaners (inside pipeline)
# ----------------------------
class NHISCleaner(BaseEstimator, TransformerMixin):
    """
    - Drops survey design/weight columns (WT/PSU/STRAT/RECTYPE patterns)
    - Drops ID-like object columns (very high cardinality)
    - Replaces NHIS missing codes with NaN (numeric columns only)
    """
    def __init__(
        self,
        missing_codes=(7, 8, 9, 97, 98, 99, 997, 998, 999),
        drop_patterns=("WT", "PSU", "STRAT", "RECTYPE"),
        id_unique_ratio_threshold=0.98,
        max_id_nunique=5000,
        numeric_as_category_max_unique=30,
        numeric_as_category_max_ratio=0.05,
        verbose=False
    ):
        self.missing_codes = missing_codes
        self.drop_patterns = drop_patterns
        self.id_unique_ratio_threshold = id_unique_ratio_threshold
        self.max_id_nunique = max_id_nunique
        self.numeric_as_category_max_unique = numeric_as_category_max_unique
        self.numeric_as_category_max_ratio = numeric_as_category_max_ratio
        self.verbose = verbose

    def fit(self, X, y=None):
        X = X.copy()

        # Drop by name pattern
        pattern_drop = []
        upper_cols = {c: str(c).upper() for c in X.columns}
        for c, cu in upper_cols.items():
            if any(pat in cu for pat in self.drop_patterns):
                pattern_drop.append(c)

        # ID-like object columns
        id_like = []
        n = len(X)
        for c in X.columns:
            if X[c].dtype == "object":
                nunique = X[c].nunique(dropna=True)
                ratio = nunique / max(n, 1)
                if (ratio >= self.id_unique_ratio_threshold) and (nunique <= self.max_id_nunique or ratio > 0.995):
                    id_like.append(c)

        self.drop_cols_ = sorted(set(pattern_drop + id_like))

        # Treat low-cardinality numeric codes as categorical (helps linear models a lot)
        X2 = X.drop(columns=self.drop_cols_, errors="ignore").copy()
        for c in X2.columns:
            if pd.api.types.is_numeric_dtype(X2[c]):
                X2[c] = X2[c].replace(list(self.missing_codes), np.nan)

        numeric_as_cat = []
        n = len(X2)
        for c in X2.columns:
            if pd.api.types.is_numeric_dtype(X2[c]):
                nunique = X2[c].nunique(dropna=True)
                ratio = nunique / max(n, 1)
                if (nunique > 1) and (nunique <= self.numeric_as_category_max_unique) and (ratio <= self.numeric_as_category_max_ratio):
                    numeric_as_cat.append(c)

        self.numeric_as_cat_cols_ = sorted(numeric_as_cat)

        if self.verbose:
            print(f"[NHISCleaner] Dropping {len(self.drop_cols_)} columns")
            print(f"[NHISCleaner] Treating {len(self.numeric_as_cat_cols_)} numeric columns as categorical")
            if len(self.drop_cols_) < 50:
                print("Dropped columns:", self.drop_cols_)

        return self

    def transform(self, X):
        X = X.copy()

        # Drop identified columns
        X = X.drop(columns=self.drop_cols_, errors="ignore")

        # Replace missing codes in numeric columns
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].replace(list(self.missing_codes), np.nan)

        # Cast selected numeric code columns to object so they go through OneHotEncoder
        for c in getattr(self, "numeric_as_cat_cols_", []):
            if c in X.columns:
                X[c] = X[c].astype("Int64").astype(str)

        return X

class DropAllMissing(BaseEstimator, TransformerMixin):
    """Drops columns that are entirely missing (all NaN)."""
    def fit(self, X, y=None):
        self.keep_cols_ = [c for c in X.columns if not X[c].isna().all()]
        self.drop_cols_ = [c for c in X.columns if c not in self.keep_cols_]
        return self

    def transform(self, X):
        return X[self.keep_cols_].copy()


# ----------------------------
# 2) Preprocessor + model pipelines
# ----------------------------
def build_preprocessor_linear():
    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_include=object)

    def _make_one_hot_encoder():
        sig = inspect.signature(OneHotEncoder)
        # Keep sparse output to avoid huge dense matrices (much faster for many categories)
        if "sparse_output" in sig.parameters:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # with sparse features present, we must not center
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_selector),
            ("cat", categorical_pipe, categorical_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


def build_preprocessor_tree():
    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_include=object)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_selector),
            ("cat", categorical_pipe, categorical_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_ridge_pipeline(alpha=1000.0):
    return Pipeline(steps=[
        ("clean", NHISCleaner(verbose=False, numeric_as_category_max_unique=30, numeric_as_category_max_ratio=0.05)),
        ("drop_all_missing", DropAllMissing()),
        ("prep", build_preprocessor_linear()),
        # sparse-friendly solver (avoids dense SVD)
        ("model", Ridge(alpha=alpha, solver="sparse_cg")),
    ])


def build_ridgecv_pipeline():
    return Pipeline(steps=[
        ("clean", NHISCleaner(verbose=False, numeric_as_category_max_unique=30, numeric_as_category_max_ratio=0.05)),
        ("drop_all_missing", DropAllMissing()),
        ("prep", build_preprocessor_linear()),
        ("model", RidgeCV(alphas=np.logspace(-2, 4, 25), cv=5)),
    ])


def build_gb_pipeline(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=10,
    subsample=0.8
):
    return Pipeline(steps=[
        # For tree models, keep numeric codes numeric (avoid one-hot blow-up)
        ("clean", NHISCleaner(verbose=False, numeric_as_category_max_unique=0, numeric_as_category_max_ratio=0.0)),
        ("drop_all_missing", DropAllMissing()),
        ("prep", build_preprocessor_tree()),
        ("model", GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=42
        )),
    ])


def build_hist_gb_pipeline(
    learning_rate=0.05,
    max_depth=None,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0
):
    return Pipeline(steps=[
        ("clean", NHISCleaner(verbose=False, numeric_as_category_max_unique=0, numeric_as_category_max_ratio=0.0)),
        ("drop_all_missing", DropAllMissing()),
        ("prep", build_preprocessor_tree()),
        ("model", HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=True,
            random_state=42
        )),
    ])


def build_rf_pipeline(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=5,
    max_features="sqrt"
):
    return Pipeline(steps=[
        ("clean", NHISCleaner(verbose=False, numeric_as_category_max_unique=0, numeric_as_category_max_ratio=0.0)),
        ("drop_all_missing", DropAllMissing()),
        ("prep", build_preprocessor_tree()),  # ordinal for categoricals
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=42
        )),
    ])


def with_log_target(regressor):
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )



# ----------------------------
# 3) Evaluation helper
# ----------------------------
def evaluate_pipeline(pipe, X, y, name="model"):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse = -cross_val_score(
        pipe, X, y,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,   # parallelize folds
    ).mean()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    test_mse = mean_squared_error(y_test, pred)

    cv_rmse = float(np.sqrt(cv_mse))
    test_rmse = float(np.sqrt(test_mse))

    print(f"\n[{name}] CV MSE:   {cv_mse:.4f}")
    print(f"[{name}] Test MSE: {test_mse:.4f}")
    print(f"[{name}] CV RMSE:  {cv_rmse:.4f}")
    print(f"[{name}] Test RMSE:{test_rmse:.4f}")
    return cv_mse, test_mse


# ----------------------------
# 4) Main
# ----------------------------
def load_training_df(excel_path: str, sheet_name: str, target: str, drop_patterns=("WT", "PSU", "STRAT", "RECTYPE")) -> pd.DataFrame:
    """
    Speed-ups:
    1) Read only header first, compute usecols to skip WT/PSU/STRAT/RECTYPE columns early.
    2) Cache to parquet (or pickle fallback) to avoid re-parsing Excel every run.
    """
    excel_path = Path(excel_path)
    parquet_cache = excel_path.with_suffix(".train.cache.parquet")
    pickle_cache = excel_path.with_suffix(".train.cache.pkl")

    if parquet_cache.exists():
        t0 = time.perf_counter()
        df = pd.read_parquet(parquet_cache)
        print(f"[load] loaded parquet cache in {time.perf_counter() - t0:.2f}s: {parquet_cache.name}")
        return df

    if pickle_cache.exists():
        t0 = time.perf_counter()
        df = pd.read_pickle(pickle_cache)
        print(f"[load] loaded pickle cache in {time.perf_counter() - t0:.2f}s: {pickle_cache.name}")
        return df

    # 1) read header only (fast) to build usecols
    t0 = time.perf_counter()
    header = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=0, engine="openpyxl")
    cols = list(header.columns)

    if target not in cols:
        raise ValueError(f"Target '{target}' not found in sheet '{sheet_name}'. Available columns: {len(cols)}")

    drop_patterns_u = tuple(p.upper() for p in drop_patterns)
    usecols = [c for c in cols if not any(p in str(c).upper() for p in drop_patterns_u)]
    if target not in usecols:
        usecols.append(target)

    print(f"[load] header read in {time.perf_counter() - t0:.2f}s. Reading {len(usecols)}/{len(cols)} columns...")

    # 2) read filtered columns
    t1 = time.perf_counter()
    # try faster engine if installed, else fallback to openpyxl
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=usecols, engine="calamine")
        engine_used = "calamine"
    except Exception:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=usecols, engine="openpyxl")
        engine_used = "openpyxl"

    print(f"[load] excel read ({engine_used}) in {time.perf_counter() - t1:.2f}s. Shape={df.shape}")

    # 3) cache for next runs
    try:
        df.to_parquet(parquet_cache, index=False)
        print(f"[load] wrote parquet cache: {parquet_cache.name}")
    except Exception:
        df.to_pickle(pickle_cache)
        print(f"[load] wrote pickle cache: {pickle_cache.name}")

    return df


if __name__ == "__main__":
    excel_path = "/Users/alexandrebredillot/Documents/GitHub/ML-Assignement/training_dataset.xlsx"
    sheet_name = "train"
    target = "WEIGHTLBTC_A"

    df = load_training_df(excel_path, sheet_name=sheet_name, target=target)

    y = df[target]
    X = df.drop(columns=[target])

    # Baseline Ridge
    ridge_pipe = build_ridge_pipeline(alpha=1000.0)
    evaluate_pipeline(ridge_pipe, X, y, name="Ridge")

    # Ridge with automatic alpha selection
    ridgecv_pipe = build_ridgecv_pipeline()
    evaluate_pipeline(ridgecv_pipe, X, y, name="RidgeCV")

    # Often improves heavily-skewed targets (like weight/income/etc)
    evaluate_pipeline(with_log_target(ridgecv_pipe), X, y, name="RidgeCV (log-target)")

    # Base GB
    gb_pipe = build_gb_pipeline()
    evaluate_pipeline(gb_pipe, X, y, name="GradientBoosting (base, L2)")

    # Stronger/faster baseline on tabular data
    hgb_pipe = build_hist_gb_pipeline()
    evaluate_pipeline(hgb_pipe, X, y, name="HistGradientBoosting (base)")
    evaluate_pipeline(with_log_target(hgb_pipe), X, y, name="HistGradientBoosting (log-target)")

    # ---- Tune GB (GridSearchCV) ----
    param_grid = {
        "model__n_estimators": [300, 600],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [10, 20],
        "model__subsample": [0.8, 1.0],
        "model__max_features": [None, "sqrt"],
    }

    gb_search = GridSearchCV(
        estimator=gb_pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=2
    )

    gb_search.fit(X, y)

    print("\n[Tuned GB] Best CV MSE:", -gb_search.best_score_)
    print("[Tuned GB] Best parameters:")
    for k, v in gb_search.best_params_.items():
        print(f"  {k}: {v}")

    # Evaluate the tuned best estimator on a holdout split
    best_gb = gb_search.best_estimator_
    evaluate_pipeline(best_gb, X, y, name="GradientBoosting (tuned, L2)")

    # ---- Tune HistGB (often best score/compute tradeoff) ----
    hgb_param_grid = {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 50],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    }

    hgb_search = GridSearchCV(
        estimator=hgb_pipe,
        param_grid=hgb_param_grid,
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=2,
    )

    hgb_search.fit(X, y)

    print("\n[Tuned HistGB] Best CV MSE:", -hgb_search.best_score_)
    print("[Tuned HistGB] Best parameters:")
    for k, v in hgb_search.best_params_.items():
        print(f"  {k}: {v}")

    best_hgb = hgb_search.best_estimator_
    evaluate_pipeline(best_hgb, X, y, name="HistGradientBoosting (tuned)")
    evaluate_pipeline(with_log_target(best_hgb), X, y, name="HistGradientBoosting (tuned, log-target)")