import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor


# ----------------------------
# 1) Custom Transformers
# ----------------------------
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
def build_preprocessor():
    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_include=object)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_selector),
            ("cat", categorical_pipe, categorical_selector),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False
    )


def build_tree_feature_selector(
    n_estimators=200,
    max_features="sqrt",
    min_samples_leaf=2,
    threshold="median",
):
    estimator = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
    )
    return SelectFromModel(estimator=estimator, threshold=threshold)


def build_ridge_pipeline(alpha=1000.0):
    return Pipeline(steps=[
        ("drop_all_missing", DropAllMissing()),
        ("prep", build_preprocessor()),
        ("fs", build_tree_feature_selector(threshold="median")),
        ("model", Ridge(alpha=alpha, random_state=42)),
    ])


def build_gb_pipeline(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=10,
    subsample=0.8
):
    return Pipeline(steps=[
        ("drop_all_missing", DropAllMissing()),
        ("prep", build_preprocessor()),
        ("fs", build_tree_feature_selector(threshold="median")),
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


def build_histgb_pipeline(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=1.0
):
    return Pipeline(steps=[
        ("drop_all_missing", DropAllMissing()),
        ("prep", build_preprocessor()),
        ("model", HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )),
    ])


# ----------------------------
# 3) Evaluation helper
# ----------------------------
def evaluate_pipeline(pipe, X, y, name="model"):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse = -cross_val_score(pipe, X, y, scoring="neg_mean_squared_error", cv=cv).mean()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    test_mse = mean_squared_error(y_test, pred)

    print(f"\n[{name}] CV MSE:   {cv_mse:.4f}")
    print(f"[{name}] Test MSE: {test_mse:.4f}")
    return cv_mse, test_mse


# ----------------------------
# 4) Main
# ----------------------------
if __name__ == "__main__":
    # Load your dataset
    data_path = "/Users/alexandrebredillot/Documents/GitHub/ML-Assignement/training_dataset.csv"

    # NOTE: `sheet_name` is only valid for Excel files, not CSV.
    if data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path, low_memory=False)
    else:
        df = pd.read_excel(data_path, sheet_name="train")  # adjust if needed

    # List of irrelevant/redundant variables to remove
    to_drop = [
    # --- Metadata and Survey Design ---
    "HHX", "WTFA_A", "PPSU", "PSTRAT", "SRVY_YR", "INTV_QRT", "INTV_MON",
    "RECTYPE", "HHRESPSA_FLG", "IMPINCFLG_A", "SDMSRSOFT_A", "SDMSRS_A",

    # --- Redundant Demographics ---
    # Keeping AGEP_A and SEX_A
    "AGE65", "OVER65FLG_A", "MAFLG_A", "CHFLG_A",

    # --- Redundant Income/Poverty ---
    # Keeping POVRATTC_A
    "INCGRP_A", "RATCAT_A",

    # --- Redundant Race/Identity ---
    # Keeping HISPALLP_A and RACEALLP_A
    "HISP_A", "SASPPHISP_A", "SASPPRACE_A", "HISDETP_A",

    # --- Calculated Exercise Scores ---
    # Keeping PA18_02R_A
    "MODFREQW_A", "MODMIN_A", "VIGFREQW_A", "VIGMIN_A",
    "MODNR_A", "MODTPR_A", "VIGNR_A", "VIGTPR_A",

    # --- Target Leakage (Weight & BMI) ---
    # NOTE: Don't drop target yet, we need it!
    "BMICAT_A"
]

    df = df.drop(columns=to_drop, errors="ignore")
    print(f"Remaining variables: {df.shape[1]}")

    # Ensure target exists
    target = "WEIGHTLBTC_A"
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Check your sheet name/column names.")

    # Ensure target is numeric and drop rows with missing/invalid target
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # ------------------------------------------------------------
    # Tree-based feature selection at the beginning (Random Forest)
    # ------------------------------------------------------------
    # 1. Define your target (Y) and features (X)
    X = df.drop(columns=[target])
    y = df[target]

    # Handle missing values (Random Forest requires no NaNs)
    # Also encode object columns to numeric codes for RF importance computation.
    X_rf = X.copy()
    for c in X_rf.columns:
        if X_rf[c].dtype == "object":
            # factorize -> integer codes, missing becomes -1
            X_rf[c] = pd.factorize(X_rf[c], sort=True)[0].astype(np.int32)
            X_rf[c] = X_rf[c].replace(-1, np.nan)
            X_rf[c] = X_rf[c].fillna(-1)
        else:
            X_rf[c] = pd.to_numeric(X_rf[c], errors="coerce")

    # fill numeric NaNs with median (fallback to 0 if median is NaN)
    med = X_rf.median(numeric_only=True)
    X_rf = X_rf.fillna(med).fillna(0)

    # 2. Initialize and fit the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_rf, y)

    # 3. Get feature importances and rank them
    importances = pd.Series(model.feature_importances_, index=X_rf.columns)

    # Full ranking (descending)
    feature_ranking = (
        importances.sort_values(ascending=False)
        .rename("importance")
        .to_frame()
    )
    feature_ranking.insert(0, "rank", np.arange(1, len(feature_ranking) + 1))

    # Print full ranking (or slice if too long)
    pd.set_option("display.max_rows", None)
    print("\nFull feature importance ranking (desc):")
    print(feature_ranking)

    # Optional: save to CSV for your report
    feature_ranking.to_csv("feature_importance_ranking.csv", index_label="feature")


#Graph of the feature by importance
    # Plot top-K features by importance
    top_k = 30
    topk = feature_ranking.head(top_k).copy()
    topk = topk.iloc[::-1]  # reverse for nicer barh ordering

    plt.figure(figsize=(10, max(6, 0.25 * top_k)))
    plt.barh(topk.index, topk["importance"])
    plt.xlabel("RandomForest feature importance")
    plt.title(f"Top {top_k} features by importance")
    plt.tight_layout()
    plt.savefig(f"feature_importance_top{top_k}.png", dpi=200)
    # plt.show()  # uncomment if you want an interactive window
    print(f"\nSaved plot: feature_importance_top{top_k}.png")

    # Keep top 50 as you already do
    top_50_features = importances.sort_values(ascending=False).head(50)

    # 4. Keep only the top 50 variables in your dataframe
    top_feature_names = top_50_features.index.tolist()
    df_reduced = df[top_feature_names + [target]]

    # 5. Print the Top 10 for your report
    print("Top 10 Most Predictive Features:")
    print(top_50_features.head(10))

    # Use reduced dataframe from here on
    df = df_reduced

    # Rebuild X/y for the pipelines (with original dtypes preserved in df)
    y = df[target]
    X = df.drop(columns=[target])

    # Baseline Ridge
    ridge_pipe = build_ridge_pipeline(alpha=1000.0)
    evaluate_pipeline(ridge_pipe, X, y, name="Ridge")

    # Base GB
    gb_pipe = build_gb_pipeline()
    evaluate_pipeline(gb_pipe, X, y, name="GradientBoosting (base, L2)")

    # ---- Tune GB (GridSearchCV) ----
    param_grid = {
        "model__n_estimators": [300, 600],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [10, 20],
        "model__subsample": [0.8, 1.0],
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

    # Base HistGB
    histgb_pipe = build_histgb_pipeline()
    evaluate_pipeline(histgb_pipe, X, y, name="HistGradientBoosting")