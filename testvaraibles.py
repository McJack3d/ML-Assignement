import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

DATA_PATH = "/Users/alexandrebredillot/Documents/GitHub/ML-Assignement/training_dataset.csv"
TARGET = "WEIGHTLBTC_A"

# Columns you already exclude in pipeline3.py (kept in sync)
TO_DROP = [
    "HHX","WTFA_A","PPSU","PSTRAT","SRVY_YR","INTV_QRT","INTV_MON",
    "RECTYPE","HHRESPSA_FLG","IMPINCFLG_A","SDMSRSOFT_A","SDMSRS_A",
    "AGE65","OVER65FLG_A","MAFLG_A","CHFLG_A",
    "INCGRP_A","RATCAT_A",
    "HISP_A","SASPPHISP_A","SASPPRACE_A","HISDETP_A",
    "MODFREQW_A","MODMIN_A","VIGFREQW_A","VIGMIN_A",
    "MODNR_A","MODTPR_A","VIGNR_A","VIGTPR_A",
    "BMICAT_A",  # avoid leakage from target-derived feature
]

def build_preprocessor():
    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_include=object)

    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[("num", num, numeric_selector),
                      ("cat", cat, categorical_selector)],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False
    )

def aggregate_raw_importances(pipe):
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    importances = model.feature_importances_

    # Resolved columns after fit
    num_cols = prep.transformers_[0][2]  # numeric column names
    cat_cols = prep.transformers_[1][2]  # categorical column names

    # OneHotEncoder categories per categorical column
    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    ohe_cats = ohe.categories_

    raw_imp = {}

    # Numeric part (same order as num_cols)
    idx = 0
    for c in num_cols:
        raw_imp[c] = raw_imp.get(c, 0.0) + float(importances[idx])
        idx += 1

    # Categorical part: sum block of importances per raw col
    for c, cats in zip(cat_cols, ohe_cats):
        block = importances[idx: idx + len(cats)]
        raw_imp[c] = raw_imp.get(c, 0.0) + float(block.sum())
        idx += len(cats)

    return pd.Series(raw_imp).sort_values(ascending=False)

def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.drop(columns=TO_DROP, errors="ignore")
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not in data.")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET]).reset_index(drop=True)

    # Drop columns that are entirely missing (prevents imputer warnings)
    all_nan_cols = [c for c in df.columns if c != TARGET and df[c].notna().sum() == 0]
    if all_nan_cols:
        print(f"Dropping all-NaN columns: {len(all_nan_cols)}")
        df = df.drop(columns=all_nan_cols)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()
    gb = GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=600,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", gb),
    ])

    pipe.fit(X_train, y_train)

    # 1) Transformed feature importances (as you already had)
    feat_names = pipe.named_steps["prep"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    print("\nTop 30 transformed features by model importance:")
    print(fi.head(30))
    fi.to_csv("feature_importance_model_based.csv", header=["importance"])

    # 1b) Aggregated to RAW columns (sum OHE blocks)
    raw_fi = aggregate_raw_importances(pipe)
    print("\nTop 30 RAW columns by aggregated model importance:")
    print(raw_fi.head(30))
    raw_fi.to_csv("feature_importance_raw_aggregated.csv", header=["importance"])

    # 2) FAST permutation importance on TOP-K raw columns only
    TOP_K = 50
    top_raw_cols = raw_fi.head(TOP_K).index.tolist()

    # Refit a small pipeline restricted to top-K columns (much faster)
    X_train_small = X_train[top_raw_cols].copy()
    X_test_small = X_test[top_raw_cols].copy()

    pipe_small = Pipeline([
        ("prep", build_preprocessor()),
        ("model", GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )),
    ])
    pipe_small.fit(X_train_small, y_train)

    perm = permutation_importance(
        pipe_small, X_test_small, y_test,
        n_repeats=3, random_state=42, n_jobs=-1, scoring="r2"
    )
    perm_series = pd.Series(perm.importances_mean, index=top_raw_cols).sort_values(ascending=False)
    print("\nTop 30 raw columns by permutation importance (ΔR²) on TOP-K:")
    print(perm_series.head(30))
    perm_series.to_csv("feature_importance_permutation_topK.csv", header=["delta_r2_mean"])

    # Suggest columns to drop (near-zero ΔR² among evaluated TOP-K)
    drop_threshold = 0.0002
    low_impact_cols = perm_series[perm_series <= drop_threshold].index.tolist()
    print(f"\nSuggested low-impact columns among top-{TOP_K} (<= {drop_threshold} ΔR²): {len(low_impact_cols)}")
    print(low_impact_cols[:50])
    pd.Series(low_impact_cols).to_csv("suggested_drop_columns_topK.csv", index=False, header=["column"])

    # Final test score with full model (for reference)
    y_pred = pipe.predict(X_test)
    print(f"\nTest R² (full model): {r2_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()