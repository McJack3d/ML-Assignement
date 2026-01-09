import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor


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


# Best GB parameters from tuning
BEST_GB_PARAMS = {
    "n_estimators": 1000,      # More trees for slow learning
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_samples_leaf": 10,
    "subsample": 0.8,
}


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
def evaluate_pipeline(pipe, X_train, X_test, y_train, y_test, name="model"):
    pipe.fit(X_train, y_train)
    
    # Training metrics
    train_pred = pipe.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    
    # Test metrics
    test_pred = pipe.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    print(f"\n[{name}]")
    print(f"Training - MSE: {train_mse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
    print(f"Test     - MSE: {test_mse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
    
    return test_mse, test_r2


# ----------------------------
# 4) Main
# ----------------------------
if __name__ == "__main__":
    # Load your dataset
    data_path = "/Users/alexandrebredillot/Documents/GitHub/ML-Assignement/training_dataset.csv"

    if data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path, low_memory=False)
    else:
        df = pd.read_excel(data_path, sheet_name="train")

    # List of irrelevant/redundant variables to remove
    to_drop = [
        "HHX","WTFA_A","PPSU","PSTRAT","SRVY_YR","INTV_QRT","INTV_MON","RECTYPE",
        "HHRESPSA_FLG","IMPINCFLG_A","SDMSRSOFT_A","SDMSRS_A","RSNHIMISS_A",
        "AGE65","OVER65FLG_A","MAFLG_A","CHFLG_A","INCGRP_A","RATCAT_A",
        "HISP_A","SASPPHISP_A","SASPPRACE_A","HISDETP_A",
        "MODFREQW_A","MODMIN_A","VIGFREQW_A","VIGMIN_A","MODNR_A","MODTPR_A","VIGNR_A","VIGTPR_A",
        # weight-related pregnancy
        "NOWWGTPRG_A","ADVWGTPRG_A",
        # cancer flags
        "COLRCCAN_A","HDNCKCAN_A","UTERUCAN_A","THYROCAN_A","THROACAN_A","STOMACAN_A",
        "SKNDKCAN_A","SKNNMCAN_A","SKNMCAN_A","RECTUCAN_A","PROSTCAN_A","PANCRCAN_A",
        "OVARYCAN_A","MOUTHCAN_A","MELANCAN_A","LYMPHCAN_A","LUNGCAN_A","LIVERCAN_A",
        "LEUKECAN_A","LARYNCAN_A","GALLBCAN_A","ESOPHCAN_A","COLONCAN_A","CERVICAN_A",
        "BREASCAN_A","BRAINCAN_A","BONECAN_A","BLOODCAN_A","BLADDCAN_A","OTHERCANP_A",
        # cancer age at dx
        "COLRCAGETC_A","HDNCKAGETC_A","OTHERAGETC_A","UTERUAGETC_A","THYROAGETC_A",
        "THROAAGETC_A","STOMAAGETC_A","SKNDKAGETC_A","SKNNMAGETC_A","SKNMAGETC_A",
        "RECTUAGETC_A","PROSTAGETC_A","PANCRAGETC_A","OVARYAGETC_A","MOUTHAGETC_A",
        "MELANAGETC_A","LYMPHAGETC_A","LUNGAGETC_A","LIVERAGETC_A","LEUKEAGETC_A",
        "LARYNAGETC_A","GALLBAGETC_A","ESOPHAGETC_A","COLONAGETC_A","CERVIAGETC_A",
        "BREASAGETC_A","BRAINAGETC_A","BONEAGETC_A","BLOODAGETC_A","BLADDAGETC_A",
        # insurance / coverage
        "IHS_A","MILITARY_A","CHIP_A","MEDICAID_A","MEDICARE_A","PRIVATE_A",
        "PRPLCOV1_C_A","PRPLCOV2_C_A","PLEXCHOG_A","PLEXCHOP_A","EXCHPR1_A","EXCHPR2_A",
        "VACAREEV_A","VAHOSP_A","VADISB_A","COMBAT_A","AFVETTRN_A","AFVET_A",
        # sun exposure
        "ANYSBURN_A","SUNTAN_A","SUNSCREEN_A","SUNSHIRT_A","SUNHAT_A","SUNSHADE_A","SUNSKIN_A",
        # walk environment
        "ANIMALWLK_A","CRIMEWLK_A","TRAFFICWLK_A","PEOPLEWLK_A","WEATHERWLK_A","SIDEWLK_A",
        "RELAXWLK_A","FUNWLK_A","TRANSITWLK_A","SHOPSWLK_A","ROADSWLK_A","HOMEWLK_A",
        "WLKLEISTPD_A","WLKLEISDAY_A","WLKTRANTPD_A","WLKTRANDAY_A",
        # keep only WLKLEIS_A and WLKTRAN_A (so we drop the above)
        # soc / family
        "SAPARENTSC_A","PCNTFAM_A","NUMCAN_A","NATUSBORN_A",
        # interview / weight fields already listed but ensure included
        "WTFA_A",
        # proxy / household
        "PROXYREL_A","PROXY_A","HHSTAT_A",
        # military specifics
        "HILASTMY_A","MILSPC3_A","MILSPC2_A","MILSPC1_A",
        # plan / premium / exchange fields
        "OGHDHP_A","OGDEDUC_A","OGPREM_A","OGXCHNG_A",
        "OPHDHP_A","OPDEDUC_A","OPPREM_A","OPXCHNG_A",
        "CHHDHP_A","CHDEDUC_A","CHPREM_A","CHXCHNG_A",
        "PRVSCOV2_A","PRVSCOV1_A","PRDNCOV2_A","PRDNCOV1_A",
        "PRRXCOV2_A","PRRXCOV1_A","HSAHRA2_A","HSAHRA1_A",
        "PRHDHP2_A","PRHDHP1_A","PRDEDUC2_A","PRDEDUC1_A",
        "PLN2PAY6_A","PLN2PAY5_A","PLN2PAY4_A","PLN2PAY3_A","PLN2PAY2_A","PLN2PAY1_A",
        "PLN1PAY6_A","PLN1PAY5_A","PLN1PAY4_A","PLN1PAY3_A","PLN1PAY2_A","PLN1PAY1_A",
        "PLNEXCHG2_A","PLNEXCHG1_A","PRPOLH2_A","PRPOLH1_A","PRPLCOV2_A","PRPLCOV1_A",
        "POLHLD2_A","POLHLD1_A","MAHDHP_A","MADEDUC_A","MAPREM_A","MAXCHNG_A",
        "MCPARTD_A","MCHMO_A","MCCHOICE_A","MCPART_A","SINCOVRX_A","SINCOVVS_A","SINCOVDE_A",
        "HIKIND10_A","HIKIND09_A","HIKIND08_A","HIKIND07_A","HIKIND06_A","HIKIND05_A",
        "HIKIND04_A","HIKIND03_A","HIKIND02_A","HIKIND01_A","HICOV_A",
        # social/functional
        "SOCWRKLIM_A","SOCSCLPAR_A","SOCERRNDS_A","UPPOBJCT_A","EQSTEPS_A","PERASST_A",
        "HEARINGDF_A","HEARAIDFR_A","HEARAID_A","VISIONDF_A",
        # diabetes insurance/time
        "DIBINSSTYR_A","DIBINSSTOP_A","DIBINSTIME_A","DIBINS_A",
        # asthma
        "ASJOB_A","ASPREVR_A","ASINHALE3M_A","ASER12M_A","ASAT12M_A","ASTILL_A","ASEV_A",
    ]
    df = df.drop(columns=to_drop, errors="ignore")
    print(f"Remaining variables: {df.shape[1]}")

    # Ensure target exists
    target = "WEIGHTLBTC_A"
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    # Clean target
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # Define X and y
    X = df.drop(columns=[target])
    y = df[target]

    # Split data once
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline Ridge
    ridge_pipe = build_ridge_pipeline(alpha=1000.0)
    evaluate_pipeline(ridge_pipe, X_train, X_test, y_train, y_test, name="Ridge")

    # Base GB
    gb_pipe = build_gb_pipeline()
    evaluate_pipeline(gb_pipe, X_train, X_test, y_train, y_test, name="GradientBoosting (base)")

    # Tuned GB with best parameters
    gb_tuned_pipe = build_gb_pipeline(**BEST_GB_PARAMS)
    evaluate_pipeline(gb_tuned_pipe, X_train, X_test, y_train, y_test, name="GradientBoosting (tuned params)")

    # HistGB
    histgb_pipe = build_histgb_pipeline()
    evaluate_pipeline(histgb_pipe, X_train, X_test, y_train, y_test, name="HistGradientBoosting")

    # Try aggressive parameters
    AGGRESSIVE_PARAMS = {
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "max_depth": 4,          # Deeper trees
        "min_samples_leaf": 5,   # Less regularization
        "subsample": 0.9,
    }
    
    gb_aggressive_pipe = build_gb_pipeline(**AGGRESSIVE_PARAMS)
    evaluate_pipeline(gb_aggressive_pipe, X_train, X_test, y_train, y_test, name="GradientBoosting (aggressive)")
    
    # Ensemble of best models
    from sklearn.ensemble import VotingRegressor
    
    ensemble = VotingRegressor([
        ('gb_tuned', gb_tuned_pipe),
        ('histgb', histgb_pipe)
    ])
    
    ensemble.fit(X_train, y_train)
    test_pred = ensemble.predict(X_test)
    train_pred = ensemble.predict(X_train)
    
    print(f"\n[Ensemble (GB+HistGB)]")
    print(f"Training - MSE: {mean_squared_error(y_train, train_pred):.4f}, R²: {r2_score(y_train, train_pred):.4f}, MAE: {mean_absolute_error(y_train, train_pred):.4f}")
    print(f"Test     - MSE: {mean_squared_error(y_test, test_pred):.4f}, R²: {r2_score(y_test, test_pred):.4f}, MAE: {mean_absolute_error(y_test, test_pred):.4f}")