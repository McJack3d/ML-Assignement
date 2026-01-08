import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('/Users/alexandrebredillot/Documents/GitHub/ML-Assignement/training_dataset.csv', low_memory=False)

# Separate target variable
y = df['WEIGHTLBTC_A']

# Drop weight-related and BMI-related columns to avoid data leakage
columns_to_drop = [
    'WEIGHTLBTC_A',  # Target variable
    'BMICAT_A',      # Derived from weight
    'BMI_A',         # If it exists - derived from weight
]

X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Remove rows with missing target
mask = y.notna()
X = X[mask]
y = y[mask]

# Select only numeric columns
numeric_columns = X.select_dtypes(include=[np.number]).columns
X_numeric = X[numeric_columns]

# Replace inf with NaN, then drop columns that are all NaN
X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
X_numeric = X_numeric.dropna(axis=1, how='all')

# Impute missing values (median is robust)
imputer = SimpleImputer(strategy="median")
X_numeric = pd.DataFrame(
    imputer.fit_transform(X_numeric),
    columns=X_numeric.columns,
    index=X.index
)

print(f"Features used: {X_numeric.shape[1]}")
print(f"Samples: {X_numeric.shape[0]}")
print(f"Non-numeric columns dropped: {X.shape[1] - X_numeric.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# Gradient Boosting with L2 regularization (alpha parameter)
print("\n[GradientBoosting (tuned, L2)]")

# Narrower grid to reduce failures/time
param_grid = {
    'n_estimators': [300, 500],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 1.0],          # stochastic GBM
    'max_features': ['sqrt', None],   # feature subsampling
    'alpha': [0.9, 0.95]  # Huber's alpha
}

# Initialize model
gb_model = GradientBoostingRegressor(random_state=42, loss='huber')

# Perform grid search
print("Performing hyperparameter tuning...")
grid_search = GridSearchCV(gb_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1, error_score='raise')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"\nBest Parameters: {grid_search.best_params_}")

# Predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Metrics
print(f"\nTraining Scores:")
print(f"  R² Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")

print(f"\nTest Scores:")
print(f"  R² Score: {r2_score(y_test, y_test_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")