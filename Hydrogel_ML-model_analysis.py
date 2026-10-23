"""
Generate a dense candidate pool with full predicted rheology features,
HB pressure, and GPR-predicted printability score. Trains on measured data
and a surrogate RF on the dense pool for SHAP output.
"""
import os
import sys
import math
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C  # Ensure C is imported
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from scipy.spatial.distance import cdist
from scipy.linalg import pinvh
from scipy.stats import chi2

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------- CONFIG ----------------
INPUT_CSV = "Input data file-1.1.csv"
OUT_DIR = "final_results"
MODELS_DIR = os.path.join(OUT_DIR, "models")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
SHAP_DIR = os.path.join(OUT_DIR, "shap")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

CANDIDATE_POOL_SIZE = 50000     # change to 50000 in production
SURROGATE_SAMPLE_SIZE = 5000   # sample from pool to train RF surrogate if desired
TOP_K_SAVE = 1000               # save top X for inspection
FINAL_K = 5

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

ADJUST_LAMBDA = 0.8   # penalty weight used in Adjusted = mu_clipped - ADJUST_LAMBDA * sigma

ALG_MIN, ALG_MAX = 0.1, 4.0
PVA_MIN, PVA_MAX = 1.0, 10.0
GEL_MIN, GEL_MAX = 2.0, 20.0
GLY_MIN, GLY_MAX = 1.0, 10.0

DISTANCE_THRESHOLD = 3.0
MAHALANOBIS_P = 0.975
SIGMA_THRESHOLD = 0.08

R_nozzle_m = 3e-4
L_nozzle_m = 1.97e-2
w_target_m = 0.00072
h_target_m = 0.00048
v_print_m_s = 0.02

# ---------------- Helpers ----------------
def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    return df

def compute_derived(df):
    eps = 1e-8
    for c in ['Alginate_wt%', 'PVA_wt%', 'Gelatin_wt%', 'Glycerol_wt%']:
        if c not in df.columns:
            df[c] = np.nan
    if 'Water_wt%' not in df.columns or df['Water_wt%'].isnull().all():
        df['Water_wt%'] = 100.0 - (df['Alginate_wt%'].fillna(0) +
                                   df['PVA_wt%'].fillna(0) +
                                   df['Gelatin_wt%'].fillna(0) +
                                   df['Glycerol_wt%'].fillna(0))
    df['Total_polymer_wt%'] = df[['Alginate_wt%', 'PVA_wt%', 'Gelatin_wt%']].sum(axis=1)
    df['Polymer_over_Glycerol'] = df['Total_polymer_wt%'] / (df['Glycerol_wt%'].replace(0, np.nan).fillna(eps))
    return df

def compute_target(df):
    if 'Total printability Score' in df.columns:
        df['Enhanced_Score'] = pd.to_numeric(df['Total printability Score'], errors='coerce')
    else:
        for c in ['Continuity score and Straightness', 'Line Uniformity', 'Width Tolerance', 'Start-end blob']:
            if c not in df.columns:
                df[c] = np.nan
        width = df['Width Tolerance'].fillna(0.0)
        width_pen = np.where(width < 0, width * 2, width)
        raw = (df['Continuity score and Straightness'].fillna(0.0) * 0.15 +
               df['Line Uniformity'].fillna(0.0) * 0.15 +
               width_pen * 0.40 +
               df['Start-end blob'].fillna(0.0) * 0.30)
        mn, mx = raw.min(), raw.max()
        df['Enhanced_Score'] = 0.5 if mx == mn else (raw - mn) / (mx - mn)
    return df

def build_feature_matrix(df):
    df = compute_derived(df)
    feature_cols = [
        'Alginate_wt%', 'PVA_wt%', 'Gelatin_wt%', 'Glycerol_wt%',
        'Total_polymer_wt%', 'Polymer_over_Glycerol',
        '?y (Pa)', '?yd (Pa)', 'n', 'K', "G' at LVR", "G'' at LVR", 'Loss Factor'
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feature_cols].astype(float).copy()
    col_means = X.mean()
    X = X.fillna(col_means)
    return X, feature_cols, col_means

# ---------------- Main flow ----------------
def main():
    # 1) Load measured data
    if not os.path.exists(INPUT_CSV):
        print("Input CSV not found:", INPUT_CSV)
        sys.exit(1)
    df = pd.read_csv(INPUT_CSV)
    if 'Sample-ID' not in df.columns:
        df.insert(0, 'Sample-ID', ['S' + str(i + 1) for i in range(len(df))])

    df = ensure_numeric(df, df.columns)
    df = compute_derived(df)
    df = compute_target(df)

    # Save cleaned data
    df.to_csv(os.path.join(OUT_DIR, 'measured_data_cleaned.csv'), index=False)

    # 2) Build feature matrix
    X, feature_cols, col_means = build_feature_matrix(df)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    y = df['Enhanced_Score'].astype(float).values

    # Train and compare GPR models (WITH and WITHOUT rheology features)
    print("\nTraining GPR models with and without rheology features...\n")

    # With Rheology Features
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gpr_with_rheo = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=6, alpha=1e-2, random_state=RANDOM_SEED)
    gpr_with_rheo.fit(X_scaled, y)
    y_pred_with_rheo = gpr_with_rheo.predict(X_scaled)
    r2_with_rheo = r2_score(y, y_pred_with_rheo)
    rmse_with_rheo = math.sqrt(mean_squared_error(y, y_pred_with_rheo))
    print(f"Train R2 (WITH rheology features): {r2_with_rheo:.4f}")
    print(f"Train RMSE (WITH rheology features): {rmse_with_rheo:.4f}")

    # Without Rheology Features
    rheo_features = ['?y (Pa)', '?yd (Pa)', 'n', 'K', "G' at LVR", "G'' at LVR", 'Loss Factor', '?P_HB (MPa)', '?P_used (MPa)']
    non_rheo_cols = [col for col in feature_cols if col not in rheo_features]
    X_no_rheo = X[non_rheo_cols].astype(float).copy()
    X_no_rheo_scaled = StandardScaler().fit_transform(X_no_rheo)

    gpr_no_rheo = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=6, alpha=1e-2, random_state=RANDOM_SEED)
    gpr_no_rheo.fit(X_no_rheo_scaled, y)
    y_pred_no_rheo = gpr_no_rheo.predict(X_no_rheo_scaled)
    r2_no_rheo = r2_score(y, y_pred_no_rheo)
    rmse_no_rheo = math.sqrt(mean_squared_error(y, y_pred_no_rheo))
    print(f"Train R2 (WITHOUT rheology features): {r2_no_rheo:.4f}")
    print(f"Train RMSE (WITHOUT rheology features): {rmse_no_rheo:.4f}")

    # Comparison Report
    print("\nComparison of Metrics:")
    print(f"With Rheology Features -> R2: {r2_with_rheo:.4f}, RMSE: {rmse_with_rheo:.4f}")
    print(f"Without Rheology Features -> R2: {r2_no_rheo:.4f}, RMSE: {rmse_no_rheo:.4f}")

if __name__ == "__main__":
    main()
