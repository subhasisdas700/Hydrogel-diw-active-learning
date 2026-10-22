"""
Fixed selection-by-mean script with robust handling for missing rheology columns.
"""
import os
import sys
import pickle
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------- CONFIG (adjust if needed) ----------
FINAL_RESULTS_DIR = "final_results"
MODELS_DIR = os.path.join(FINAL_RESULTS_DIR, "models")

CANDIDATE_POOL_FILE = os.path.join(FINAL_RESULTS_DIR, "candidate_pool_predictions_sample.csv")
FALLBACK_SMALL_FILE = os.path.join(FINAL_RESULTS_DIR, "final_candidates_corrected.csv")
GPR_PICKLE = os.path.join(MODELS_DIR, "final_gpr.pkl")
SCALER_PICKLE = os.path.join(MODELS_DIR, "final_scaler.pkl")
RHEO_MODELS_PICKLE = os.path.join(MODELS_DIR, "rheo_models.pkl")
P_USED_PICKLE = os.path.join(MODELS_DIR, "p_used_gpr.pkl")

REQUIRE_IN_DISTRIBUTION = True
OOD_COLUMN = "is_ood"
OOD_ALLOW_IF_MISSING = True

TOP_N_BY_MEAN = 50
LOCAL_PERTURB_SAMPLES = 1000
LOCAL_PERTURB_SCALE = [0.15, 0.4, 0.5, 0.4]  # Alg, PVA, Gel, Gly std (wt%)
FINAL_K = 5
PRESSURE_MAX_MPA = 5.0
TOTAL_POLYMER_MIN = 5.0
TOTAL_POLYMER_MAX = 30.0
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# composition bounds for local search
ALG_RANGE = (0.1, 4.0)
PVA_RANGE = (1.0, 10.0)
GEL_RANGE = (2.0, 20.0)
GLY_RANGE = (1.0, 10.0)

# feature sets
FEATURE_COLS = [
    'Alginate_wt%','PVA_wt%','Gelatin_wt%','Glycerol_wt%',
    'Total_polymer_wt%','Polymer_over_Glycerol',
    '?y (Pa)','?yd (Pa)','n','K',"G' at LVR","G'' at LVR",'Loss Factor'
]
comp_feature_cols = ['Alginate_wt%','PVA_wt%','Gelatin_wt%','Glycerol_wt%','Total_polymer_wt%','Polymer_over_Glycerol']

# printing geometry for HB (from your earlier message)
R_nozzle_m = 3e-4
L_nozzle_m = 1.97e-2
w_target_m = 0.00072
h_target_m = 0.00048
v_print_m_s = 0.02

# ---------- helper utilities ----------
def load_candidates():
    if os.path.exists(CANDIDATE_POOL_FILE):
        return pd.read_csv(CANDIDATE_POOL_FILE)
    if os.path.exists(FALLBACK_SMALL_FILE):
        return pd.read_csv(FALLBACK_SMALL_FILE)
    raise FileNotFoundError("No candidate pool found. Run the final analysis script first.")

def load_models():
    if not os.path.exists(GPR_PICKLE) or not os.path.exists(SCALER_PICKLE):
        raise FileNotFoundError("Missing GPR or scaler pickles in models/ - run final analysis first.")
    with open(GPR_PICKLE,'rb') as f:
        gpr = pickle.load(f)
    with open(SCALER_PICKLE,'rb') as f:
        scaler = pickle.load(f)
    rheo_models = None
    if os.path.exists(RHEO_MODELS_PICKLE):
        with open(RHEO_MODELS_PICKLE,'rb') as f:
            rheo_models = pickle.load(f)
    p_used_model = None
    if os.path.exists(P_USED_PICKLE):
        with open(P_USED_PICKLE,'rb') as f:
            p_used_model = pickle.load(f)
    return gpr, scaler, rheo_models, p_used_model

def compute_P_HB_from_rheo_row(row):
    try:
        tau_y = float(row.get('?y (Pa)', np.nan))
        n = float(row.get('n', np.nan))
        K = float(row.get('K', np.nan))
    except Exception:
        return np.nan
    if np.isnan(tau_y) or np.isnan(n) or np.isnan(K) or n <= 0:
        return np.nan
    A = w_target_m * h_target_m
    Q = v_print_m_s * A
    gamma_dot_app = 4.0 * Q / (np.pi * (R_nozzle_m**3))
    gamma_dot_w = ((3.0 * n + 1.0) / (4.0 * n)) * gamma_dot_app
    tau_w = tau_y + K * (gamma_dot_w ** n)
    deltaP_Pa = (2.0 * L_nozzle_m / R_nozzle_m) * tau_w
    return float(deltaP_Pa / 1e6)  # MPa

def ensure_feature_columns(df, col_means):
    # ensure derived and model input columns exist; fill with col_means where needed
    if 'Total_polymer_wt%' not in df.columns:
        df['Total_polymer_wt%'] = df[['Alginate_wt%','PVA_wt%','Gelatin_wt%']].sum(axis=1)
    if 'Polymer_over_Glycerol' not in df.columns:
        df['Polymer_over_Glycerol'] = df['Total_polymer_wt%'] / (df['Glycerol_wt%'].replace(0, np.nan).fillna(1e-8))
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # fill NaNs with col_means mapping where provided
    if col_means is not None:
        df[FEATURE_COLS] = df[FEATURE_COLS].astype(float).fillna(col_means)
    else:
        df[FEATURE_COLS] = df[FEATURE_COLS].astype(float).fillna(0.0)
    return df

# ---------- main ----------
def main():
    df_cand = load_candidates()
    gpr, scaler, rheo_models, p_used_model = load_models()

    # get column means from scaler if available
    if hasattr(scaler, 'mean_'):
        col_means = dict(zip(FEATURE_COLS, scaler.mean_))
    else:
        col_means = {c:0.0 for c in FEATURE_COLS}

    # ensure features for candidate pool
    df_cand = ensure_feature_columns(df_cand, col_means)

    # predict rheology for candidate pool with trained rheo_models if available
    if rheo_models:
        for target, model in rheo_models.items():
            if model is not None and target in ['?y (Pa)','?yd (Pa)','n','K',"G' at LVR","G'' at LVR",'Loss Factor']:
                try:
                    Xcomp = df_cand[comp_feature_cols].astype(float).fillna([col_means[c] for c in comp_feature_cols])
                    df_cand[target] = model.predict(Xcomp)
                except Exception:
                    pass

    # fill any remaining rheology NaNs with safe heuristics
    if df_cand['?y (Pa)'].isna().any():
        df_cand['?y (Pa)'] = df_cand['?y (Pa)'].fillna(df_cand['Gelatin_wt%'] * 8 + df_cand['Alginate_wt%'] * 15 + df_cand['PVA_wt%'] * 3 - df_cand['Glycerol_wt%'] * 12)
        df_cand['?y (Pa)'] = df_cand['?y (Pa)'].clip(lower=0.05)
    if df_cand['?yd (Pa)'].isna().any():
        df_cand['?yd (Pa)'] = df_cand['?yd (Pa)'].fillna(df_cand['?y (Pa)'])
    if df_cand['n'].isna().any():
        df_cand['n'] = df_cand['n'].fillna(df_cand['n'].median() if not df_cand['n'].dropna().empty else 0.4)
    if df_cand['K'].isna().any():
        df_cand['K'] = df_cand['K'].fillna(df_cand['Total_polymer_wt%']*20)
    if "G' at LVR" not in df_cand.columns or df_cand["G' at LVR"].isna().any():
        df_cand["G' at LVR"] = df_cand.get("G' at LVR", df_cand['?y (Pa)']*1.2)
    if "G'' at LVR" not in df_cand.columns or df_cand["G'' at LVR"].isna().any():
        df_cand["G'' at LVR"] = df_cand.get("G'' at LVR", df_cand["G' at LVR"]*0.8)
    if df_cand['Loss Factor'].isna().any():
        df_cand['Loss Factor'] = df_cand['Loss Factor'].fillna(df_cand["G'' at LVR"]/df_cand["G' at LVR"].replace(0,np.nan).fillna(1.0))

    # compute mu/sigma if missing
    recompute = ('Predicted_Score' not in df_cand.columns) or (df_cand['Predicted_Score'].isna().any())
    if recompute:
        Xcand_feat = df_cand[FEATURE_COLS].astype(float).fillna(col_means)
        Xcand_scaled = scaler.transform(Xcand_feat)
        mu_cand, sigma_cand = gpr.predict(Xcand_scaled, return_std=True)
        df_cand['Predicted_Score'] = mu_cand
        df_cand['Predicted_Std'] = sigma_cand

    # compute Predicted_?P_used
    if p_used_model is not None:
        Xcand_feat = df_cand[FEATURE_COLS].astype(float).fillna(col_means)
        df_cand['Predicted_?P_used (MPa)'] = p_used_model.predict(scaler.transform(Xcand_feat))
    else:
        df_cand['Estimated_?P_HB (MPa)'] = df_cand.apply(compute_P_HB_from_rheo_row, axis=1)
        df_cand['Predicted_?P_used (MPa)'] = df_cand['Estimated_?P_HB (MPa)'] * 1.05

    # approximate standardized distance to training (distance to training mean in standardized space)
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        Xcand_feat = df_cand[FEATURE_COLS].astype(float).fillna(col_means)
        Xcand_std = (Xcand_feat - scaler.mean_) / (scaler.scale_ + 1e-12)
        df_cand['min_std_dist_to_train'] = np.linalg.norm(Xcand_std, axis=1)
        df_cand['is_ood'] = df_cand['min_std_dist_to_train'] > 3.0
    else:
        df_cand['min_std_dist_to_train'] = 0.0
        df_cand['is_ood'] = False

    # save top_1000 by predicted score for inspection
    df_top1000 = df_cand.sort_values('Predicted_Score', ascending=False).head(1000).copy()
    df_top1000.to_csv(os.path.join(FINAL_RESULTS_DIR,'top_1000_candidates_by_predicted_score_recomputed.csv'), index=False)
    print("Saved top_1000_candidates_by_predicted_score_recomputed.csv")

    # filter in-distribution if requested
    if REQUIRE_IN_DISTRIBUTION and (OOD_COLUMN in df_cand.columns):
        df_filtered = df_cand[df_cand[OOD_COLUMN] == False].copy()
    else:
        if REQUIRE_IN_DISTRIBUTION and (OOD_COLUMN not in df_cand.columns) and OOD_ALLOW_IF_MISSING:
            print("is_ood missing; proceeding without OOD filter.")
        df_filtered = df_cand.copy()

    # feasibility filters
    df_filtered = df_filtered[(df_filtered['Predicted_?P_used (MPa)'] <= PRESSURE_MAX_MPA) &
                              (df_filtered['Total_polymer_wt%'] >= TOTAL_POLYMER_MIN) &
                              (df_filtered['Total_polymer_wt%'] <= TOTAL_POLYMER_MAX)].copy()
    print(f"Candidates after filters: {len(df_filtered)}")

    # rank purely by mean
    df_sorted = df_filtered.sort_values('Predicted_Score', ascending=False).reset_index(drop=True)
    topN = min(TOP_N_BY_MEAN, len(df_sorted))
    top_initial = df_sorted.head(topN).copy()
    print("Top initial (by predicted mean) preview:")
    print(top_initial[['Alginate_wt%','PVA_wt%','Gelatin_wt%','Glycerol_wt%','Predicted_Score']].head(10))

    # load rheo_models (for local refinement)
    rheo_models_loaded = {}
    if os.path.exists(RHEO_MODELS_PICKLE):
        with open(RHEO_MODELS_PICKLE,'rb') as f:
            rheo_models_loaded = pickle.load(f)

    # local refinement loop
    refined_rows = []
    for idx, row in top_initial.iterrows():
        base_comp = np.array([row['Alginate_wt%'], row['PVA_wt%'], row['Gelatin_wt%'], row['Glycerol_wt%']], dtype=float)
        perturbs = np.random.normal(loc=0.0, scale=LOCAL_PERTURB_SCALE, size=(LOCAL_PERTURB_SAMPLES,4))
        lower = np.array([ALG_RANGE[0], PVA_RANGE[0], GEL_RANGE[0], GLY_RANGE[0]])
        upper = np.array([ALG_RANGE[1], PVA_RANGE[1], GEL_RANGE[1], GLY_RANGE[1]])
        local_pts = np.clip(base_comp + perturbs, lower, upper)
        df_local = pd.DataFrame(local_pts, columns=['Alginate_wt%','PVA_wt%','Gelatin_wt%','Glycerol_wt%'])
        df_local['Total_polymer_wt%'] = df_local[['Alginate_wt%','PVA_wt%','Gelatin_wt%']].sum(axis=1)
        df_local['Polymer_over_Glycerol'] = df_local['Total_polymer_wt%'] / (df_local['Glycerol_wt%'].replace(0,np.nan).fillna(1e-8))
        # predict rheo for local points using rheo_models_loaded if present
        for target, model in (rheo_models_loaded or {}).items():
            if model is not None and target in ['?y (Pa)','?yd (Pa)','n','K',"G' at LVR","G'' at LVR",'Loss Factor']:
                try:
                    Xlocal_comp = df_local[comp_feature_cols].astype(float).fillna([col_means[c] for c in comp_feature_cols])
                    df_local[target] = model.predict(Xlocal_comp)
                except Exception:
                    df_local[target] = col_means.get(target, np.nan)
        # fallback fills for any remaining NaNs
        if df_local['?y (Pa)'].isna().any():
            df_local['?y (Pa)'] = df_local['?y (Pa)'].fillna(df_local['Gelatin_wt%'] * 8 + df_local['Alginate_wt%']*15 + df_local['PVA_wt%']*3)
        if df_local['?yd (Pa)'].isna().any():
            df_local['?yd (Pa)'] = df_local['?yd (Pa)'].fillna(df_local['?y (Pa)'])
        if df_local['n'].isna().any():
            df_local['n'] = df_local['n'].fillna(df_sorted['n'].median() if not df_sorted['n'].dropna().empty else 0.4)
        if df_local['K'].isna().any():
            df_local['K'] = df_local['K'].fillna(df_local['Total_polymer_wt%']*20)
        df_local["G' at LVR"] = df_local.get("G' at LVR", df_local['?y (Pa)']*1.2)
        df_local["G'' at LVR"] = df_local.get("G'' at LVR", df_local["G' at LVR"]*0.8)
        df_local['Loss Factor'] = df_local.get('Loss Factor', df_local["G'' at LVR"]/df_local["G' at LVR"].replace(0,np.nan).fillna(1.0))

        # compute mu with GPR for local points
        Xloc = df_local[FEATURE_COLS].astype(float).fillna([col_means[c] for c in FEATURE_COLS])
        try:
            mu_loc = gpr.predict(scaler.transform(Xloc))
        except Exception:
            mu_loc = gpr.predict((Xloc - scaler.mean_)/ (scaler.scale_ + 1e-12))
        best_i = int(np.nanargmax(mu_loc))
        best_row = df_local.iloc[best_i].copy()
        best_row['Predicted_Score_local'] = float(mu_loc[best_i])
        best_row['Estimated_?P_HB (MPa)'] = compute_P_HB_from_rheo_row(best_row)
        best_row['Predicted_?P_used (MPa)'] = best_row['Estimated_?P_HB (MPa)'] * 1.05
        refined_rows.append(best_row)

    refined_df = pd.DataFrame(refined_rows)

    combined = pd.concat([top_initial.reset_index(drop=True), refined_df.reset_index(drop=True)], ignore_index=True, sort=False)
    combined['final_mu'] = combined.get('Predicted_Score_local', combined['Predicted_Score'])
    combined = combined.sort_values('final_mu', ascending=False).reset_index(drop=True)
    final_df = combined.head(min(FINAL_K, len(combined))).copy()

    out_csv = os.path.join(FINAL_RESULTS_DIR, "top_by_mu_candidates_fixed.csv")
    final_df.to_csv(out_csv, index=False)
    try:
        final_df.to_excel(os.path.join(FINAL_RESULTS_DIR, "top_by_mu_candidates_fixed.xlsx"), index=False)
    except Exception:
        pass

    print("Saved top_by_mu_candidates_fixed.csv")
    disp_cols = ['Alginate_wt%','PVA_wt%','Gelatin_wt%','Glycerol_wt%','final_mu']
    if 'Predicted_?P_used (MPa)' in final_df.columns:
        disp_cols.append('Predicted_?P_used (MPa)')
    if 'min_std_dist_to_train' in final_df.columns:
        disp_cols.append('min_std_dist_to_train')
    if 'is_ood' in final_df.columns:
        disp_cols.append('is_ood')
    print(final_df[disp_cols])

if __name__ == "__main__":
    main()
