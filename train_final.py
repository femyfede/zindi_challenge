"""Final pipeline: time-aware CV, location-specific climate anomalies,
CatBoost native, pseudo-labeling, advanced stacking. Target > 0.87.
"""
import os, sys, time, warnings, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Add current directory to path for module imports
sys.path.insert(0, ".")

# Import custom modules
from src.data_loader import load_data, merge_climate
from src.features_advanced import add_features_advanced, compute_location_climatology, ADV_NUM, ADV_CAT
from src.encoding import target_encode_oof, target_encode_full, spatial_features
from src.validation import make_preprocess, fold_metrics, combined_score
from src.submission import build_submission, verify_submission

# Create directories for experiments if they don't exist
os.makedirs("experiments", exist_ok=True)

# --- 1. Load and prepare base data ---
tr_raw, te_raw, cl = load_data()
tr = merge_climate(tr_raw, cl)
te = merge_climate(te_raw, cl)

# --- 2. Location-specific climatology (from train only) ---
location_climatology = compute_location_climatology(tr)
tr = add_features_advanced(tr, location_climatology)
te = add_features_advanced(te, location_climatology) # Apply train climatology to test

# --- 3. Spatial features (from train only, applied to test) ---
tr = spatial_features(tr, n_clusters=8, seed=42)
te = spatial_features(te, n_clusters=8, seed=42)
SPATIAL_DIST = [f"dist_cluster_{c}" for c in range(8)] + ["dist_nearest_cluster"]
SPATIAL_CAT = ["geo_cluster"]

# --- 4. Target Encoding for Location (OOF for train, full train for test) ---
# Using pd.Series(y) to ensure consistency with target_encode_oof's expectation
loc_oof, loc_glob_mean = target_encode_oof(tr, pd.Series(tr["is_climate_sensitive"]), "location", n_splits=5, smooth=30)
tr["loc_te"] = loc_oof
_, loc_te_test, _, _ = target_encode_full(tr, pd.Series(tr["is_climate_sensitive"]), te, "location", smooth=30)
te["loc_te"] = loc_te_test

# --- 5. Frequency Encoding (Location and Zone) ---
loc_freq_map = tr["location"].value_counts(normalize=True).to_dict()
tr["loc_freq"] = tr["location"].map(loc_freq_map).fillna(0)
te["loc_freq"] = te["location"].map(loc_freq_map).fillna(0)

zone_freq_map = tr["zone"].value_counts(normalize=True).to_dict()
tr["zone_freq"] = tr["zone"].map(zone_freq_map)
te["zone_freq"] = te["zone"].map(zone_freq_map)

# --- 6. Define final feature sets ---
NUM_FEATURES = ADV_NUM + SPATIAL_DIST + ["loc_te", "loc_freq", "zone_freq", "latitude", "longitude"]
CAT_FEATURES = ADV_CAT + SPATIAL_CAT + ["location"]

y = tr["is_climate_sensitive"].values # Keep y as numpy array for model training
X = tr[NUM_FEATURES + CAT_FEATURES] # X as DataFrame
Xt = te[NUM_FEATURES + CAT_FEATURES] # Xt as DataFrame

# --- 7. Time-Aware Cross-Validation (manual rolling split) ---
# Ensure `tr` is sorted by deathdate before generating splits for time-aware CV
tr_sorted = tr.sort_values("deathdate").reset_index(drop=True)
y_sorted = tr_sorted["is_climate_sensitive"].values
X_sorted = tr_sorted[NUM_FEATURES + CAT_FEATURES]

def time_aware_cv_splits(df, n_splits=5, n_initial_years=5):
    unique_years = sorted(df["death_year"].unique())
    
    if len(unique_years) < n_initial_years + 1: # Need at least initial + 1 year for first split
        print("Not enough unique years for time-aware CV, falling back to StratifiedKFold.", flush=True)
        return StratifiedKFold(n_splits, shuffle=True, random_state=42).split(df, y_sorted) # Use y_sorted here
    
    folds = []
    current_train_years_end_idx = n_initial_years - 1

    for i in range(n_splits):
        if current_train_years_end_idx >= len(unique_years) -1: # Ensure there's at least one year for test
            break 

        train_years = unique_years[:current_train_years_end_idx + 1]
        test_years = [unique_years[current_train_years_end_idx + 1]] # Test on the very next year

        train_idx = df[df["death_year"].isin(train_years)].index.tolist()
        test_idx = df[df["death_year"].isin(test_years)].index.tolist()

        if not train_idx or not test_idx:
            print(f"Skipping fold {i} due to empty train or test set.", flush=True)
            current_train_years_end_idx += 1
            continue

        folds.append((train_idx, test_idx))
        current_train_years_end_idx += 1 # Advance train window by one year for next fold
    return folds

# Adjust CatBoost parameters for native categorical handling
def mk_cat_native(seed=42):
    return CatBoostClassifier(iterations=800, depth=6, learning_rate=0.04,
                              l2_leaf_reg=3.0, auto_class_weights="Balanced",
                              random_seed=seed, verbose=False, allow_writing_files=False)

# Adjusted XGBoost with slightly more aggressive params
def mk_xgb(seed=42):
    return XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03,
                         subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
                         scale_pos_weight=1.5, n_jobs=-1, random_state=seed,
                         tree_method="hist", eval_metric="logloss")

# Random Forest
def mk_rf(seed=42):
    return RandomForestClassifier(n_estimators=1200, min_samples_leaf=2,
                                  max_features="sqrt", max_depth=20,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=seed)

models_to_train = {
    "xgb_s42": lambda: mk_xgb(42),
    "rf_s42": lambda: mk_rf(42),
    "cat_s42": lambda: mk_cat_native(42),
}

oofs = {}
model_metrics = {}

# --- 8. Model Training and OOF Prediction (Time-Aware CV) ---
print("Starting Time-Aware CV Training...", flush=True)
# `splits` will now always be a list, so no need for conditional list conversion
splits = list(time_aware_cv_splits(tr_sorted, n_splits=5, n_initial_years=5))

if not splits: # If time_aware_cv_splits returned an empty list
    print("Time-aware CV did not generate any valid splits. Falling back to StratifiedKFold.", flush=True)
    splits = StratifiedKFold(5, shuffle=True, random_state=42).split(X_sorted, y_sorted)

for name, model_fn in models_to_train.items():
    print(f"Training {name}...", flush=True)
    oof_preds = np.zeros(len(X_sorted)) # Use X_sorted length
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, y_train = X_sorted.iloc[train_idx], y_sorted[train_idx]
        X_val, y_val = X_sorted.iloc[val_idx], y_sorted[val_idx]

        # Separate preprocessing for CatBoost native
        if "cat" in name:
            # CatBoost handles categoricals natively, so only numeric imputation
            num_imputer = SimpleImputer(strategy="median")
            X_train_num_imputed = pd.DataFrame(num_imputer.fit_transform(X_train[NUM_FEATURES]),
                                               columns=NUM_FEATURES, index=X_train.index)
            X_val_num_imputed = pd.DataFrame(num_imputer.transform(X_val[NUM_FEATURES]),
                                             columns=NUM_FEATURES, index=X_val.index)
            
            X_train_processed = pd.concat([X_train_num_imputed, X_train[CAT_FEATURES].astype(str)], axis=1)
            X_val_processed = pd.concat([X_val_num_imputed, X_val[CAT_FEATURES].astype(str)], axis=1)
            
            model = model_fn()
            cat_feature_indices = [X_train_processed.columns.get_loc(c) for c in CAT_FEATURES]
            model.fit(X_train_processed, y_train, cat_features=cat_feature_indices)
            oof_preds[val_idx] = model.predict_proba(X_val_processed)[:, 1]
        else:
            # Sklearn-compatible models use ColumnTransformer
            preprocessor = make_preprocess(NUM_FEATURES, CAT_FEATURES)
            model = Pipeline([('preprocessor', preprocessor), ('classifier', model_fn())])
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        fold_metrics_data = fold_metrics(y_val, oof_preds[val_idx])
        fold_results.append(fold_metrics_data)
        print(f"  {name} Fold {fold_idx}: F1={fold_metrics_data['f1']:.4f} AUC={fold_metrics_data['auc']:.4f}", flush=True)

    overall_metrics = fold_metrics(y_sorted, oof_preds)
    overall_metrics["combined"] = combined_score(overall_metrics["f1"], overall_metrics["auc"])
    oofs[name] = oof_preds
    model_metrics[name] = overall_metrics
    print(f"Overall {name}: F1={overall_metrics['f1']:.4f} AUC={overall_metrics['auc']:.4f} combined={overall_metrics['combined']:.4f}", flush=True)

# --- 9. Simple Ensemble (Uniform Average) --- 
# For now, let's keep it simple with the best performing models from Time-Aware CV
# Re-evaluate ranking after Time-Aware CV

ranked_models = sorted(model_metrics.items(), key=lambda item: item[1]["combined"], reverse=True)
print("\nRanked Models (Time-Aware CV):", flush=True)
for name, metrics in ranked_models:
    print(f"  {name}: Combined={metrics['combined']:.4f} F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}", flush=True)

ensemble_members_oof = []
ensemble_names = []

# Greedy ensemble over top 3 models (adjust as needed based on ranking)
if len(ranked_models) >= 1: ensemble_members_oof.append(oofs[ranked_models[0][0]]); ensemble_names.append(ranked_models[0][0])
if len(ranked_models) >= 2: ensemble_members_oof.append(oofs[ranked_models[1][0]]); ensemble_names.append(ranked_models[1][0])
if len(ranked_models) >= 3: ensemble_members_oof.append(oofs[ranked_models[2][0]]); ensemble_names.append(ranked_models[2][0])

final_oof_ensemble = np.mean(ensemble_members_oof, axis=0) if ensemble_members_oof else np.zeros(len(X_sorted))
ensemble_overall_metrics = fold_metrics(y_sorted, final_oof_ensemble)
ensemble_overall_metrics["combined"] = combined_score(ensemble_overall_metrics["f1"], ensemble_overall_metrics["auc"])
print(f"\nFinal Ensemble ({'+'.join(ensemble_names)}): F1={ensemble_overall_metrics['f1']:.4f} AUC={ensemble_overall_metrics['auc']:.4f} combined={ensemble_overall_metrics['combined']:.4f}", flush=True)

# --- Pseudo-labeling (Optional, initially commented out) ---
PSEUDO_LABELING_ENABLED = True # Set to True to enable pseudo-labeling
if PSEUDO_LABELING_ENABLED:
    print("\nStarting Pseudo-labeling...", flush=True)
    # Train a "teacher" model on all available training data
    teacher_model_name = ranked_models[0][0] # Use the best model as teacher
    teacher_model_fn = models_to_train[teacher_model_name]
    
    if "cat" in teacher_model_name:
        num_imputer_teacher = SimpleImputer(strategy="median")
        X_full_num_imputed_teacher = pd.DataFrame(num_imputer_teacher.fit_transform(X_sorted[NUM_FEATURES]),
                                                  columns=NUM_FEATURES, index=X_sorted.index)
        X_full_processed_teacher = pd.concat([X_full_num_imputed_teacher, X_sorted[CAT_FEATURES].astype(str)], axis=1)
        
        Xt_num_imputed_teacher = pd.DataFrame(num_imputer_teacher.transform(Xt[NUM_FEATURES]),
                                              columns=NUM_FEATURES, index=Xt.index)
        Xt_processed_teacher = pd.concat([Xt_num_imputed_teacher, Xt[CAT_FEATURES].astype(str)], axis=1)
        
        teacher_model = teacher_model_fn()
        cat_feature_indices_teacher = [X_full_processed_teacher.columns.get_loc(c) for c in CAT_FEATURES]
        teacher_model.fit(X_full_processed_teacher, y_sorted, cat_features=cat_feature_indices_teacher)
        test_proba_teacher = teacher_model.predict_proba(Xt_processed_teacher)[:, 1]
    else:
        preprocessor_teacher = make_preprocess(NUM_FEATURES, CAT_FEATURES)
        teacher_model = Pipeline([('preprocessor', preprocessor_teacher), ('classifier', teacher_model_fn())])
        teacher_model.fit(X_sorted, y_sorted)
        test_proba_teacher = teacher_model.predict_proba(Xt)[:, 1]

    # Identify high-confidence pseudo-labels
    PSEUDO_THRESHOLD_HIGH = 0.95
    PSEUDO_THRESHOLD_LOW = 0.05
    
    high_conf_pos_idx = np.where(test_proba_teacher >= PSEUDO_THRESHOLD_HIGH)[0]
    high_conf_neg_idx = np.where(test_proba_teacher <= PSEUDO_THRESHOLD_LOW)[0]
    
    pseudo_labels_X = pd.concat([Xt.iloc[high_conf_pos_idx], Xt.iloc[high_conf_neg_idx]])
    pseudo_labels_y = np.concatenate([np.ones(len(high_conf_pos_idx)), np.zeros(len(high_conf_neg_idx))])

    print(f"Generated {len(high_conf_pos_idx)} positive and {len(high_conf_neg_idx)} negative pseudo-labels.", flush=True)
    
    # Combine original train data with pseudo-labels
    X_train_final = pd.concat([X_sorted, pseudo_labels_X]).reset_index(drop=True)
    y_train_final = np.concatenate([y_sorted, pseudo_labels_y])

    # Retrain ensemble members with pseudo-labeled data (simplified for this draft)
    final_test_probabilities = []
    for model_name in ensemble_names:
        model_fn = models_to_train[model_name]
        if "cat" in model_name:
            num_imputer_final = SimpleImputer(strategy="median")
            X_final_num_imputed = pd.DataFrame(num_imputer_final.fit_transform(X_train_final[NUM_FEATURES]),
                                                columns=NUM_FEATURES, index=X_train_final.index)
            X_final_processed = pd.concat([X_final_num_imputed, X_train_final[CAT_FEATURES].astype(str)], axis=1)

            Xt_num_imputed_final = pd.DataFrame(num_imputer_final.transform(Xt[NUM_FEATURES]),
                                                 columns=NUM_FEATURES, index=Xt.index)
            Xt_processed_final = pd.concat([Xt_num_imputed_final, Xt[CAT_FEATURES].astype(str)], axis=1)
            
            model_final = model_fn()
            cat_feature_indices_final = [X_final_processed.columns.get_loc(c) for c in CAT_FEATURES]
            model_final.fit(X_final_processed, y_train_final, cat_features=cat_feature_indices_final)
            final_test_probabilities.append(model_final.predict_proba(Xt_processed_final)[:, 1])
        else:
            preprocessor_final = make_preprocess(NUM_FEATURES, CAT_FEATURES)
            model_final = Pipeline([('preprocessor', preprocessor_final), ('classifier', model_fn())])
            model_final.fit(X_train_final, y_train_final)
            final_test_probabilities.append(model_final.predict_proba(Xt)[:, 1])
    final_test_probabilities = np.mean(final_test_probabilities, axis=0) # Average pseudo-labeled model predictions
    print("Pseudo-labeling complete.", flush=True)

else:
    # --- 10. Retrain final models on full dataset and predict test (NO PSEUDO-LABELING) --- 
    print("\nRetraining final models on full dataset and predicting test (no pseudo-labeling)...", flush=True)
    test_predictions_parts = []
    
    for model_name in ensemble_names:
        model_fn = models_to_train[model_name]
        if "cat" in model_name:
            # CatBoost native full train
            num_imputer = SimpleImputer(strategy="median")
            X_full_num_imputed = pd.DataFrame(num_imputer.fit_transform(X_sorted[NUM_FEATURES]),
                                              columns=NUM_FEATURES, index=X_sorted.index)
            X_full_processed = pd.concat([X_full_num_imputed, X_sorted[CAT_FEATURES].astype(str)], axis=1)
            
            Xt_num_imputed = pd.DataFrame(num_imputer.transform(Xt[NUM_FEATURES]),
                                          columns=NUM_FEATURES, index=Xt.index)
            Xt_processed = pd.concat([Xt_num_imputed, Xt[CAT_FEATURES].astype(str)], axis=1)
            
            model = model_fn()
            cat_feature_indices = [X_full_processed.columns.get_loc(c) for c in CAT_FEATURES]
            model.fit(X_full_processed, y_sorted, cat_features=cat_feature_indices)
            test_predictions_parts.append(model.predict_proba(Xt_processed)[:, 1])
        else:
            # Sklearn-compatible models full train
            preprocessor = make_preprocess(NUM_FEATURES, CAT_FEATURES)
            model = Pipeline([('preprocessor', preprocessor), ('classifier', model_fn())])
            model.fit(X_sorted, y_sorted)
            test_predictions_parts.append(model.predict_proba(Xt)[:, 1])
    
    final_test_probabilities = np.mean(test_predictions_parts, axis=0)

# --- 11. Generate Submission ---
submission = build_submission(te_raw["ID"], final_test_probabilities)
errors = verify_submission(submission, te_raw, sample_path="SampleSubmission.csv")
assert not errors, f"Submission verification failed: {errors}"

submission.to_csv("submission_final_push.csv", index=False)
print(f"Submission 'submission_final_push.csv' generated successfully with positive rate: {submission['TargetF1'].mean():.4f}", flush=True)

# --- 12. Save OOF predictions and final ensemble info ---
pd.DataFrame({"ID": tr_raw["ID"], "y": y_sorted, **{f"oof_{k}": v for k, v in oofs.items()}, "oof_ensemble": final_oof_ensemble}).to_csv(
    "experiments/oof_final_push.csv", index=False)

final_ensemble_info = {
    "ensemble_members": ensemble_names,
    "combined_score": ensemble_overall_metrics["combined"],
    "f1_score": ensemble_overall_metrics["f1"],
    "roc_auc_score": ensemble_overall_metrics["auc"],
    "test_pos_rate": submission['TargetF1'].mean()
}
json.dump(final_ensemble_info, open("experiments/final_ensemble_info.json", "w"), indent=4)
print("DONE with final push.", flush=True)
