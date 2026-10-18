"""Shared config: seeds, columns, paths. No hardcoded assumptions beyond verified schema."""
import os

SEED = 42
ID_COL = "ID"
TARGET = "is_climate_sensitive"
TARGET_F1 = "TargetF1"
TARGET_RAUC = "TargetRAUC"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
    if "__file__" in dir() else os.getcwd()

TRAIN_PATH = os.path.join(BASE_DIR, "Train.csv")
TEST_PATH = os.path.join(BASE_DIR, "Test.csv")
CLIMATE_PATH = os.path.join(BASE_DIR, "climate_features.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "SampleSubmission.csv")

# Columns verified from data_dictionary.csv + inspection
DEMOGRAPHIC_COLS = ["age", "gender", "zone"]
GEO_RAW_COLS = ["latitude", "longitude", "location"]  # location = high-cardinality string, OOD risk
GEO_NUMERIC_COLS = ["latitude", "longitude", "elevation", "slope"]
CLIMATE_BASE_COLS = ["avg_temperature", "max_temperature", "min_temperature", "precipitation"]
CLIMATE_DOWNLOADED_COLS = [
    "rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "rain_days_30d",
    "max_daily_rain_30d", "tavg_7d", "tavg_30d", "tavg_90d",
    "tmax_30d", "tmin_30d", "temp_range_mean_30d",
    "ndvi_30d", "ndvi_90d", "elevation", "slope",
    # "hot_days_30d" excluded: constant 0 in all 4176 rows -> zero variance
]
