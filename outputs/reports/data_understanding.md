# Data understanding report


Train rows=3146, Test rows=1030, Climate rows=4176.

ID col=ID, Target col=is_climate_sensitive.

Target distribution: {1: 2047, 0: 1099} -> positive rate 0.6507

Missing values: none in Train/Test/climate (verified isnull().sum()==0).

Duplicate IDs: train=0, test=0, climate=0.

ID overlap train/test: 0 (none).

Climate coverage: train 3146/3146, test 1030/1030; deathdate consistent between base and climate tables (0 mismatches).

Locations: train nuniq=39, test nuniq=11, overlap=1 (only 'Nakigo I, Iganga, Uganda').

GEOGRAPHIC SHIFT: test contains far-OOD sites (Kalangala island, Mitooma, Sembabule, Kamuli) outside the train Iganga cluster; lat/lon ranges differ. Consequence: raw location must NOT be used; lat/lon/elevation/slope treated as risky and decided by grouped (leave-location-out) validation.

Zone shift: train Peri_urban 37.3% vs test 23.8%. Gender balanced in both.

Age signal (train): P(target|age<=1)=~0.82, age 1-5 =~0.91, 60+ =~0.30. Age is the dominant predictor.

hot_days_30d is constant 0 across all 4176 climate rows -> dropped (zero variance).

Climate correlations with target are weak (|r|<0.06); NDVI positive, rainfall negative-small. Climate adds modest but real signal on top of demographics (tested by ablation).

Leakage checks: no target leakage (all climate windows end BEFORE deathdate: 7/30/90d lookback); no ID overlap; no duplicate rows; preprocessing fitted inside each CV fold via Pipeline; no test information used in training; no future data (all features are backward-looking).

External data decision: NO new external download. Provided climate_features.csv already joins CHIRPS (rain), ERA5-Land (temp), MODIS MOD13Q1 (NDVI), SRTM (elevation/slope) with verified ID+deathdate alignment. Re-downloading risks misalignment/leakage for no expected gain. Sources documented from downloaded_climate_features_data_dictionary.csv.