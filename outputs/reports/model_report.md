# Model report

Selected model: **rf** (use_geo=False).

Primary metric: 0.60*F1 + 0.40*ROC-AUC, binary predictions at threshold 0.5 only.

## Stratified 5-fold comparison (no-geo features)

     model features  mean_F1   std_F1  mean_AUC  std_AUC  mean_combined  overall_F1  overall_AUC  overall_combined  precision   recall  train_time_s
        rf   no-geo 0.797062 0.016664  0.812152 0.017478       0.803098    0.797241     0.811075          0.802774   0.804175 0.790425          21.2
extratrees   no-geo 0.789058 0.014652  0.806465 0.015010       0.796021    0.789341     0.804810          0.795529   0.813050 0.766976          14.7
   xgboost   no-geo 0.791919 0.015202  0.798896 0.016910       0.794710    0.791988     0.798552          0.794614   0.782546 0.801661          28.0
  lightgbm   no-geo 0.790410 0.019814  0.797458 0.018824       0.793229    0.790479     0.796956          0.793070   0.802618 0.778701          23.0
    histgb   no-geo 0.780524 0.020739  0.796682 0.020006       0.786988    0.780561     0.796056          0.786759   0.801028 0.761114          93.4
    logreg   no-geo 0.767846 0.017089  0.804155 0.010846       0.782369    0.768017     0.803575          0.782240   0.831909 0.713239           1.9

## Ablation (geo + climate)

     model  strat_nogeo_combined  strat_geo_F1  strat_geo_AUC  strat_geo_combined  group_nogeo_combined  group_geo_combined  noclimate_combined
        rf              0.802774      0.796465       0.809417            0.801645              0.805248            0.805048            0.788216
extratrees              0.795529      0.784906       0.803854            0.792485              0.801800            0.801212            0.779761

## Holdout 20%% validation (rf): F1=0.7903 AUC=0.8114 combined=0.7988 prec=0.7855 rec=0.7951 cm=[[131, 89], [84, 326]]

Class distribution: train 65.1% positive / 34.9% negative; class_weight=balanced used, no resampling, no threshold tuning.

Interpretation (association, NOT causation): age dominates (infants/under-5 higher predicted probability; elderly lower); climate variables (rainfall shortage/intensity, temperature anomalies, NDVI change) add modest signal over the no-climate baseline; geographic coordinates were excluded due to train/test geographic shift and worse grouped-CV.

## Top features

                 feature  importance
         num__age_x_tmax    0.076868
         num__death_year    0.075844
                num__age    0.072016
    num__under5_x_rain30    0.050028
          num__is_under5    0.035970
       num__rain_sum_90d    0.028206
 num__rain_intensity_30d    0.025819
           num__ndvi_90d    0.024844
       num__rain_sum_30d    0.024750
    num__rain_30_90_diff    0.023301
           num__ndvi_30d    0.023186
num__temp_range_mean_30d    0.023090
         num__temp_range    0.022972
    num__max_temperature    0.022746
    num__min_temperature    0.022411
           num__tavg_30d    0.022301
        num__t_anom_7_30    0.022223
 num__max_daily_rain_30d    0.022192
        num__ndvi_change    0.022170
           num__tavg_90d    0.022112
       num__t_anom_30_90    0.022023
           num__heat_30d    0.021647
    num__avg_temperature    0.021195
           num__tmin_30d    0.020610
      num__precip_x_temp    0.020445
        num__rain_sum_7d    0.020425
            num__tavg_7d    0.020360
      num__precipitation    0.020155
      num__rain_7d_share    0.019985
           num__tmax_30d    0.019155
