covid@winterfell:~/Desktop/Covid-19-BTP/covid_audio_btp$ ls -lh reports/tables/coswara_layout_audit.csv
ls -lh reports/tables/validation_issues.csv
ls -lh data/interim/coswara_index.csv
ls -lh data/interim/split_manifest.csv
ls -lh data/processed/metadata_clean.csv
ls -lh data/processed/audio_quality.csv
ls -lh data/processed/features_mfcc.csv
ls -lh data/outputs/metrics/ml_baseline_metrics.csv
ls -lh data/outputs/metrics/calibration_metrics.csv
ls -lh data/outputs/metrics/fusion_metrics.csv
-rw-rw-r-- 1 covid covid 3.3M Jun 11 16:24 reports/tables/coswara_layout_audit.csv
-rw-rw-r-- 1 covid covid 76 Jun 11 16:29 reports/tables/validation_issues.csv
-rw-rw-r-- 1 covid covid 14M Jun 11 16:24 data/interim/coswara_index.csv
-rw-rw-r-- 1 covid covid 238K Jun 11 16:24 data/interim/split_manifest.csv
-rw-rw-r-- 1 covid covid 15M Jun 11 16:29 data/processed/metadata_clean.csv
-rw-rw-r-- 1 covid covid 8.3M Jun 11 16:29 data/processed/audio_quality.csv
-rw-rw-r-- 1 covid covid 87M Jun 11 16:59 data/processed/features_mfcc.csv
-rw-rw-r-- 1 covid covid 2.4K Jun 11 17:02 data/outputs/metrics/ml_baseline_metrics.csv
-rw-rw-r-- 1 covid covid 2.3K Jun 11 17:02 data/outputs/metrics/calibration_metrics.csv
-rw-rw-r-- 1 covid covid 390 Jun 11 17:02 data/outputs/metrics/fusion_metrics.csv
covid@winterfell:~/Desktop/Covid-19-BTP/covid_audio_btp$ python - <<'PY'
from pathlib import Path
import pandas as pd

paths = [
    "reports/tables/coswara_layout_audit.csv",
    "reports/tables/validation_issues.csv",
    "data/interim/coswara_index.csv",
    "data/interim/split_manifest.csv",
    "data/processed/metadata_clean.csv",
    "data/processed/audio_quality.csv",
    "data/processed/features_mfcc.csv",
    "data/outputs/metrics/ml_baseline_metrics.csv",
    "data/outputs/metrics/calibration_metrics.csv",
    "data/outputs/metrics/fusion_metrics.csv",
]

for path in paths:
    p = Path(path)
    print("\n==", path, "==")
    if not p.exists():
        print("MISSING")
        continue
    df = pd.read_csv(p)
    print("shape:", df.shape)
    print(df.head(3).to_string(index=False))
PY
Command 'python' not found, did you mean:
  command 'python3' from deb python3
  command 'python' from deb python-is-python3
covid@winterfell:~/Desktop/Covid-19-BTP/covid_audio_btp$ python3 - <<'PY'
from pathlib import Path
import pandas as pd

paths = [
    "reports/tables/coswara_layout_audit.csv",
    "reports/tables/validation_issues.csv",
    "data/interim/coswara_index.csv",
    "data/interim/split_manifest.csv",
    "data/processed/metadata_clean.csv",
    "data/processed/audio_quality.csv",
    "data/processed/features_mfcc.csv",
    "data/outputs/metrics/ml_baseline_metrics.csv",
    "data/outputs/metrics/calibration_metrics.csv",
    "data/outputs/metrics/fusion_metrics.csv",
]

for path in paths:
    p = Path(path)
    print("\n==", path, "==")
    if not p.exists():
        print("MISSING")
        continue
    df = pd.read_csv(p)
    print("shape:", df.shape)
    print(df.head(3).to_string(index=False))
PY
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ModuleNotFoundError: No module named 'pandas'
covid@winterfell:~/Desktop/Covid-19-BTP/covid_audio_btp$ source .venv/bin/activate
(.venv) covid@winterfell:~/Desktop/Covid-19-BTP/covid_audio_btp$ python3 - <<'PY'
from pathlib import Path
import pandas as pd

paths = [
    "reports/tables/coswara_layout_audit.csv",
    "reports/tables/validation_issues.csv",
    "data/interim/coswara_index.csv",
    "data/interim/split_manifest.csv",
    "data/processed/metadata_clean.csv",
    "data/processed/audio_quality.csv",
    "data/processed/features_mfcc.csv",
    "data/outputs/metrics/ml_baseline_metrics.csv",
    "data/outputs/metrics/calibration_metrics.csv",
    "data/outputs/metrics/fusion_metrics.csv",
]

for path in paths:
    p = Path(path)
    print("\n==", path, "==")
    if not p.exists():
        print("MISSING")
        continue
    df = pd.read_csv(p)
    print("shape:", df.shape)
    print(df.head(3).to_string(index=False))
PY

== reports/tables/coswara_layout_audit.csv ==
shape: (27540, 7)
         relative_path suffix  depth   parent  is_audio  is_metadata  size_bytes
csv_labels_legend.json  .json      1  coswara     False         True        1615
     combined_data.csv   .csv      1  coswara     False         True      359150
 20200416/20200416.csv   .csv      2 20200416     False         True       23044

== reports/tables/validation_issues.csv ==
shape: (1, 3)
severity          check                       message
 warning unknown_labels 5688 rows have unknown labels

== data/interim/coswara_index.csv ==
shape: (24716, 18)
              participant_id     recording_id dataset modality    submodality                                                                                                                                   audio_path               label_raw label_binary recording_date  age gender country                                                                                                                                      symptoms_json                                                                                                           comorbidities_json test_status test_type  manual_quality_score manual_quality_label
0Rlzhiz6bybk77wdLjxwy7yLDhg1 rec_2b79258d6878 coswara   breath    deep_breath    /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200413/0Rlzhiz6bybk77wdLjxwy7yLDhg1/breathing-deep.wav no_resp_illness_exposed      unknown     2020-04-13 37.0   male   India {"cough": NaN, "cold": NaN, "fever": NaN, "diarrhoea": NaN, "loss_of_smell": NaN, "bd": NaN, "st": NaN, "ftg": NaN, "mp": NaN, "others_resp": NaN} {"asthma": NaN, "ht": NaN, "diabetes": NaN, "ihd": NaN, "cld": NaN, "pneumonia": NaN, "smoker": NaN, "others_preexist": NaN}         NaN       NaN                   NaN              unknown
0Rlzhiz6bybk77wdLjxwy7yLDhg1 rec_4901a19e38a7 coswara   breath shallow_breath /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200413/0Rlzhiz6bybk77wdLjxwy7yLDhg1/breathing-shallow.wav no_resp_illness_exposed      unknown     2020-04-13 37.0   male   India {"cough": NaN, "cold": NaN, "fever": NaN, "diarrhoea": NaN, "loss_of_smell": NaN, "bd": NaN, "st": NaN, "ftg": NaN, "mp": NaN, "others_resp": NaN} {"asthma": NaN, "ht": NaN, "diabetes": NaN, "ihd": NaN, "cld": NaN, "pneumonia": NaN, "smoker": NaN, "others_preexist": NaN}         NaN       NaN                   NaN              unknown
0Rlzhiz6bybk77wdLjxwy7yLDhg1 rec_d095141d5935 coswara    cough    heavy_cough       /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200413/0Rlzhiz6bybk77wdLjxwy7yLDhg1/cough-heavy.wav no_resp_illness_exposed      unknown     2020-04-13 37.0   male   India {"cough": NaN, "cold": NaN, "fever": NaN, "diarrhoea": NaN, "loss_of_smell": NaN, "bd": NaN, "st": NaN, "ftg": NaN, "mp": NaN, "others_resp": NaN} {"asthma": NaN, "ht": NaN, "diabetes": NaN, "ihd": NaN, "cld": NaN, "pneumonia": NaN, "smoker": NaN, "others_preexist": NaN}         NaN       NaN                   NaN              unknown

== data/interim/split_manifest.csv ==
shape: (2114, 10)
              participant_id dataset split label_binary  n_recordings modalities_available age_bucket gender     split_stratify_group  split_seed
P0S15NZ7DhQ2Crao3tm0m1tgDUh1 coswara train     positive             9  breath,cough,speech      30-44   male label_age;temp=label_age          42
dpA0EeRrtJUeKJjEuf7BL0AeTJZ2 coswara train     positive             9  breath,cough,speech      30-44   male label_age;temp=label_age          42
BjSaOnCo37bIz4AgpvWPiAYqwzA3 coswara train     negative             9  breath,cough,speech        <30 female label_age;temp=label_age          42

== data/processed/metadata_clean.csv ==
shape: (24716, 21)
              participant_id     recording_id dataset modality    submodality                                                                                                                                   audio_path               label_raw label_binary label_group recording_date  age gender country                                                                                                                                      symptoms_json                                                                                                           comorbidities_json  manual_quality_score manual_quality_label  split  duration_sec  sample_rate_original quality_flag
0Rlzhiz6bybk77wdLjxwy7yLDhg1 rec_2b79258d6878 coswara   breath    deep_breath    /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200413/0Rlzhiz6bybk77wdLjxwy7yLDhg1/breathing-deep.wav no_resp_illness_exposed      unknown     unknown     2020-04-13 37.0   male   India {"cough": NaN, "cold": NaN, "fever": NaN, "diarrhoea": NaN, "loss_of_smell": NaN, "bd": NaN, "st": NaN, "ftg": NaN, "mp": NaN, "others_resp": NaN} {"asthma": NaN, "ht": NaN, "diabetes": NaN, "ihd": NaN, "cld": NaN, "pneumonia": NaN, "smoker": NaN, "others_preexist": NaN}                   NaN              unknown unused     25.002688               48000.0           ok
0Rlzhiz6bybk77wdLjxwy7yLDhg1 rec_4901a19e38a7 coswara   breath shallow_breath /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200413/0Rlzhiz6bybk77wdLjxwy7yLDhg1/breathing-shallow.wav no_resp_illness_exposed      unknown     unknown     2020-04-13 37.0   male   India {"cough": NaN, "cold": NaN, "fever": NaN, "diarrhoea": NaN, "loss_of_smell": NaN, "bd": NaN, "st": NaN, "ftg": NaN, "mp": NaN, "others_resp": NaN} {"asthma": NaN, "ht": NaN, "diabetes": NaN, "ihd": NaN, "cld": NaN, "pneumonia": NaN, "smoker": NaN, "others_preexist": NaN}                   NaN              unknown unused     14.592000               48000.0           ok
0Rlzhiz6bybk77wdLjxwy7yLDhg1 rec_d095141d5935 coswara    cough    heavy_cough       /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200413/0Rlzhiz6bybk77wdLjxwy7yLDhg1/cough-heavy.wav no_resp_illness_exposed      unknown     unknown     2020-04-13 37.0   male   India {"cough": NaN, "cold": NaN, "fever": NaN, "diarrhoea": NaN, "loss_of_smell": NaN, "bd": NaN, "st": NaN, "ftg": NaN, "mp": NaN, "others_resp": NaN} {"asthma": NaN, "ht": NaN, "diabetes": NaN, "ihd": NaN, "cld": NaN, "pneumonia": NaN, "smoker": NaN, "others_preexist": NaN}                   NaN              unknown unused      5.205375               48000.0           ok

== data/processed/audio_quality.csv ==
shape: (24716, 18)
    recording_id                                                                                                                                   audio_path  duration_sec  sample_rate_original  rms_mean  rms_std  zero_crossing_rate_mean  silence_ratio  clipping_ratio  spectral_centroid_mean  spectral_flatness_mean  snr_proxy  event_start_sec  event_end_sec  event_duration_sec  active_audio_ratio quality_flag quality_reasons
rec_2b79258d6878    /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200413/0Rlzhiz6bybk77wdLjxwy7yLDhg1/breathing-deep.wav     25.002688               48000.0  0.006720 0.014747                 0.458928       0.050342        0.000000             3737.451597                0.070207  50.196503            9.190         23.905              14.715            0.588537           ok             NaN
rec_4901a19e38a7 /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200413/0Rlzhiz6bybk77wdLjxwy7yLDhg1/breathing-shallow.wav     14.592000               48000.0  0.002375 0.004756                 0.383818       0.032895        0.000000             3316.926750                0.088225  42.826541            0.000         14.112              14.112            0.967105           ok             NaN
rec_d095141d5935       /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200413/0Rlzhiz6bybk77wdLjxwy7yLDhg1/cough-heavy.wav      5.205375               48000.0  0.029118 0.061632                 0.218241       0.625003        0.000012             2144.293126                0.038413  70.840730            1.568          3.403               1.835            0.352520           ok             NaN

== data/processed/features_mfcc.csv ==
shape: (19028, 261)
    recording_id               participant_id dataset modality    submodality label_binary split  event_start_sec  event_end_sec  event_duration_sec  active_audio_ratio segmentation_method  mfcc_1_mean  mfcc_1_std  mfcc_2_mean  mfcc_2_std  mfcc_3_mean  mfcc_3_std  mfcc_4_mean  mfcc_4_std  mfcc_5_mean  mfcc_5_std  mfcc_6_mean  mfcc_6_std  mfcc_7_mean  mfcc_7_std  mfcc_8_mean  mfcc_8_std  mfcc_9_mean  mfcc_9_std  mfcc_10_mean  mfcc_10_std  mfcc_11_mean  mfcc_11_std  mfcc_12_mean  mfcc_12_std  mfcc_13_mean  mfcc_13_std  mfcc_14_mean  mfcc_14_std  mfcc_15_mean  mfcc_15_std  mfcc_16_mean  mfcc_16_std  mfcc_17_mean  mfcc_17_std  mfcc_18_mean  mfcc_18_std  mfcc_19_mean  mfcc_19_std  mfcc_20_mean  mfcc_20_std  mfcc_21_mean  mfcc_21_std  mfcc_22_mean  mfcc_22_std  mfcc_23_mean  mfcc_23_std  mfcc_24_mean  mfcc_24_std  mfcc_25_mean  mfcc_25_std  mfcc_26_mean  mfcc_26_std  mfcc_27_mean  mfcc_27_std  mfcc_28_mean  mfcc_28_std  mfcc_29_mean  mfcc_29_std  mfcc_30_mean  mfcc_30_std  mfcc_31_mean  mfcc_31_std  mfcc_32_mean  mfcc_32_std  mfcc_33_mean  mfcc_33_std  mfcc_34_mean  mfcc_34_std  mfcc_35_mean  mfcc_35_std  mfcc_36_mean  mfcc_36_std  mfcc_37_mean  mfcc_37_std  mfcc_38_mean  mfcc_38_std  mfcc_39_mean  mfcc_39_std  mfcc_40_mean  mfcc_40_std  delta_mfcc_1_mean  delta_mfcc_1_std  delta_mfcc_2_mean  delta_mfcc_2_std  delta_mfcc_3_mean  delta_mfcc_3_std  delta_mfcc_4_mean  delta_mfcc_4_std  delta_mfcc_5_mean  delta_mfcc_5_std  delta_mfcc_6_mean  delta_mfcc_6_std  delta_mfcc_7_mean  delta_mfcc_7_std  delta_mfcc_8_mean  delta_mfcc_8_std  delta_mfcc_9_mean  delta_mfcc_9_std  delta_mfcc_10_mean  delta_mfcc_10_std  delta_mfcc_11_mean  delta_mfcc_11_std  delta_mfcc_12_mean  delta_mfcc_12_std  delta_mfcc_13_mean  delta_mfcc_13_std  delta_mfcc_14_mean  delta_mfcc_14_std  delta_mfcc_15_mean  delta_mfcc_15_std  delta_mfcc_16_mean  delta_mfcc_16_std  delta_mfcc_17_mean  delta_mfcc_17_std  delta_mfcc_18_mean  delta_mfcc_18_std  delta_mfcc_19_mean  delta_mfcc_19_std  delta_mfcc_20_mean  delta_mfcc_20_std  delta_mfcc_21_mean  delta_mfcc_21_std  delta_mfcc_22_mean  delta_mfcc_22_std  delta_mfcc_23_mean  delta_mfcc_23_std  delta_mfcc_24_mean  delta_mfcc_24_std  delta_mfcc_25_mean  delta_mfcc_25_std  delta_mfcc_26_mean  delta_mfcc_26_std  delta_mfcc_27_mean  delta_mfcc_27_std  delta_mfcc_28_mean  delta_mfcc_28_std  delta_mfcc_29_mean  delta_mfcc_29_std  delta_mfcc_30_mean  delta_mfcc_30_std  delta_mfcc_31_mean  delta_mfcc_31_std  delta_mfcc_32_mean  delta_mfcc_32_std  delta_mfcc_33_mean  delta_mfcc_33_std  delta_mfcc_34_mean  delta_mfcc_34_std  delta_mfcc_35_mean  delta_mfcc_35_std  delta_mfcc_36_mean  delta_mfcc_36_std  delta_mfcc_37_mean  delta_mfcc_37_std  delta_mfcc_38_mean  delta_mfcc_38_std  delta_mfcc_39_mean  delta_mfcc_39_std  delta_mfcc_40_mean  delta_mfcc_40_std  delta2_mfcc_1_mean  delta2_mfcc_1_std  delta2_mfcc_2_mean  delta2_mfcc_2_std  delta2_mfcc_3_mean  delta2_mfcc_3_std  delta2_mfcc_4_mean  delta2_mfcc_4_std  delta2_mfcc_5_mean  delta2_mfcc_5_std  delta2_mfcc_6_mean  delta2_mfcc_6_std  delta2_mfcc_7_mean  delta2_mfcc_7_std  delta2_mfcc_8_mean  delta2_mfcc_8_std  delta2_mfcc_9_mean  delta2_mfcc_9_std  delta2_mfcc_10_mean  delta2_mfcc_10_std  delta2_mfcc_11_mean  delta2_mfcc_11_std  delta2_mfcc_12_mean  delta2_mfcc_12_std  delta2_mfcc_13_mean  delta2_mfcc_13_std  delta2_mfcc_14_mean  delta2_mfcc_14_std  delta2_mfcc_15_mean  delta2_mfcc_15_std  delta2_mfcc_16_mean  delta2_mfcc_16_std  delta2_mfcc_17_mean  delta2_mfcc_17_std  delta2_mfcc_18_mean  delta2_mfcc_18_std  delta2_mfcc_19_mean  delta2_mfcc_19_std  delta2_mfcc_20_mean  delta2_mfcc_20_std  delta2_mfcc_21_mean  delta2_mfcc_21_std  delta2_mfcc_22_mean  delta2_mfcc_22_std  delta2_mfcc_23_mean  delta2_mfcc_23_std  delta2_mfcc_24_mean  delta2_mfcc_24_std  delta2_mfcc_25_mean  delta2_mfcc_25_std  delta2_mfcc_26_mean  delta2_mfcc_26_std  delta2_mfcc_27_mean  delta2_mfcc_27_std  delta2_mfcc_28_mean  delta2_mfcc_28_std  delta2_mfcc_29_mean  delta2_mfcc_29_std  delta2_mfcc_30_mean  delta2_mfcc_30_std  delta2_mfcc_31_mean  delta2_mfcc_31_std  delta2_mfcc_32_mean  delta2_mfcc_32_std  delta2_mfcc_33_mean  delta2_mfcc_33_std  delta2_mfcc_34_mean  delta2_mfcc_34_std  delta2_mfcc_35_mean  delta2_mfcc_35_std  delta2_mfcc_36_mean  delta2_mfcc_36_std  delta2_mfcc_37_mean  delta2_mfcc_37_std  delta2_mfcc_38_mean  delta2_mfcc_38_std  delta2_mfcc_39_mean  delta2_mfcc_39_std  delta2_mfcc_40_mean  delta2_mfcc_40_std  rms_mean  rms_std  zcr_mean  zcr_std  spectral_centroid_mean  spectral_bandwidth_mean  spectral_rolloff_mean  spectral_flatness_mean  duration_sec
rec_5193930d4bf2 0zexHIcM7tQDdnFiEj2Eb0v3g212 coswara   breath    deep_breath     negative train            0.800         25.685              24.885            0.956134   rms_active_region  -518.586670  163.249161    62.352905   86.951080    -1.191671   43.950371    -1.231120   23.975090     2.620152    8.074373    -6.785753   14.279822    -8.475062   13.854881    -6.929000   10.545823    -8.266113   11.799720     -6.158070     8.905217     -8.016822    10.809397     -6.886836     9.313065     -8.547222    10.234943     -7.311082     8.098156     -8.101422     9.121155     -5.956508     6.598931     -6.472458     6.973149     -5.807528     6.793514     -5.496536     6.103499     -4.426714     5.183304     -3.829468     4.105313     -3.684022     4.326916     -3.287491     4.435203     -2.381766     4.007702     -2.040902     3.337330     -1.876280     3.082766     -1.154901     2.611444     -0.888190     2.568118     -0.580803     2.056055     -0.430040     2.605823     -0.098631     2.527875      0.161722     2.051431      0.425218     1.813863      0.690133     1.798608      0.678705     1.949342      0.644703     1.853110      0.631741     1.934471      0.783354     2.073142      0.632233     1.959341      0.547894     1.452543           1.417309         16.489651          -0.542832          8.761892           0.130405          5.906139           0.101397          3.380542          -0.106597          1.562679           0.148138          1.729652          -0.057321          1.785461           0.126761          1.559194           0.040298          1.381880            0.026921           1.266670           -0.012673           1.422714            0.027192           1.251534            0.015945           1.381953            0.060281           1.103034           -0.035230           1.211887            0.025224           1.097952            0.017939           0.970611            0.116298           1.142582           -0.002249           0.891407            0.041263           0.955962            0.002554           0.551490            0.107246           0.538423            0.035612           0.572031            0.058280           0.583143            0.011619           0.549541            0.044139           0.437552            0.033501           0.409352            0.027980           0.470359           -0.002213           0.316972            0.081432           0.331437            0.055097           0.369508            0.045210           0.365743            0.026836           0.276642            0.045757           0.381769            0.019524           0.365668            0.018174           0.257891           -0.014511           0.322125            0.008088           0.348417            0.001792           0.327890           -0.015774           0.244760            0.116056           5.014722           -0.194056           4.243813            0.042576           3.102921            0.009078           1.730755           -0.086303           1.057332            0.017389           1.178409            0.016062           1.199144           -0.002572           0.913745            0.062991           0.713182            -0.013701            0.755086             0.053203            0.721936             0.010228            0.860644             0.049762            0.724793            -0.032279            0.713215             0.049063            0.689541            -0.006538            0.749169             0.007187            0.625866             0.031191            0.697117             0.053934            0.608031            -0.003979            0.537988             0.024586            0.354082            -0.004774            0.327190             0.013447            0.335033            -0.023890            0.465999             0.009527            0.345929            -0.001654            0.293117             0.021660            0.304450            -0.035876            0.389306             0.000254            0.219055             0.004147            0.322681             0.003881            0.252711             0.009720            0.255741             0.016086            0.207535            -0.012641            0.253675             0.015849            0.255579            -0.003218            0.191416            -0.031904            0.262626            -0.032407            0.325068            -0.023305            0.288816            -0.020128            0.210170  0.021330 0.051081  0.413254 0.281983             1504.132489              1429.700405            3010.201035                0.074137           5.0
rec_4c5cc6315aaf 0zexHIcM7tQDdnFiEj2Eb0v3g212 coswara   breath shallow_breath     negative train            0.672          5.568               4.896            0.775335   rms_active_region  -390.161133  250.415039    82.728714   67.673912   -33.169781   61.506264   -11.574691   26.575333    -4.897049   14.580311   -13.084115   16.712162   -16.197577   15.992892   -11.013606   10.083979   -18.219021   15.777676     -7.960275     8.324125    -18.851351    15.894089     -7.327313     7.817947    -15.163736    13.320446     -5.905865     6.522887    -12.139471    10.682844     -5.763358     6.153618    -10.652884     9.097119     -5.721479     6.215651     -8.862149     7.540847     -5.126774     5.274529     -7.147110     6.077211     -4.111594     4.348607     -5.036545     5.043262     -2.951146     3.894444     -3.265318     3.764595     -1.684503     3.610791     -2.265487     3.311306     -1.005945     3.123784     -1.219059     2.740964      0.024842     2.814257     -0.318368     2.746101      0.642012     2.896996      0.147288     2.251898      1.103646     2.212458      0.676001     2.146554      1.658808     2.447381      0.789690     2.436689      1.216129     2.688303      0.582542     2.169895      1.489185     2.599799          -0.018579         50.258698           0.062440         13.606400           0.260889         10.958117           0.248289          4.416456           0.199069          2.652869           0.103831          3.217187           0.111061          3.040380           0.061998          1.868296           0.012122          3.306583            0.000178           1.539945           -0.040137           3.186041           -0.010949           1.502046           -0.073281           2.699828           -0.107379           1.240458           -0.062243           2.199353           -0.022876           1.185575           -0.039848           1.820821           -0.080862           1.165136           -0.096367           1.529464           -0.051838           0.972870           -0.035209           1.167482           -0.051287           0.752072           -0.047212           0.947018           -0.042060           0.633194           -0.023847           0.705769            0.005043           0.612264            0.010118           0.569826            0.005920           0.492976            0.015793           0.492968           -0.006241           0.425188           -0.032007           0.461767           -0.015376           0.483333           -0.004198           0.385189            0.012477           0.364588            0.005286           0.342809            0.003950           0.327563           -0.007979           0.443847           -0.007203           0.561898           -0.012834           0.378282           -0.022700           0.536403           -2.388384          16.998297           -0.878146           8.296353            0.689938           4.572376            0.266714           2.102531            0.143555           1.441320            0.201989           1.637600            0.242055           1.429092            0.072022           1.128227            0.163163           1.425474             0.069752            0.979228             0.162815            1.435302             0.095087            1.063803             0.135526            1.246292             0.053368            0.722420             0.085943            0.931900             0.040821            0.800942             0.071437            0.792603             0.019776            0.652634             0.038512            0.690261             0.013695            0.619325             0.064259            0.558290             0.057167            0.604036             0.036320            0.549578             0.011305            0.511917             0.026286            0.400866             0.031592            0.488615             0.014064            0.376394             0.002203            0.413140            -0.012000            0.335950             0.010860            0.330891             0.036962            0.439581             0.025413            0.409676            -0.017772            0.331573            -0.013863            0.261050            -0.005969            0.234815            -0.003956            0.231407            -0.009129            0.347322             0.002379            0.382840             0.007182            0.295237            -0.008201            0.315215  0.065996 0.099947  0.332712 0.260871             1393.269741              1214.443000            2444.118232                0.061955           5.0
rec_b863418edc23 0zexHIcM7tQDdnFiEj2Eb0v3g212 coswara    cough    heavy_cough     negative train            0.320          5.664               5.344            0.869792     rms_event_cough  -329.505035  254.365372    49.144581   50.211918   -21.441397   33.348579    -4.517945   16.891481    -2.709720   14.811897    -3.565845    9.903111   -15.375985   15.486049    -9.951588   10.402489   -11.395367   12.104491     -5.193020     8.271523    -10.761534    11.752549     -5.022573     5.990168     -8.026153    10.724300     -3.170787     7.796874     -7.640127    10.130614     -3.459457     7.043159     -6.847294     6.654620     -3.611018     6.463013     -4.512019     7.503275     -3.049652     6.470440     -4.582008     5.851743     -0.519716     4.091553     -3.100835     5.240199     -0.239485     4.301455     -1.493386     4.958275     -1.282494     3.309509     -1.551049     4.238592      0.796174     3.419361     -0.778858     2.834122      0.796342     4.203721     -0.016271     2.862658      1.043189     2.917748     -0.307885     2.733754      1.243249     2.910816      0.260691     3.522874      1.197271     2.761096      0.611419     2.993478      1.343945     2.891246      0.399508     2.484824      1.863140     2.929458          -0.704850         36.871864           0.455880          9.019701          -0.091896          5.856516          -0.047590          3.517831           0.143746          2.838166          -0.045069          1.932615           0.018871          2.314138          -0.155837          1.942899          -0.111800          1.765275           -0.063211           1.532676           -0.025844           1.677059            0.019741           1.084205            0.089745           1.660708            0.106813           1.406510            0.125545           1.754924            0.041089           1.376602            0.016443           1.240980           -0.053855           1.272066            0.019772           1.143430           -0.063974           1.255093           -0.069047           1.009604           -0.106336           0.711347           -0.099440           0.782289           -0.081088           0.844558            0.012144           0.794437            0.008783           0.682439            0.059777           0.793929           -0.026636           0.690784            0.011539           0.444272           -0.001140           0.722903            0.074235           0.577201            0.018619           0.494645           -0.011458           0.460557           -0.027392           0.475366            0.057718           0.764791           -0.002940           0.458541            0.018789           0.505416            0.007966           0.460078           -0.053861           0.465957           -0.030022           0.409634           -0.804580          14.670297           -0.002737           4.544366            0.022433           3.220901            0.010610           1.910441            0.120840           1.617834            0.014276           1.300390            0.029618           1.351734           -0.038219           1.015678           -0.028327           0.898182             0.034175            0.916264             0.111036            1.220413            -0.021351            0.752902             0.104239            0.904300             0.064402            0.684398             0.111569            0.887354             0.071896            0.636774             0.105762            0.744173             0.031180            0.725008             0.084679            0.699404             0.017663            0.681728             0.007761            0.592667            -0.031056            0.515239             0.008509            0.460195            -0.022655            0.525029             0.043140            0.442785            -0.020793            0.518379             0.030856            0.425894            -0.003195            0.460124             0.028327            0.331445             0.033035            0.440510             0.016909            0.365526            -0.023949            0.370054            -0.015796            0.290335            -0.017300            0.350892             0.022249            0.506114            -0.054237            0.363885             0.020105            0.419636             0.044898            0.401419             0.015056            0.332422            -0.040406            0.379866  0.094481 0.139286  0.268570 0.180773             1705.116652              1435.098677            3060.081845                0.087556           4.0

== data/outputs/metrics/ml_baseline_metrics.csv ==
shape: (12, 13)
   auroc    auprc  balanced_accuracy       f1  sensitivity  specificity    brier      ece      nll  threshold  n_samples          model_name modality
0.500000 0.323899           0.500000 0.000000     0.000000     1.000000 0.323899 0.323899 4.474836        0.5      636.0 dummy_most_frequent    cough
0.479454 0.315672           0.479454 0.296117     0.296117     0.662791 0.455975 0.455975 6.299526        0.5      636.0    dummy_stratified    cough
0.795947 0.672562           0.739512 0.647773     0.776699     0.702326 0.190426 0.161473 0.571753        0.5      636.0 logistic_regression    cough

== data/outputs/metrics/calibration_metrics.csv ==
shape: (12, 14)
 auroc    auprc  balanced_accuracy  f1  sensitivity  specificity    brier      ece      nll  threshold  n_samples          model_name modality calibration_method
   0.5 0.323899                0.5 0.0          0.0          1.0 0.218993 0.002134 0.629784        0.5      636.0 dummy_most_frequent   breath              platt
   0.5 0.323899                0.5 0.0          0.0          1.0 0.218993 0.002134 0.629784        0.5      636.0 dummy_most_frequent    cough              platt
   0.5 0.323899                0.5 0.0          0.0          1.0 0.218993 0.002134 0.629784        0.5     1590.0 dummy_most_frequent   speech              platt

== data/outputs/metrics/fusion_metrics.csv ==
shape: (2, 12)
   auroc    auprc  balanced_accuracy  f1  sensitivity  specificity    brier      ece      nll  threshold  n_samples             fusion_method
0.879070 0.837577                0.5 0.0          0.0          1.0 0.189724 0.149166 0.564733        0.5      318.0              uniform_mean
0.880515 0.842599                0.5 0.0          0.0          1.0 0.189448 0.143521 0.564107        0.5      318.0 validation_weighted_auprc
(.venv) covid@winterfell:~/Desktop/Covid-19-BTP/covid_audio_btp$ 
