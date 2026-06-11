In the notebook : RUN_EVERYTHING_PUBLICATION
following errors and results came : 

# 1 . 

Project root: /home/covid/Desktop/Covid-19-BTP/covid_audio_btp
Python: /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python
Started: 2026-06-11T16:24:08

# 2

Coswara: /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara
COUGHVID: None

# 3

Runner ready

# 4

path 	exists 	size_bytes
0 	data/interim/coswara_index.csv 	False 	0
1 	data/processed/metadata_clean.csv 	False 	0
2 	data/interim/modality_availability.csv 	False 	0
3 	data/interim/split_manifest.csv 	False 	0
4 	data/processed/audio_quality.csv 	False 	0
5 	data/processed/features_mfcc.csv 	False 	0
6 	data/processed/spectrogram_index.csv 	False 	0
7 	data/outputs/metrics/ml_baseline_metrics.csv 	False 	0
8 	data/outputs/metrics/calibrated_branch_predict... 	False 	0
9 	data/outputs/metrics/fusion_metrics.csv 	False 	0
10 	data/outputs/metrics/quality_weighted_fusion_m... 	False 	0
11 	reports/tables/paper_metric_table.csv 	False 	0
12 	data/outputs/metrics/paired_model_comparison.csv 	False 	0
13 	reports/tables/confounding_balance.csv 	False 	0
14 	reports/tables/feature_shift_report.csv 	False 	0
15 	reports/experiment_manifest.json 	False 	0


# 5


## local preflight
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/00_local_preflight.py --coswara-dir /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara
Project root: /home/covid/Desktop/Covid-19-BTP/covid_audio_btp
Python: /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python
Coswara path: /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara
Notebook syntax: OK
Python syntax: OK
Required imports: OK
Coswara audio files discovered: 24716
Coswara CSV files discovered: 73

WARNINGS
- xgboost (xgboost): No module named 'xgboost'
- torch (torch): No module named 'torch'
- torchaudio (torchaudio): No module named 'torchaudio'
- pytest (pytest): No module named 'pytest'
- streamlit (streamlit): No module named 'streamlit'

Preflight passed. It is safe to start the notebook pipeline.


## environment check
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/00_check_environment.py
Python: 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]
OK required covid_audio_btp: 0.1.0
OK required numpy: 2.4.6
OK required pandas: 3.0.3
OK required scipy: 1.17.1
OK required librosa: 0.11.0
OK required soundfile: 0.14.0
OK required sklearn: 1.9.0
OK required matplotlib: 3.10.9
OK required seaborn: 0.13.2
OK required joblib: 1.5.3
OK required tqdm: 4.68.2
WARN optional xgboost unavailable: No module named 'xgboost'
WARN optional torch unavailable: No module named 'torch'
WARN optional torchaudio unavailable: No module named 'torchaudio'
WARN optional streamlit unavailable: No module named 'streamlit'
WARN optional pytest unavailable: No module named 'pytest'
OK optional jupyterlab: 4.5.8
OK optional ipykernel: 7.3.0
Environment check passed


## raw layout audit
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/00_inspect_dataset_layout.py --raw-dir /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara --output reports/tables/coswara_layout_audit.csv
Wrote layout audit: reports/tables/coswara_layout_audit.csv
Audio files sampled/listed: 24716
Metadata-like files sampled/listed: 2824
  suffix  is_audio  is_metadata      n
0   .csv     False         True     73
1  .json     False         True   2751
2   .wav      True        False  24716
Example files:
         relative_path suffix  depth   parent  is_audio  is_metadata  size_bytes
csv_labels_legend.json  .json      1  coswara     False         True        1615
     combined_data.csv   .csv      1  coswara     False         True      359150
 20200416/20200416.csv   .csv      2 20200416     False         True       23044
 20200901/20200901.csv   .csv      2 20200901     False         True        2777
 20200417/20200417.csv   .csv      2 20200417     False         True       19829
 20200413/20200413.csv   .csv      2 20200413     False         True        7866
 20200505/20200505.csv   .csv      2 20200505     False         True        1754
 20210419/20210419.csv   .csv      2 20210419     False         True        4539
 20201031/20201031.csv   .csv      2 20201031     False         True        3447
 20210426/20210426.csv   .csv      2 20210426     False         True        5250
 20200419/20200419.csv   .csv      2 20200419     False         True        3854
 20200919/20200919.csv   .csv      2 20200919     False         True        3729
 20200824/20200824.csv   .csv      2 20200824     False         True        2339
 20200803/20200803.csv   .csv      2 20200803     False         True        3160
 20200814/20200814.csv   .csv      2 20200814     False         True        9401
 20200424/20200424.csv   .csv      2 20200424     False         True        2828
 20201221/20201221.csv   .csv      2 20201221     False         True        3211
 20210914/20210914.csv   .csv      2 20210914     False         True        6187
 20200720/20200720.csv   .csv      2 20200720     False         True        2500
 20201130/20201130.csv   .csv      2 20201130     False         True        2124


## build Coswara index
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/01_build_coswara_index.py --raw-dir /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara --output data/interim/coswara_index.csv
Wrote 24716 rows to data/interim/coswara_index.csv
modality  submodality    
breath    deep_breath        2748
cough     heavy_cough        2747
speech    vowel_e            2747
          counting_fast      2746
breath    shallow_breath     2746
speech    vowel_a            2746
          counting_normal    2746
cough     shallow_cough      2745
speech    vowel_o            2745
dtype: int64


## clean metadata
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/02_clean_metadata.py --index data/interim/coswara_index.csv --output data/processed/metadata_clean.csv --availability-output data/interim/modality_availability.csv --audit-output reports/tables/dataset_audit.csv
Wrote metadata: data/processed/metadata_clean.csv (24716 rows)
Wrote availability: data/interim/modality_availability.csv (2746 participants)
Wrote audit: reports/tables/dataset_audit.csv
label_binary
negative    12897
positive     6131
unknown      5688
Name: count, dtype: int64


## participant splits
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/03_create_splits.py --metadata data/processed/metadata_clean.csv --output data/interim/split_manifest.csv --metadata-output data/processed/metadata_clean.csv --audit-output reports/tables/split_audit.csv
Wrote split manifest: data/interim/split_manifest.csv (2114 participants)
Updated metadata with split column: data/processed/metadata_clean.csv
Wrote split audit: reports/tables/split_audit.csv
split
train         1479
test           318
validation     317
Name: count, dtype: int64


## quality audit
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/04_quality_audit.py --metadata data/processed/metadata_clean.csv --output data/processed/audio_quality.csv --metadata-output data/processed/metadata_clean.csv --summary-output reports/tables/quality_summary.csv
Wrote quality audit: data/processed/audio_quality.csv (24716 rows)
Updated metadata: data/processed/metadata_clean.csv
Wrote quality summary: reports/tables/quality_summary.csv
quality_flag
ok                23539
mostly_silence      597
corrupt             394
short               186
Name: count, dtype: int64

/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/spectrum.py:266: UserWarning: n_fft=2048 is too large for input signal of length=1366
  warnings.warn(
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/src/covid_audio_btp/audio_io.py:61: UserWarning: PySoundFile failed. Trying audioread instead.
  y, original_sr = librosa.load(resolved_path, sr=None, mono=True)
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
	Deprecated as of librosa version 0.10.0.
	It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/src/covid_audio_btp/audio_io.py:61: UserWarning: PySoundFile failed. Trying audioread instead.
  y, original_sr = librosa.load(resolved_path, sr=None, mono=True)
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
	Deprecated as of librosa version 0.10.0.
	It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/src/covid_audio_btp/audio_io.py:61: UserWarning: PySoundFile failed. Trying audioread instead.
  y, original_sr = librosa.load(resolved_path, sr=None, mono=True)
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
	Deprecated as of librosa version 0.10.0.
	It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/src/covid_audio_btp/audio_io.py:61: UserWarning: PySoundFile failed. Trying audioread instead.
  y, original_sr = librosa.load(resolved_path, sr=None, mono=True)
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
	Deprecated as of librosa version 0.10.0.
	It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/spectrum.py:266: UserWarning: n_fft=2048 is too large for input signal of length=1366
  warnings.warn(
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/spectrum.py:266: UserWarning: n_fft=2048 is too large for input signal of length=1487
  warnings.warn(


## artifact validation
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/12_validate_artifacts.py --strict
severity          check                       message
 warning unknown_labels 5688 rows have unknown labels

True

# 6

reports/tables/validation_issues.csv: 1 rows x 3 columns
label gate passed: both positive and negative classes are present
leakage gate passed: 2746 participants appear in one split each
validation gate passed with warnings only
quality ok rate: 0.952


# 7

## feature and spectrogram extraction
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/05_extract_features.py --metadata data/processed/metadata_clean.csv --features-output data/processed/features_mfcc.csv --spectrogram-dir data/processed/spectrograms --spectrogram-index-output data/processed/spectrogram_index.csv --quality-mode all_samples
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/src/covid_audio_btp/audio_io.py:61: UserWarning: PySoundFile failed. Trying audioread instead.
  y, original_sr = librosa.load(resolved_path, sr=None, mono=True)
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
    Deprecated as of librosa version 0.10.0.
    It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
Traceback (most recent call last):
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/src/covid_audio_btp/audio_io.py", line 56, in load_audio
    y, original_sr = sf.read(resolved_path, always_2d=False)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/soundfile.py", line 313, in read
    with SoundFile(file, 'r', samplerate, channels,
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/soundfile.py", line 708, in __init__
    self._file = self._open(file, mode_int, closefd)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/soundfile.py", line 1296, in _open
    raise LibsndfileError(err, prefix=f"Error opening {self.name!r}: ")
soundfile.LibsndfileError: Error opening '/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200820/5q8gLM0yCrgGCT8F9fWlH4ycl1D3/._cough-heavy.wav': Format not recognised.
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/audio.py", line 176, in load
    y, sr_native = __soundfile_load(path, offset, duration, dtype)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/audio.py", line 209, in __soundfile_load
    context = sf.SoundFile(path)
              ^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/soundfile.py", line 708, in __init__
    self._file = self._open(file, mode_int, closefd)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/soundfile.py", line 1296, in _open
    raise LibsndfileError(err, prefix=f"Error opening {self.name!r}: ")
soundfile.LibsndfileError: Error opening '/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/data/raw/coswara/Extracted_data/20200820/5q8gLM0yCrgGCT8F9fWlH4ycl1D3/._cough-heavy.wav': Format not recognised.
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/scripts/05_extract_features.py", line 40, in <module>
    main()
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/scripts/05_extract_features.py", line 27, in main
    features = extract_feature_table(metadata, quality_mode=args.quality_mode)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/src/covid_audio_btp/features.py", line 73, in extract_feature_table
    rows = [extract_features_for_row(row) for _, row in df.iterrows()]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/src/covid_audio_btp/features.py", line 50, in extract_features_for_row
    y, sample_rate, _ = load_audio(Path(row["audio_path"]))
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/src/covid_audio_btp/audio_io.py", line 61, in load_audio
    y, original_sr = librosa.load(resolved_path, sr=None, mono=True)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/audio.py", line 184, in load
    y, sr_native = __audioread_load(path, offset, duration, dtype)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/decorator/__init__.py", line 247, in fun
    return caller(func, *(extras + args), **kw)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/util/decorators.py", line 63, in __wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/librosa/core/audio.py", line 240, in __audioread_load
    reader = audioread.audio_open(path)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/audioread/__init__.py", line 131, in audio_open
    raise NoBackendError()
audioread.exceptions.NoBackendError
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
Cell In[7], line 1
----> 1 run_step(
      2     "feature and spectrogram extraction",
      3     [sys.executable, "scripts/05_extract_features.py", "--metadata", "data/processed/metadata_clean.csv", "--features-output", "data/processed/features_mfcc.csv", "--spectrogram-dir", "data/processed/spectrograms", "--spectrogram-index-output", "data/processed/spectrogram_index.csv", "--quality-mode", "all_samples"],
      4     enabled=RUN_FEATURES,
Cell In[3], line 40, in run_step(name, args, enabled, requires, creates, strict_requires, force)
     36     if result.stdout:
     37         print(result.stdout)
     38     if result.stderr:
     39         print(result.stderr)
---> 40     result.check_returncode()
     41     return True
File /usr/lib/python3.12/subprocess.py:502, in CompletedProcess.check_returncode(self)
    500 """Raise CalledProcessError if the exit code is non-zero."""
    501 if self.returncode:
--> 502     raise CalledProcessError(self.returncode, self.args, self.stdout,
    503                              self.stderr)
CalledProcessError: Command '['/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python', 'scripts/05_extract_features.py', '--metadata', 'data/processed/metadata_clean.csv', '--features-output', 'data/processed/features_mfcc.csv', '--spectrogram-dir', 'data/processed/spectrograms', '--spectrogram-index-output', 'data/processed/spectrogram_index.csv', '--quality-mode', 'all_samples']' returned non-zero exit status 1.

---

due to these blunders we have made the changes in the audio_io.py and features.py

you can see it from the pushed files

---

again new errors in # 7: 

## feature and spectrogram extraction
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/05_extract_features.py --metadata data/processed/metadata_clean.csv --features-output data/processed/features_mfcc.csv --spectrogram-dir data/processed/spectrograms --spectrogram-index-output data/processed/spectrogram_index.csv --quality-mode all_samples
Wrote features: data/processed/features_mfcc.csv (19028 rows, 261 columns)
Wrote spectrogram index: data/processed/spectrogram_index.csv (19028 rows)


## classical ML baselines
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/06_train_ml_baselines.py --features data/processed/features_mfcc.csv
Trained dummy_most_frequent / cough: AUROC=0.5
Trained dummy_stratified / cough: AUROC=0.47945360126439385
Trained logistic_regression / cough: AUROC=0.7959471664032512
Trained random_forest / cough: AUROC=0.8089749379092346
Trained dummy_most_frequent / breath: AUROC=0.5
Trained dummy_stratified / breath: AUROC=0.47945360126439385
Trained logistic_regression / breath: AUROC=0.7523030029351997
Trained random_forest / breath: AUROC=0.7963535786859337
Trained dummy_most_frequent / speech: AUROC=0.5
Trained dummy_stratified / speech: AUROC=0.5145947166403251
Trained logistic_regression / speech: AUROC=0.7460347708286295
Trained random_forest / speech: AUROC=0.7569988710769926
Wrote metrics: data/outputs/metrics/ml_baseline_metrics.csv

SKIP compact CNN cough branch: disabled

## branch calibration
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/08_calibrate_branches.py --validation-predictions data/outputs/metrics/ml_predictions_validation.csv --test-predictions data/outputs/metrics/ml_predictions_test.csv --method platt
Wrote calibrated predictions: data/outputs/metrics/calibrated_branch_predictions.csv
Wrote calibration metrics: data/outputs/metrics/calibration_metrics.csv


## standard fusion
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/09_run_fusion.py --predictions data/outputs/metrics/calibrated_branch_predictions.csv --validation-metrics data/outputs/metrics/ml_baseline_metrics.csv
Wrote fusion predictions: data/outputs/metrics/fusion_predictions.csv
Wrote fusion metrics: data/outputs/metrics/fusion_metrics.csv


## subgroup and confounding checks
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/10_shift_and_confounding_checks.py --predictions data/outputs/metrics/fusion_predictions.csv --metadata data/processed/metadata_clean.csv
Traceback (most recent call last):
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/scripts/10_shift_and_confounding_checks.py", line 55, in <module>
    main()
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/scripts/10_shift_and_confounding_checks.py", line 39, in main
    merged = predictions.merge(metadata[[c for c in cols if c in metadata.columns]], on="recording_id", how="left")
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/pandas/core/frame.py", line 12900, in merge
    return merge(
           ^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/pandas/core/reshape/merge.py", line 385, in merge
    op = _MergeOperation(
         ^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/pandas/core/reshape/merge.py", line 1018, in __init__
    ) = self._get_merge_keys()
        ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/pandas/core/reshape/merge.py", line 1633, in _get_merge_keys
    left_keys.append(left._get_label_or_level_values(lk))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 1776, in _get_label_or_level_values
    raise KeyError(key)
KeyError: 'recording_id'

---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
Cell In[8], line 36
     32     enabled=RUN_FUSION,
     33     requires=["data/outputs/metrics/calibrated_branch_predictions.csv", "data/outputs/metrics/ml_baseline_metrics.csv"],
     34     creates=["data/outputs/metrics/fusion_predictions.csv", "data/outputs/metrics/fusion_metrics.csv"],
     35 )
---> 36 run_step(
     37     "subgroup and confounding checks",
     38     [sys.executable, "scripts/10_shift_and_confounding_checks.py", "--predictions", "data/outputs/metrics/fusion_predictions.csv", "--metadata", "data/processed/metadata_clean.csv"],
     39     enabled=RUN_SHIFT_CHECKS,

Cell In[3], line 40, in run_step(name, args, enabled, requires, creates, strict_requires, force)
     36     if result.stdout:
     37         print(result.stdout)
     38     if result.stderr:
     39         print(result.stderr)
---> 40     result.check_returncode()
     41     return True

File /usr/lib/python3.12/subprocess.py:502, in CompletedProcess.check_returncode(self)
    500 """Raise CalledProcessError if the exit code is non-zero."""
    501 if self.returncode:
--> 502     raise CalledProcessError(self.returncode, self.args, self.stdout,
    503                              self.stderr)

CalledProcessError: Command '['/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python', 'scripts/10_shift_and_confounding_checks.py', '--predictions', 'data/outputs/metrics/fusion_predictions.csv', '--metadata', 'data/processed/metadata_clean.csv']' returned non-zero exit status 1.


----

we solved the above errors by changing in the 10_shift_and_confounding_checks.py

---

# 7 Final output after resolving and modifying the fiesl for the changes

SKIP feature and spectrogram extraction: outputs already exist
SKIP classical ML baselines: outputs already exist
SKIP compact CNN cough branch: disabled
SKIP branch calibration: outputs already exist
SKIP standard fusion: outputs already exist

## subgroup and confounding checks
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/10_shift_and_confounding_checks.py --predictions data/outputs/metrics/fusion_predictions.csv --metadata data/processed/metadata_clean.csv
Wrote subgroup metrics: data/outputs/metrics/subgroup_metrics.csv

/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/lib/python3.12/site-packages/sklearn/metrics/_classification.py:614: UserWarning: A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.
  warnings.warn(


## basic report assets
$ /home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python scripts/11_make_report_assets.py --metadata data/processed/metadata_clean.csv --predictions data/outputs/metrics/fusion_predictions.csv
Report assets generated

True


---


# 8 

SKIP metadata-only baseline: disabled
SKIP quality-weighted fusion: disabled
SKIP abstention analysis: disabled
SKIP bootstrap CI for quality-weighted fusion: disabled
SKIP COUGHVID index: COUGHVID_RAW is None
SKIP COUGHVID feature extraction: disabled
SKIP cross-dataset cough evaluation: disabled
SKIP paper metric tables: disabled

False

---

# 9 

SKIP paired ML model comparison: disabled
SKIP confounding matched subset: disabled
SKIP feature shift report: disabled
SKIP experiment manifest: disabled

False

---

# 10

 	path 	exists 	size_bytes
0 	data/interim/coswara_index.csv 	True 	14469443
1 	data/processed/metadata_clean.csv 	True 	15194957
2 	data/interim/modality_availability.csv 	True 	211550
3 	data/interim/split_manifest.csv 	True 	243624
4 	data/processed/audio_quality.csv 	True 	8643961
5 	data/processed/features_mfcc.csv 	True 	90532655
6 	data/processed/spectrogram_index.csv 	True 	3880532
7 	data/outputs/metrics/ml_baseline_metrics.csv 	True 	2367
8 	data/outputs/metrics/calibrated_branch_predict... 	True 	1480440
9 	data/outputs/metrics/fusion_metrics.csv 	True 	390
10 	data/outputs/metrics/quality_weighted_fusion_m... 	False 	0
11 	reports/tables/paper_metric_table.csv 	False 	0
12 	data/outputs/metrics/paired_model_comparison.csv 	False 	0
13 	reports/tables/confounding_balance.csv 	False 	0
14 	reports/tables/feature_shift_report.csv 	False 	0
15 	reports/experiment_manifest.json 	False 	0


data/outputs/metrics/ml_baseline_metrics.csv

	auroc 	auprc 	balanced_accuracy 	f1 	sensitivity 	specificity 	brier 	ece 	nll 	threshold 	n_samples 	model_name 	modality
0 	0.500000 	0.323899 	0.500000 	0.000000 	0.000000 	1.000000 	0.323899 	0.323899 	4.474836 	0.5 	636.0 	dummy_most_frequent 	cough
1 	0.479454 	0.315672 	0.479454 	0.296117 	0.296117 	0.662791 	0.455975 	0.455975 	6.299526 	0.5 	636.0 	dummy_stratified 	cough
2 	0.795947 	0.672562 	0.739512 	0.647773 	0.776699 	0.702326 	0.190426 	0.161473 	0.571753 	0.5 	636.0 	logistic_regression 	cough
3 	0.808975 	0.716905 	0.653725 	0.484642 	0.344660 	0.962791 	0.170685 	0.120063 	0.522140 	0.5 	636.0 	random_forest 	cough
4 	0.500000 	0.323899 	0.500000 	0.000000 	0.000000 	1.000000 	0.323899 	0.323899 	4.474836 	0.5 	636.0 	dummy_most_frequent 	breath
5 	0.479454 	0.315672 	0.479454 	0.296117 	0.296117 	0.662791 	0.455975 	0.455975 	6.299526 	0.5 	636.0 	dummy_stratified 	breath
6 	0.752303 	0.554993 	0.690269 	0.595420 	0.757282 	0.623256 	0.206329 	0.149757 	0.597724 	0.5 	636.0 	logistic_regression 	breath
7 	0.796354 	0.671386 	0.671630 	0.532110 	0.422330 	0.920930 	0.169075 	0.064934 	0.516400 	0.5 	636.0 	random_forest 	breath
8 	0.500000 	0.323899 	0.500000 	0.000000 	0.000000 	1.000000 	0.323899 	0.323899 	4.474836 	0.5 	1590.0 	dummy_most_frequent 	speech
9 	0.514595 	0.330683 	0.514595 	0.342746 	0.341748 	0.687442 	0.424528 	0.424528 	5.865076 	0.5 	1590.0 	dummy_stratified 	speech
10 	0.746035 	0.603802 	0.678234 	0.582375 	0.737864 	0.618605 	0.210588 	0.161955 	0.605100 	0.5 	1590.0 	logistic_regression 	speech
11 	0.756999 	0.619671 	0.639327 	0.465359 	0.345631 	0.933023 	0.181491 	0.069858 	0.545127 	0.5 	1590.0 	random_forest 	speech


data/outputs/metrics/fusion_metrics.csv

	auroc 	auprc 	balanced_accuracy 	f1 	sensitivity 	specificity 	brier 	ece 	nll 	threshold 	n_samples 	fusion_method
0 	0.879070 	0.837577 	0.5 	0.0 	0.0 	1.0 	0.189724 	0.149166 	0.564733 	0.5 	318.0 	uniform_mean
1 	0.880515 	0.842599 	0.5 	0.0 	0.0 	1.0 	0.189448 	0.143521 	0.564107 	0.5 	318.0 	validation_weighted_auprc