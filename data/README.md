# Data

The datasets are **not** redistributed here (size and licence). Fetch them into
this folder in the layout below, then verify the digests with
`python reproduce.py check-data` — it compares every file against
`data/expected.json` (size and SHA-256) and refuses to continue on a mismatch.

```
repro/data/
├── expected.json                    # names, sizes, SHA-256, sources
├── KronoDroid_real_device.csv       # 87,957,264 bytes
└── lamda/
    ├── 2013_test.parquet            # 2,280,807 bytes
    ├── 2014_test.parquet            # 2,632,337 bytes
    ├── 2016_test.parquet            # 3,010,165 bytes
    └── feature_mapping.csv          # 57,873 bytes
```

`KronoDroid_emulator.csv` (72,138,877 bytes) is **optional**: only the
full-scope manuscript variant uses it, and `check-data` reports it as absent
without failing.

## KronoDroid

The KronoDroid feature sets (real device and emulator) are distributed by the
Canadian Institute for Cybersecurity (UNB):
<https://www.unb.ca/cic/datasets/kronodroid.html>. The page asks for a short
registration form, so there is no fixed URL to hard-code. Once you have the CSV:

```bash
python scripts/download_kronodroid.py \
    --url "<direct csv url>" --out data/KronoDroid_real_device.csv
```

The helper streams the bytes unchanged and checks the digest against
`expected.json`; it deletes the file and exits non-zero if the URL is not the
CSV used in the paper. Do **not** re-export the CSV through pandas: a
round-trip re-serialises the floats, which changes the SHA-256 without changing
the content.

SHA-256 of the file used for the paper:

```
29579a66eda59e7d989310ee2b3758302ec1007f5856bc4e2e756bba65f6c31a  KronoDroid_real_device.csv
4acd89ea03b867669a6823fa7a02680f9825df1b631969169ba8caefb9542f55  KronoDroid_emulator.csv (optional)
```

The loader detects the timestamp, family, label and numeric columns by name, so
an equivalent export with the same columns is accepted once the digest matches.

## LAMDA

```bash
python scripts/download_lamda.py --out-dir data/lamda
```

The helper pulls `var_thresh_0.01` from the `IQSeC-Lab/LAMDA` dataset on Hugging
Face (<https://huggingface.co/datasets/IQSeC-Lab/LAMDA>), then verifies every
file it wrote against `expected.json` and exits non-zero on a mismatch. The
paper's LAMDA streams use the 2013, 2014 and 2016 partitions, so the folder
above is sufficient.

SHA-256 of the files used for the paper:

```
4ebf0eee41593bd8a09070862a264cd236d91ac1024af4c0852d0a86be3d2e43  2013_test.parquet
302a11d152d34612ee65826d7838083d9ee72bfc27f506011d2e8031a3ed8715  2014_test.parquet
0937443619408635e64b4ac3f22dcc0873f93d998878aa0a0722325517cbe9f8  2016_test.parquet
5d1a5e6378b60d090854c556326a997599ec3bd953acdbc708f84cb9b85f0c87  feature_mapping.csv
```

## McNdroid (optional)

`scripts/build_baseline_source.py` additionally reads a McNdroid export
(`data/mcndroid`). It is required only to regenerate
`artifacts/baseline_source.json`, which backs the 2,390-dimensional row of
Section V-A; Tables I and II and the Section V-C sweep do not need it.

## Using a copy kept elsewhere

`reproduce.py` reads `KRONO` and `LAMDA` from the environment, so the data does
not have to live inside this folder:

```bash
KRONO=/path/to/KronoDroid_real_device.csv \
LAMDA=/path/to/lamda \
python reproduce.py check-data
```
