# Reproducibility artifact

Code and per-stream artefacts for

> **Cost-Aware and Drift-Resilient AI Model Maintenance: A Robust Two-Signal
> Retraining Policy for Enterprise Security Architectures**
> Amirhossein Irankhah, Saeed Araban — Ferdowsi University of Mashhad

The policy retrains a malware classifier only when a cheap distributional screen
(Population Stability Index, PSI) and a structural confirmation (change in the
Silhouette coefficient, ΔS) both fire, with both thresholds calibrated from the
reference period by a robust MAD rule.

This package regenerates **Table I**, **Table II** (the component ablation), the **reference-period sweep
of Section V-C** and **Fig. 1** from the raw data, and checks the result against
the numbers printed in the manuscript. It is a frozen snapshot: the same scripts
also drive the manuscript build in the authors' tree.

## Quick start

```bash
cd repro
python reproduce.py check-data   # verifies data/ is in place
python reproduce.py tables       # ~15 min: Table I and Table II
python reproduce.py sweep        # ~40 min: Section V-C sweep (runs tables first)
python reproduce.py figure       # ~5 s: Fig. 1
python reproduce.py verify       # ~5 s, data-free: checks the shipped artefacts
```

`make <target>` does the same thing where `make` is installed. Set up the
environment once — the commands below assume Windows, which is where the
manuscript is built and where the figure font lives:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

On Linux or macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`.
`reproduce.py` picks either up automatically, or pass `--python`.

Every target is deterministic given the seeds below; `reproduce.py verify` prints
`all checks passed` when the artefacts match the numbers printed in the paper.

## What maps to what

| Paper item | command | artefact | script |
|---|---|---|---|
| Table I (periodic vs. proposed, 13 streams) | `make tables` | `artifacts/table1_source.json` | `run_baseline_matrix.py` → `build_table1_source.py` |
| Table II (component ablation: static / periodic / PSI-only / silhouette-only / two-signal) | `make tables` | `artifacts/baseline_components.json` | `build_component_source.py` |
| §V-C reference-period sweep (3 / 5 / 10 windows) | `make sweep` | `artifacts/reference_period_mlp43.json` | `build_reference_sweep.py` |
| §V-A / §V-B multi-benchmark view (incl. McNdroid) | `build_baseline_source.py` | `artifacts/baseline_source.json` | needs McNdroid, see `data/README.md` |
| Fig. 1 (algorithm flow chart) | `make figure` | `figures/fig_algorithm_flowchart.png` | `generate_algorithm_flowchart.py` |

Each artefact is JSON with a `provenance` block recording the run directories it
was built from. The two table builders **refuse to write** if the proposed-policy
rows do not reproduce Table I exactly, and the sweep builder applies the same
cross-check to its five-window run; a silently inconsistent table is therefore
not possible.

## Reported cell

| Setting | Value |
|---|---|
| Classifier | `MLPClassifier(hidden_layer_sizes=(50,), max_iter=300)` behind a `StandardScaler` |
| Classifier seed | 43 (the Random-Forest replication of §V-D uses 42) |
| Bootstrap seed | 42, 1,000 resamples, one-sided 95% bound |
| Window | 50 samples per temporal batch |
| Reference period | 5 windows |
| Threshold rule | robust MAD, `median + c · 1.4826 · MAD`, `c = 1.5` |
| Non-inferiority margin | δ = 2% accuracy against periodic retraining |
| Streams | 13: 5 KronoDroid families (478-d), 8 LAMDA families (925-d) |

## Expected numbers

Table I / Table II (the paper's headline results):

| Policy | updates | median saving | mean ΔAcc | in margin | certified |
|---|---:|---:|---:|---:|---:|
| Static | 13 | 98.4% | −8.95 | 4/13 | 3/13 |
| Periodic | 822 | 0.0% | +0.00 | 13/13 | 13/13 |
| PSI-only | 548 | 21.0% | −0.61 | 11/13 | 10/13 |
| Silhouette-only | 318 | 70.4% | −1.00 | 12/13 | 9/13 |
| **Two-signal** | **256** | **78.9%** | **−1.55** | **10/13** | **8/13** |

Section V-C sweep: 397 updates at 3 windows (51.7% saved), 256 at 5 (68.9%) and
210 at 10 (74.5%); median gap −0.30 / −1.14 / −1.17 pp; pooled certificates
10 / 8 / 7.

## Layout

```
repro/
├── reproduce.py             # every target in one place (stdlib only, no make needed)
├── Makefile                 # optional wrapper around reproduce.py
├── requirements.txt         # exact versions used for the reported runs
├── pyproject.toml           # installs `driftpolicy` from src/
├── LICENSE                  # MIT (code); artefacts are CC-BY-4.0
├── CITATION.cff             # release metadata for the archived DOI
├── artifacts/               # shipped JSON: the numbers behind Tables I & II and §V-C
├── data/                    # datasets (not redistributed), expected.json + acquisition notes
├── figures/                 # Fig. 1 output
├── scripts/
│   ├── run_baseline_matrix.py   # five-policy matrix, Table I / II / sweep
│   ├── build_table1_source.py   # per-stream Table I artefact
│   ├── build_component_source.py# Table II artefact (cross-checked vs Table I)
│   ├── build_reference_sweep.py # §V-C sweep artefact (cross-checked at 5 windows)
│   ├── build_baseline_source.py # multi-benchmark view (needs McNdroid)
│   ├── run_classifier_matrix.py # classifier replication of §V-D
│   ├── generate_algorithm_flowchart.py
│   ├── download_kronodroid.py, download_lamda.py
│   └── verify_numbers.py        # data-free check of the shipped artefacts
├── src/driftpolicy/         # the policy library (signals, policies, evaluation)
├── tests/                   # unit and end-to-end tests (synthetic data)
└── MANIFEST.json            # SHA-256 of every file, versions, provenance
```

## Fidelity notes

* **Figure font.** `scripts/generate_algorithm_flowchart.py` draws Fig. 1 with
  the Times New Roman files in `fonts/`, falling back to
  `%SystemRoot%\Fonts\times.ttf` on Windows. The bundled copies come first so
  the PNG hash in `MANIFEST.json` does not depend on the operating system; both
  are the same face Word sets the manuscript in.
* **Determinism.** `MLPClassifier` is seeded; the bootstrap uses its own seed.
  Re-running `make tables` reproduces the artefacts bit-for-bit in the authors'
  environment. Minor last-digit drift across BLAS builds is possible and does not
  move any verdict.
* **Sweep cell.** `reference_period_mlp43.json` was rebuilt in the Table I cell;
  an earlier sweep existed for the Random-Forest cell and is superseded. The
  builder's cross-check is what keeps the two from being mixed.
* **McNdroid.** The 2,390-dimensional stream is not redistributable with this
  package; `artifacts/baseline_source.json` is shipped so the Section V-A view
  can be inspected without it.
* **Not included.** The 14-scenario full-scope manuscript variant
  (`table1_source_full_scope.json`) and the manuscript typesetting code live in
  the authors' tree; this package covers the 13-stream results that the paper
  reports.

## Requirements and runtimes

| | |
|---|---|
| Python | 3.10 or newer (the reported runs used 3.12) |
| Dependencies | `requirements.txt` — exact versions used for the paper |
| Disk | ~1.1 GB after fetching both datasets (not shipped, see `data/README.md`) |
| Wall clock, one x86-64 core | `tables` ~15 min, `sweep` ~40 min, `figure` ~5 s, `verify` ~5 s |

## Step by step

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .   # installs the driftpolicy package
.\\.venv\\Scripts\\python.exe reproduce.py check-data   # both datasets present, hashes checked
.\\.venv\\Scripts\\python.exe reproduce.py tables         # Table I + Table II artefacts
.\\.venv\\Scripts\\python.exe reproduce.py sweep          # Section V-C sweep artefact
.\\.venv\\Scripts\\python.exe reproduce.py figure         # Fig. 1
.\\.venv\\Scripts\\python.exe reproduce.py verify         # all numbers against the paper
```

With the environment active (`.\.venv\Scripts\Activate.ps1`) the shorter
`python reproduce.py <target>` form is equivalent. On Linux/macOS use
`.venv/bin/python` instead of `.venv\Scripts\python.exe`.

What each step must report:

1. `check-data` lists KronoDroid and LAMDA with their SHA-256 digests, or stops
   with the missing path — it never silently falls back to synthetic data.
2. `tables` ends with `cross-check vs Table I: OK`; it refuses to write an
   artefact whose proposed-policy rows differ from Table I.
3. `sweep` prints the three reference-period rows and asserts that its
   five-window run equals Table I before writing.
4. `figure` rewrites `figures/fig_algorithm_flowchart.png`; its SHA-256 must
   match the entry in `MANIFEST.json`.
5. `verify` prints one `PASS` line per check (22 in total) and ends with
   `all checks passed`.

## Troubleshooting

* `ModuleNotFoundError: driftpolicy` — the package was not installed; run
  `./.venv/bin/pip install -e .` from this folder.
* `data hash mismatch` in `check-data` — re-download the file; the loaders
  detect the timestamp/family/label columns by name, so a re-export with the
  same columns is accepted once the digest matches.
* Last-digit differences in accuracies across BLAS builds are expected for the
  MLP; no verdict in Tables I–II or Section V-C changes.
* No `make`? Every target is a `reproduce.py` subcommand; the Makefile is a
  thin wrapper.

## Publishing the artifact

IEEE does not require a code link, but the accepted practice — and the condition
for an *Artifact Available* badge — is a **permanent archive with a DOI**: a
GitHub repository alone is not considered archival. The usual route is GitHub
for development plus one Zenodo release:

1. `git init && git add -A && git commit -m "Artifact v1.0.0"`, tag `v1.0.0`,
   push to GitHub (public, or private while the paper is under review).
2. Sign in to <https://zenodo.org> with GitHub, open *Settings → GitHub* and
   switch this repository on.
3. Create a GitHub *Release* for the tag. Zenodo archives that snapshot and
   mints a DOI immediately, free of charge; the *concept DOI* always resolves to
   the newest version, the version DOI pins `v1.0.0`.
4. Put the DOI in `CITATION.cff`, in the badge below, and in the paper:

```bash
PAPER_ARTIFACT_URL=https://github.com/<user>/<repo> \
PAPER_ARTIFACT_DOI=10.5281/zenodo.XXXXXXX \
python ../paper/build_official_ieee_paper.py --scope proposal
```

Without those variables the sentence keeps a `[URL]` placeholder, so a
half-filled link cannot reach a submission by accident.

**Double-blind venues.** Do not put the repository link in the submitted PDF;
use the artifact field of the submission form, or an anonymised mirror
(`anonymous.4open.science`), and add the DOI only to the camera-ready. The
`[URL]` placeholder in the current PDF is safe for a blind submission.

## Tests

```bash
./.venv/bin/python -m pytest tests -q
```

The end-to-end test runs the whole policy on synthetic drifted data and needs no
download.

## License and citation

Code: MIT (`LICENSE`). Artefacts (`artifacts/*.json`): CC-BY-4.0. Datasets keep
their own licences (KronoDroid, LAMDA); see `data/README.md`.
`CITATION.cff` carries the release metadata.

If you use this artifact, cite the paper and the archived release:

```bibtex
@misc{irankhah2026artifact,
  title  = {Reproducibility artifact for: Cost-Aware and Drift-Resilient AI
            Model Maintenance: A Robust Two-Signal Retraining Policy for
            Enterprise Security Architectures},
  author = {Irankhah, Amirhossein and Araban, Saeed},
  year   = {2026},
  note   = {Version 1.0.0},
  doi    = {10.5281/zenodo.XXXXXXX}
}
```
