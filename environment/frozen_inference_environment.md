# Archived inference environment and reproduction record

**Recorded:** 8 August 2026  
**Purpose:** preserve the environment provenance for the private spot-check of the serialised `Injury.top12` artifact
**Status:** private artifact spot-check passed; public study inference is not reproducible from this repository

## Platform

- WSL2 Linux x86-64: `Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- Python: `3.10.13`, conda-forge build `hd12c33a_1_cpython`
- Compiler recorded by Python: GCC 12.3.0
- Conda: 24.9.0 at environment-export time
- Artifact SHA-256: `7324cf556af10a97efc25bef3235b3e311cf968f04e2e578432540b6633b572f`
- Controlled-draw matrix SHA-256: `7c556aa956de8445114dcb36e6a3635d8aaeff786f57b16f0f4c103c2270e1f4`

The exact conda-managed base is recorded in `inference_environment_conda_explicit_linux-64.txt`; the full Python package state is recorded in `frozen_inference_requirements.txt`. These files are immutable provenance records. Some entries refer to exact historical VCS revisions and may require repository access; moreover, the participant inputs and fitted artifact are withheld. They therefore do not constitute a publicly runnable reproduction environment. The most consequential packages were:

| Package | Version or immutable revision |
| --- | --- |
| AutoPrognosis | 0.1.21; Git commit `7ca93353d5939d65638f9a065840d649cf4d1554` |
| HyperImpute | 0.1.17; Git commit `e6e444b92ab8f4605472a9b857eba70080ae70c0` |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| SciPy | 1.15.1 |
| CatBoost | 1.2.7 |
| LightGBM | 4.6.0 |
| XGBoost | 2.1.4 |
| SHAP | 0.44.0 |
| PyTorch | 2.5.1+cu124 |

## Artifact spot-check

The artifact was loaded once, and draws 0 and 399 (seeds 42 and 441) were regenerated for the complete hold-out and prospective cohorts. Each regenerated probability vector was compared row-for-row with the archived controlled-draw matrix.

| Cohort | Draw | Seed | Maximum absolute probability difference |
| --- | ---: | ---: | ---: |
| Internal hold-out | 0 | 42 | `8.3266726846886741e-17` |
| Internal hold-out | 399 | 441 | `8.3266726846886741e-17` |
| Prospective | 0 | 42 | `8.3266726846886741e-17` |
| Prospective | 399 | 441 | `8.3266726846886741e-17` |

Acceptance tolerance was `1e-9`; all four checks passed.

## Interpretation

The historical environment name was not preserved as an immutable export. This reconstructed, fully pinned environment is nevertheless artifact-equivalent for the required end-point draws and both cohorts at floating-point precision. The record establishes reproducibility of the checked artifact calls; it does not make the model inference-safe. Predictions for incomplete records remain stochastic and dependent on batch composition, and access to participant inputs and the fitted artifact remains governed.

## Historical reconstruction record

The following commands record how the private verification environment was
constructed. They require access to every referenced repository, the governed
participant inputs and the withheld fitted artifact; they cannot reproduce the
study from the public release alone.

From a Linux x86-64 host with conda available:

```bash
conda create --name camkit-inference --file environment/inference_environment_conda_explicit_linux-64.txt
conda activate camkit-inference
python -m pip install --requirement environment/frozen_inference_requirements.txt
```

The CUDA-tagged PyTorch wheels may require the corresponding PyTorch wheel index. A CPU-only substitution must not be assumed equivalent without rerunning the four artifact spot-checks.
