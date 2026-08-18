# CamKIT-AI analysis code

Analysis code, fitted-pipeline specification, archived environment provenance
and aggregate outputs for the study:

> Machine-learning triage for acute soft-tissue knee injury using
> patient-completed questionnaires did not demonstrate incremental value over a
> 12-item consensus score: a secondary analysis.

CamKIT-AI is a 12-feature machine-learning triage model for acute soft-tissue
knee injury, developed with AutoPrognosis 2.0 from clinician-documented records
and evaluated prospectively on patient-completed questionnaires.

## What this repository contains

- `src/camkit_ai/` - the analysis package (discrimination, calibration,
  threshold analysis, inference-reproducibility checks)
- `tests/` - the public unit suite
- `config/camkit_legacy.yaml` - the legacy analysis preset; its data and model
  paths are provenance records and are not runnable without the withheld inputs
  and fitted artifact
- `environment/` - the fitted-pipeline specification and the provenance record
  of the artifact-matched Python 3.10 environment
- `aggregate_outputs/` - non-disclosive aggregate results underlying the
  manuscript tables
- `toy_data/` - synthetic rows illustrating the model's 12-feature input schema,
  including missing values; they support schema inspection and software tests
  but cannot exercise the withheld fitted model
- `docs/feature_provenance.md` - feature-selection provenance and its limits

## Reproducibility of inference

The serialised model is research-only and is not included here. In the study,
probabilities were invariant for records with all 12 inputs present but varied
for every record with a missing input; the returned probability depended on the
random seed and on which other records were scored in the same batch. The
400-draw mean reported in the paper describes offline scoring of a complete
cohort and is not a specification for isolated-patient inference. This release
documents that finding through the pipeline specification and non-disclosive
aggregate outputs; it cannot reproduce the fitted-model calls themselves.

The development-derived thresholds are discharge `p < 0.29`, reassessment
`0.29 <= p < 0.69` and MRI referral `p >= 0.69`. They were not reselected on
prospective outcomes or on the averaged probabilities.

## Running the tests

```bash
python -m pip install -e ".[dev]"
pytest -q
```

The archived artifact-matched environment is described in
[`environment/frozen_inference_environment.md`](environment/frozen_inference_environment.md).
That record includes exact historical VCS revisions and documents the private
artifact spot-check; it is provenance, not a claim that study inference can be
re-run from this public repository.

## What is not included, and why

Individual participant data and record-level predictions are not released: the
governing approvals do not permit open release of sensitive health data. The
serialised fitted model is also withheld. The fitted-pipeline specification
records the evaluated object and its checksum but does not provide access to it.

Independent verification is therefore limited to inspecting the fitted-pipeline
specification, testing the public analysis and threshold utilities, and checking
the reported aggregate outputs. The predictions and inference-time imputation
behaviour reported in the paper cannot be regenerated from this repository. The
toy rows document the input schema and support software tests; they do not
reproduce participant distributions, fitted-model predictions, model
performance or study results.

Additional aggregate material may be requested from the corresponding author
subject to institutional review.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

CamKIT-AI is built on [AutoPrognosis 2.0](https://github.com/vanderschaarlab/autoprognosis),
which is also Apache-2.0 licensed.
