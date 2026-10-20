# CamKIT-AI analysis code

Analysis code, fitted-pipeline specification, archived environment provenance
and aggregate outputs for the study:

> Transportability and software reproducibility of a machine-learning triage model
> for acute soft-tissue knee injury using patient-completed questionnaires:
> a secondary analysis of a prospective cohort at a UK urgent treatment center

CamKIT-AI is a 12-feature machine-learning triage model for acute soft-tissue
knee injury, developed with AutoPrognosis 2.0 from clinician-documented records
and evaluated in a secondary analysis of patient-completed questionnaires
from 85 prospectively recruited participants.

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

The legacy label "Original CamKIT" in code and aggregate outputs denotes the
comparator reconstructed from retained responses using the published 12-item
rules, rather than the historically stored scores.

## Reproducibility of inference

The serialised model is research-only and is not included here. In the study,
probabilities were invariant for records with all 12 inputs present. During
complete-cohort batch scoring, probabilities varied for every record with a
missing input and depended on the random seed and which other records were
scored alongside it. When scored individually, each incomplete record returned
one distinct probability across 30 tested seeds, although this could differ from
its batch-scored probability. The exact missing-input mechanism during individual
scoring remains unresolved, and repeatability does not establish clinical validity. The
400-draw mean reported in the paper describes offline scoring of a complete
cohort and is not a specification for isolated-patient inference. This release
documents that finding through the pipeline specification and non-disclosive
aggregate outputs; it cannot reproduce the fitted-model calls themselves.

The development-derived thresholds are discharge `p < 0.29`, reassessment
`0.29 <= p < 0.69` and MRI referral `p >= 0.69`. They were selected on the
development-stage hold-out after earlier exploratory work using prospective
outcomes and retained for this secondary analysis. Selection used one stochastic
realization of hold-out probabilities; evaluation applied the same cut-points to
later 400-run complete-cohort means.

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

## Citation

Citation metadata for this analysis package are provided in
[`CITATION.cff`](CITATION.cff). Once a release is archived on Zenodo, cite the DOI
for the specific version used.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

CamKIT-AI is built on [AutoPrognosis 2.0](https://github.com/vanderschaarlab/autoprognosis),
which is also Apache-2.0 licensed.
