# Feature Provenance

The manuscript primary model is `Injury.top12`. Its 12 inputs are exactly the 12 highest-ranked variables by mean absolute SHAP value in the retained development-stage analysis. They were mapped from the prospective patient questionnaire for this evaluation. The 27-feature development model is not reported as a definitive performance benchmark; its retained SHAP figure documents the feature-ranking route only. Retrospective clinician-documented fields and prospective patient-entered fields were normalised into the schema below before analysis.

| Retrospective feature | Prospective encoded input | In primary 12-feature model |
| --- | --- | --- |
| `age` | `age` | No |
| `sex` | `sex` | No |
| `bmi` | `bmi` | Yes |
| `h_injury` | `h_injury` | No |
| `h_injury_c` | `h_injury_c` | No |
| `h_surgery` | `h_surgery` | No |
| `h_surgery_c` | `h_surgery_c` | No |
| `gjh` | `gjh` | No |
| `activity_risk` | `activity_risk` | Yes |
| `participation_level` | `participation_level` | No |
| `activity_train_comp` | `activity_train_comp` | No |
| `surface` | `surface` | No |
| `footwear` | `footwear` | No |
| `weather` | `weather` | No |
| `contact_noncontact` | `contact_noncontact` | No |
| `pain_scale` | `pain_scale` | Yes |
| `twisting` | `twisting` | Yes |
| `hyperextension` | `hyperextension` | Yes |
| `medial_lateral` | `medial_lateral` | No |
| `popping` | `popping` | Yes |
| `weightbear` | `weightbear` | Yes |
| `swelling` | `swelling` | Yes |
| `rapid_delayed` | `rapid_delayed` | Yes |
| `bruising` | `bruising` | No |
| `reduced_rom` | `reduced_rom` | Yes |
| `locking` | `locking` | Yes |
| `instability` | `instability` | Yes |

## Relationship to original CamKIT

The original consensus-derived CamKIT score uses `h_injury`, `activity_risk`, `contact_noncontact`, `swelling`, `rapid_delayed`, `weightbear`, `reduced_rom`, `twisting`, `hyperextension`, `instability`, `popping` and `locking`. It is an unweighted sum of 12 binary items after the specified weight-bearing inversion.

CamKIT-AI and original CamKIT therefore share 10 questionnaire-derived variables: `activity_risk`, `swelling`, `rapid_delayed`, `weightbear`, `reduced_rom`, `twisting`, `hyperextension`, `instability`, `popping` and `locking`. CamKIT-AI uses `bmi` and `pain_scale` where CamKIT uses `h_injury` and `contact_noncontact`.

## Provenance boundary

The retained SHAP display establishes the ranking and its correspondence with the serialised 12 inputs. The project record does not establish the SHAP background/explanation samples, whether selection was nested, whether the hold-out cohort influenced selection, or the exact lock chronology relative to prospective outcome inspection. Consequently, the manuscript treats hold-out results as development-stage and does not claim prespecified external validation. The complete evidence record is retained by the authors and is available from the corresponding author on request.
