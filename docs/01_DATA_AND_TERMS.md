# Data and terms

[Start](../README.md) · [Results](02_RESULTS.md) · [Training](03_METHOD_AND_TRAINING.md)

| Name | Meaning | Actually used population |
|---|---|---|
| SCDD | Our shorthand for the solar-cell defect benchmark in [Benchmark ELimages](https://github.com/TheMakiran/BenchmarkELimages), Pratt et al. | 588 audited images: 351 train, 120 validation, 117 confirmation |
| ARTS | ARTsolar, South African module manufacturer; SCDD source code | 43 matching confirmation images |
| CFVS | CFV Labs, US module test lab; SCDD source code | 27 matching confirmation images |
| CSIR | CSIR Energy Centre, South Africa; SCDD source code | 47 matching confirmation images |
| UCF | University of Central Florida's [UCF-EL-Defect](https://github.com/ucf-photovoltaics/UCF-EL-Defect) | 430 source train + 144 validation; 507 reported-real confirmation images |
| PVEL | [PVEL-AD](https://github.com/binyisu/PVEL-AD), photovoltaic anomaly data | 12,326 scored images; primary 12,292 = 943 crack/star positives + 11,349 author-normal negatives |
| ELPV | [Electroluminescence photovoltaic dataset](https://github.com/zae-bayern/elpv-dataset), expert review labels | Separate research supplement: nested CV on 1,968 development images |

ARTS, CFVS and CSIR are **three sources inside SCDD**, not extra models or independent studies. Source meanings follow the dataset authors' description. Counts describe our audited subsets, not the authors' entire datasets. File-prefix families are not verified physical modules.

## Three questions, three types of evidence

| Question | Metric | Meaning |
|---|---|---|
| Is the image suspicious for annotated cracks? | Image AUROC / AP | How well scores rank positives above negatives |
| Does the predicted mask recover cracks? | Positive-image Dice / native pixel AP | Fixed-rule overlap / whole-cohort pixel ranking |
| What happens at a decision threshold? | Recall / FPR / review fraction | Misses, false alarms and workload |

AUROC 0.95 is not 95% accuracy. High image recall does not establish a correct crack mask. PVEL boxes support coarse box-point hit, not mask Dice. Native pooled pixel AP is a cohort point estimate, not billions of independent samples.

## Exposure and population cautions

- Source validation selects settings. Held-source development can still inform study-wide design. Final confirmation uses the single frozen protocol; both UCF and SCDD development had already influenced design.
- Matched UCF 20% uses 86 training + 144 validation masks **inside** the 574-image selection pool.
- SCDD 5% uses 18 training masks **plus 120 validation masks** and upstream selection exposure.
- The 34 members of 17 reviewed PVEL near-copy pairs were excluded prospectively from the primary view. Full-pool and broader 11,927-image sensitivities reuse the same saved predictions.
- UCF normal-structure FPR and empty-denominator rates stay null. Physical modules, all cropped overlaps and pretraining membership remain unverified.

The research supplement did not use the 656-image course holdout. A later **separate course-only RBF-SVM** evaluation was authorized on 1 November; no DINOv3 evaluation of that holdout occurred. The October manuscript remains a dated snapshot.
