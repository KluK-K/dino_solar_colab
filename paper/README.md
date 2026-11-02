# Current paper: revision 63

- [Main article](main_article.pdf): 14 pages.
- [Supplement](supplementary_record.pdf): 21 pages.
- [Source/cost figure companion](source_and_cost_figures.pdf): development and efficiency context.

## Visual reading guide

| Where | Figure |
|---|---|
| Main page 5, Figure 2 | Implemented encoder, LoRA, feature fusion, decoder and image branch |
| Main page 11, Figure 4 | All six prospective fixed SCDD cases: input, annotation, CNN/Frozen/LoRA/Reference maps |
| Supplement S20 | Full detailed layer architecture |
| Supplement S21 | Decoder layer operations and tensor dimensions |

The heatmaps are saved pixel-head crack softmax scores, not image-classifier Grad-CAM. Image decisions shown below them use the original source thresholds. No training, scoring, threshold fitting or favorable-case substitution was performed for this update.

The owner authorized this specific flattened illustration and these PDFs for private review after the local visual copy was completed. “Owner-local” wording in the figure caption describes its creation stage; the later limited sharing authorization is recorded in [VISUAL_UPDATE_CONTEXT.json](VISUAL_UPDATE_CONTEXT.json). Raw datasets and separate prediction arrays remain excluded.

The unchanged revision61 paper/source bundle remains under [reference](../reference/README.md) as a dated scientific snapshot. Current visuals do not alter the original statistical findings, registry or evaluator.
