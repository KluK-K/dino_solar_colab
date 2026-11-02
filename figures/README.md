# Which figure to show?

1. [held_source_tradeoff.png](held_source_tradeoff.png): **first result figure**. All six original held-source conditions and conditional unadjusted intervals.
2. [matched_ucf20_transfer.png](matched_ucf20_transfer.png): **positive result plus boundary**. All four central matched UCF 20% systems, on UCF and transferred to SCDD.
3. [implemented_model_workflow.png](implemented_model_workflow.png): **what was built**, also main article Figure 2, page 5.
4. [fixed_confirmation_heatmaps.png](fixed_confirmation_heatmaps.png): **actual saved outputs**, also main article Figure 4, page 11. All six prospective fixed SCDD cases, four central systems, first declared seed 11.
5. [dinov3_layer_architecture.png](dinov3_layer_architecture.png): detailed encoder/decoder/image-head schematic, supplement S20.
6. [decoder_layers_detail.png](decoder_layers_detail.png): decoder close-up, supplement S21.
7. [study_design.png](study_design.png): selection exposure, freeze and confirmation.

The first two reader figures use saved aggregates via [build_reader_figures.py](../tools/build_reader_figures.py). The fixed-case figure reuses the original saved maps without model calls or new case selection. It is a pixel-head response display, not image-classifier Grad-CAM. Subsets are in [results](../results/README.md); exhaustive tables remain in [reference](../reference/README.md).

The [source/cost companion](../paper/source_and_cost_figures.pdf) supports deeper review. Its development numbers are not confirmation outcomes.

Only the owner-authorized six-case illustration is shared as a flattened PNG/PDF and within the article. Dataset files, separate mask files and per-image prediction arrays/maps remain excluded. The original case identities, full fields, shared color scale and adverse responses are retained.
