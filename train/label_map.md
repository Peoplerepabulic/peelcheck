# Label mapping — PeelCheck

Source dataset (Kaggle, Apache 2.0): 4 folders per split
`unripe / ripe / overripe / rotten` (train 11,793 · valid 1,123 · test 562; 13,478 total).

## Model output (4-class, trained as-is)
| Dataset label | PeelCheck stage | UI suggestion |
|---|---|---|
| unripe | Underripe | Wait, or cook it (fried/grilled). Don't refrigerate yet. |
| ripe | Ripe | Eat now — peak sweetness and texture. |
| overripe | Overripe | Bake / blend / freeze today. Sweetest for banana bread. |
| rotten | Past best (safety) | Don't eat raw. If moldy or sour-smelling, compost it. |

Training keeps all 4 labels (merging rotten into overripe would teach the model
that moldy = bakeable). The UI groups them as 3 ripeness stages + 1 safety flag.

## Splits
Use the dataset's train/valid/test as shipped. No reshuffle across splits —
the test set stays untouched until final evaluation.
