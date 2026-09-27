SLICKTRACE AI — JUDGE DEMO TIFF GALLERY

Purpose
-------
A compact collection of 12 high-performing REAL OIL examples
from the completed prediction run.

Selection
---------
Source run:
C:\SlickTrace\outputs\test_predictions_055\prediction_results.csv

Only PART_I_OIL examples were considered.
Examples were sorted by:
1. Dice score
2. IoU
3. Precision

No-oil examples were intentionally excluded, even when their Dice score
was 1.0, because this gallery is intended to demonstrate actual
oil-spill segmentation.

Each scene contains:
- input_sar.tif          Original SAR input
- ground_truth_mask.tif  Reference mask
- model_prediction.tif  Model output

Metrics
-------
See demo_metrics.csv for Dice, IoU, Precision, Recall and predicted pixels.

Judge presentation
-------------------
Recommended visual flow:

SAR INPUT
   ->
MODEL PREDICTION
   ->
GROUND TRUTH COMPARISON
   ->
SPILL CHARACTERIZATION
   ->
PROBABLE ORIGIN
   ->
AIS CANDIDATE EVIDENCE

Important
---------
These examples demonstrate successful model cases.
They should not be presented as representative of every possible
oil-spill scene.

SlickTrace AI is an evidence-based decision-support system.
A detected spill and a ranked vessel candidate do not establish
responsibility or guilt.
