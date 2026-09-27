# SLICKTRACE AI

**SIH26143 — Oil Spill Detection & Evidence-Based Vessel Attribution**

> **Detect the slick. Trace its probable origin. Rank the vessels.**

SlickTrace AI is an investigative decision-support prototype for oil-spill analysis using Sentinel-1 SAR imagery, image segmentation, spill characterization, backward-drift modelling, and historical AIS evidence.

The system is designed to support investigation by connecting multiple evidence sources. It does **not** declare a vessel responsible or guilty.

---

## 1. Problem

Oil-spill investigation requires more than detecting a dark region in satellite imagery.

SlickTrace AI addresses four linked questions:

1. **Is an oil-like slick present?**
2. **What are its geometric characteristics and location?**
3. **What region could the slick have originated from based on backward drift?**
4. **Which historical AIS vessel tracks are consistent with the available evidence?**

The output is an evidence-based candidate ranking for investigator review.

---

## 2. System Workflow

```text
Sentinel-1 SAR
      |
      v
SAR Preprocessing
      |
      v
DeepLabV3+ + MobileNetV2
Binary Oil-Spill Segmentation
      |
      v
Spill Characterization
Polygon / Area / Centroid
      |
      v
Backward Drift Prototype
Probable Origin Region
      |
      v
Historical AIS Processing
Spatial + Trajectory Evidence
      |
      v
Candidate Vessel Ranking
      |
      v
SlickTrace Investigation Dashboard
3. Current Implementation
Satellite / SAR
Sentinel-1 SAR imagery
Two-channel VV + VH input
2048 × 2048 input scenes
Raster-based processing
Segmentation Model
DeepLabV3+
MobileNetV2 encoder
Binary oil-spill segmentation
PyTorch
segmentation_models_pytorch
Combined BCE-with-logits and Dice loss
Batch size: 1
Learning rate: 0.0001
Weight decay: 0.0001
Automatic mixed precision enabled during training
Spill Characterization

Detected spill regions are characterized using:

spill mask
geographic centroid
estimated area
geometric spill information
Backward Drift

The current implementation contains a custom backward-drift prototype.

For the selected demonstration case:

Spill centroid: 28.942782, -88.834958
Estimated probable origin: 29.025164, -88.929210
Uncertainty radius: 5 km
Simulation duration: 12 hours
Time step: 0.5 hour
Number of steps: 25

The drift result represents a probable origin region, not an exact release location.

AIS Evidence

The prototype uses historical MarineCadastre AIS data.

Current demonstration AIS processing includes:

historical AIS records
vessel trajectory processing
spatial proximity analysis
trajectory-based evidence
candidate vessel evidence ranking

For the current demonstration case:

AIS date: 2025-01-08
Unique vessels processed: 342
Primary proximity candidates: 96
Broad discovery radius: 50 km
Primary proximity radius: 10 km

Temporal compatibility is currently unavailable for the selected demonstration scene because a verified satellite acquisition timestamp is not available to the dashboard.

4. Model Evaluation

Evaluation on the Part III test set produced the following aggregate results:

Metric	Score
Dice	0.6601
IoU	0.6261
Precision	0.7032
Recall	0.6538

These are aggregate Part III results.

A separate 12-scene gallery contains selected high-performing examples for visual demonstration. Those curated examples are not presented as the overall model performance.

5. Evidence-Based Attribution

SlickTrace does not treat the nearest vessel as the responsible vessel.

The attribution stage combines available evidence to produce a ranked list of candidate vessels for investigator review.

The ranking should be interpreted as:

Which vessels have evidence consistent with the investigated spill?

It should not be interpreted as:

Which vessel caused the spill?

The system therefore preserves the distinction between:

detection and attribution
spill location and release location
proximity and responsibility
backward drift and exact source proof
evidence ranking and guilt

A valid outcome may also be that the available evidence is insufficient to identify a strong candidate.

6. Investigation Dashboard

The Streamlit dashboard provides the following investigation views:

Case File

Incident-level summary of the selected investigation.

Model Evidence

Displays SAR imagery, model prediction, and available ground-truth evidence for demonstration scenes.

Spill Analysis

Shows detected spill characteristics including location and estimated area.

Drift & Origin

Visualizes the backward-drift prototype and the estimated probable origin region.

Vessel Attribution

Displays historical AIS evidence and ranked candidate vessels.

About

Explains the system methodology, interpretation, and limitations.

7. Repository Structure
SlickTrace-AI/
|
├── README.md
├── requirements.txt
├── .gitignore
|
├── dashboard/
|   ├── app.py
|   └── dashboard_requirements.txt
|
├── scripts/
|   ├── predict.py
|   ├── characterize_spills.py
|   ├── geospatial_spill_characterization.py
|   ├── backward_drift_prototype.py
|   ├── ais_real_parser.py
|   ├── ais_trajectory_analysis.py
|   ├── ais_geometry_analysis.py
|   ├── build_ais_evidence.py
|   └── ais_candidate_prototype.py
|
└── docs/

Large datasets, model checkpoints, raw AIS files, generated outputs, and local Python environments are intentionally excluded from the public repository.

8. Running the Dashboard Locally

Create and activate a Python environment, install the dashboard dependencies, and run Streamlit:

cd C:\SlickTrace
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -r dashboard\dashboard_requirements.txt

cd dashboard
streamlit run app.py

The current dashboard demonstration uses local project data and demonstration assets.

9. Data and Prototype Scope

The current prototype has been developed and demonstrated using Sentinel-1 SAR scenes and historical AIS data.

The current backward-drift implementation is a prototype and should not be interpreted as an operational oceanographic hindcast.

Operational OpenDrift/OpenOil integration and live CMEMS/ERA5-driven modelling are future extensions rather than claims about the current implementation.

Similarly, the current AIS workflow uses historical AIS data rather than a real-time AIS feed.

10. Limitations

The current prototype has several deliberate limitations:

backward drift is a custom prototype rather than a validated operational oceanographic model
environmental forcing is not currently an operational CMEMS/ERA5 integration
temporal AIS compatibility is unavailable when a verified satellite acquisition timestamp is absent
AIS evidence is historical rather than real-time
candidate ranking is evidence-based decision support, not proof of responsibility
the system does not determine legal liability or guilt
satellite spill detection and source attribution remain separate stages
11. Future Extensions

Potential future work includes:

operational ocean-current and wind forcing
OpenDrift/OpenOil integration
CMEMS and ERA5 environmental data integration
stronger temporal matching between satellite observations and AIS tracks
larger-scale historical AIS processing
additional validation across geographic regions
improved uncertainty modelling
deployment with larger operational data pipelines

These are future extensions and are not presented as current capabilities.

12. Project

SIH Project: SIH26143

Project: SLICKTRACE AI

USP: Detect the slick. Trace its probable origin. Rank the vessels.

SlickTrace AI is intended as an investigator-facing decision-support prototype that brings satellite detection, geospatial analysis, drift reasoning, and AIS evidence into a single workflow.