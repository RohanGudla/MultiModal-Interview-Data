# Frequently Asked Questions (FAQ)

## What exactly is being classified?
The models classify **video frames**, not full videos. During data preparation, each interview video is sampled at 1 frame per second. These frames are then paired with attention labels from the AFFDEX annotations and used as inputs to the models.

## How are frames sampled?
`extract_real_video_frames.py` extracts every 30th frame from the 30 FPS GENEX videos (1 FPS). Only a subset of frames (about 20 per participant) is stored under `data/real_frames/`.

## Are all videos used for training?
The configuration in `src/utils/config.py` assigns three participants to training (`CP 0636`, `JM 9684`, `MP 5114`), one to validation (`NS 4013`), and one to testing (`LE 3299`). Annotation CSV files exist for four participants only, so the test participant lacks annotation files.

## Can we save model outputs and true labels?
Yes. Use `scripts/evaluate_all_models.py` or a similar evaluation script. It loads saved checkpoints and writes each model's predictions and ground‑truth labels to JSON files under `experiments/model_comparison/`. You can rerun evaluation without retraining.

