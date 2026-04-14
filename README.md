# Deep Learning in Computer Vision - Assignment Skeleton

This repository is a fast-start template for a computer vision assignment when time is limited.
It is designed for image classification first, with clean separation so you can extend it to
detection or segmentation later if needed.

## 1) Where To Get Datasets

Use `docs/dataset_sources.md` for direct links and download notes.

If your assignment does not force a specific dataset, use one of these:

- Fastest start (built in): CIFAR-10 via torchvision
- Better visual report quality: Intel Image Classification (Kaggle)
- Cleaner classes and good visuals: Oxford 102 Flowers

## 2) Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Choose dataset mode in `configs/classification_baseline.yaml`.

   - For built-in quick start, set `dataset.name: cifar10`.
   - For your own folders, set `dataset.name: imagefolder` and arrange your files like this:

   ```text
   data/raw/
     train/
       class_1/*.jpg
       class_2/*.jpg
     val/
       class_1/*.jpg
       class_2/*.jpg
     test/                # optional but recommended
       class_1/*.jpg
       class_2/*.jpg
   ```

4. Train:

   ```bash
   python -m src.train --config configs/classification_baseline.yaml
   ```

5. Evaluate:

   ```bash
   python -m src.evaluate --config configs/classification_baseline.yaml --checkpoint outputs/checkpoints/best.pt
   ```

6. Predict one image:

   ```bash
   python -m src.infer --config configs/classification_baseline.yaml --checkpoint outputs/checkpoints/best.pt --image /absolute/path/to/image.jpg
   ```

## 3) File Structure

```text
Assignment/
  configs/
    classification_baseline.yaml   # Main knobs: dataset, model, training
  data/
    README.md                      # Expected dataset structure
    .gitkeep
  docs/
    dataset_sources.md             # Dataset links for classification/detection/segmentation
  outputs/
    .gitkeep
  src/
    config.py                      # Strongly-typed config loader + device selection
    train.py                       # Main training entrypoint
    evaluate.py                    # Checkpoint evaluation script
    infer.py                       # Single-image inference script
    data/
      transforms.py                # Train/eval transforms
      build.py                     # Dataloaders for ImageFolder or CIFAR-10
    models/
      factory.py                   # Model construction (timm first, torchvision fallback)
    engine/
      train_loop.py                # One-epoch train loop
      eval_loop.py                 # Validation/test loop + predictions
    utils/
      io.py                        # JSON and directory helpers
      seeding.py                   # Reproducibility seed helper
  .gitignore
  requirements.txt
  README.md
```

## 4) What To Edit First

- `configs/classification_baseline.yaml`
- `docs/dataset_sources.md` (pick one dataset path)
- If custom dataset, place images under `data/raw/...` as shown in `data/README.md`

## 5) Submission-Focused Output Checklist

- Best checkpoint in `outputs/checkpoints/best.pt`
- Training history in `outputs/history.csv`
- Evaluation metrics from `src.evaluate`
- Prediction examples from `src.infer`
- Report figures: training/validation accuracy, confusion matrix, and error analysis
