# Deep Learning for Computer Vision - Assignment (Part 1)

This repository now implements only Part 1 of the assignment PDF.
Part 2 and Part 3 are intentionally not implemented yet.

## 1) What Is Implemented

Part 1 Data Preparation:
- Uses a benchmark emotion dataset (FER2013 from Hugging Face).
- Selects 4 emotion classes.
- Removes redundant exact-duplicate images.
- Resizes to 512x512x3.
- Splits into train/val/test at 70/20/10.

Part 1 Model 1 (from scratch):
- Convolution and pooling implemented with NumPy only.
- Uses predefined 3x3x3 filters.
- Builds a 3-block convolution pipeline.
- Flattens and downsamples feature vector to 1x128.
- Uses K-means clustering (implemented from scratch with NumPy).

Part 1 Model 2 (library CNN):
- Uses PyTorch.
- Valid convolutions only.
- Convolution blocks follow the PDF constraints.
- One hidden fully connected layer with Sigmoid.
- Output layer uses Softmax.

## 2) Quick Start (Part 1 Only)

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare data (FER2013 -> four classes -> 512x512x3 -> 70/20/10 split):

```bash
python -m src.part1.data_prep --config configs/part1_data_prep.yaml
```

4. Run Model 1 (scratch conv + k-means):

```bash
python -m src.part1.model1_scratch --config configs/part1_model1.yaml
```

5. Train Model 2 (PyTorch CNN):

```bash
python -m src.part1.train_model2 --config configs/part1_model2.yaml
```

By default, `configs/part1_model2.yaml` uses a lower training image size for practical runtime.
If you need strict end-to-end 512 input training, change `data.image_size` to `512`.

## 3) Important Note About The PDF Activation Formula

The line in the provided PDF where Model 1 activation is "defined as" contains a formula
that is not machine-extractable in plain text. This implementation uses ReLU as the default
simple activation and allows easy switching in config if needed.

## 4) File Structure (Part 1 Additions)

```text
Assignment/
  configs/
    part1_data_prep.yaml          # Dataset source + split + resize rules
    part1_model1.yaml             # Model 1 scratch settings
    part1_model2.yaml             # Model 2 training settings
  docs/
    part1_requirements_map.md     # Requirement-to-implementation mapping from PDF
  src/
    part1/
      data_prep.py                # FER2013 preparation and folder export
      model1_scratch.py           # Scratch conv/pool/activation + k-means
      model2_cnn.py               # CNN architecture for Model 2
      train_model2.py             # Training script for Model 2
```

## 5) Expected Outputs

Model 1 outputs:
- outputs/part1/model1/metrics.json
- outputs/part1/model1/predictions_test.csv

Model 2 outputs:
- outputs/part1/model2/checkpoints/best.pt
- outputs/part1/model2/history.csv
- outputs/part1/model2/metrics.json
