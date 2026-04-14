# Part 1 Requirement Mapping

This document maps the assignment PDF requirements to code in this repository.

## Data Preparation (5%)

1. Use a publicly available benchmark dataset for emotion classification
- Implemented in: `src/part1/data_prep.py`
- Dataset: FER2013 via Hugging Face

2. Store images of each class in separate folders
- Implemented in: `src/part1/data_prep.py`
- Output format: `data/raw/{train,val,test}/{class_name}/*.jpg`

3. Split with 70/20/10 ratios
- Implemented in: `src/part1/data_prep.py`
- Config: `configs/part1_data_prep.yaml`

4. Remove redundant images
- Implemented in: `src/part1/data_prep.py`
- Exact duplicate removal by pixel hash

5. Resize to 512x512x3
- Implemented in: `src/part1/data_prep.py`

## First Model (35%)

1. Single convolution layer with 5 predefined 3x3x3 filters
- Implemented in: `src/part1/model1_scratch.py`
- Class: `ConvLayer`

2. PoolingLayer with 2x2 default and MAX/AVERAGE type
- Implemented in: `src/part1/model1_scratch.py`
- Class: `PoolingLayer`

3. Methods for filter iteration and forward pass
- Implemented in: `src/part1/model1_scratch.py`
- `iterate_regions(...)` and `forward(...)`

4. Activation after pooling
- Implemented in: `src/part1/model1_scratch.py`
- Configurable simple activation (default ReLU)

5. Network of 3 convolution blocks + flatten + downsample to 1x128
- Implemented in: `src/part1/model1_scratch.py`
- Class: `ScratchConvNet`

6. K-means classification
- Implemented in: `src/part1/model1_scratch.py`
- Class: `KMeansNumpy`

## Second Model (30%)

1. CNN with convolution layers + ReLU + 2x2 max pooling
- Implemented in: `src/part1/model2_cnn.py`

2. Successive filter sizes/channels as listed in the PDF
- Config: `configs/part1_model2.yaml`
- Implemented in: `src/part1/model2_cnn.py`

3. Flatten operation
- Implemented in: `src/part1/model2_cnn.py`

4. One hidden fully connected layer with Sigmoid
- Implemented in: `src/part1/model2_cnn.py`

5. Softmax output layer
- Implemented in: `src/part1/model2_cnn.py`

6. Training and checkpointing
- Implemented in: `src/part1/train_model2.py`
