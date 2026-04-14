# Dataset Sources For Computer Vision Assignments

This list is optimized for fast academic delivery and manageable training time.

## Classification (Recommended For Fast Delivery)

1. CIFAR-10 (10 classes, tiny images, very fast baseline)
   - Link: https://www.cs.toronto.edu/~kriz/cifar.html
   - Also available directly in torchvision (`dataset.name: cifar10` in config)

2. Intel Image Classification (Kaggle, realistic scenes)
   - Link: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
   - Great for report visuals and confusion matrix discussion

3. Oxford 102 Flowers
   - Link: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
   - Good if your assignment values fine-grained classes

4. Caltech-101
   - Link: https://data.caltech.edu/records/mzrjq-6wc02
   - Diverse objects, moderate size

## Object Detection

1. Pascal VOC 2007/2012
   - Link: http://host.robots.ox.ac.uk/pascal/VOC/
   - Common academic baseline for mAP reporting

2. COCO 2017 (large)
   - Link: https://cocodataset.org/
   - Use only if your compute budget is sufficient

## Segmentation

1. Oxford-IIIT Pet (includes segmentation masks)
   - Link: https://www.robots.ox.ac.uk/~vgg/data/pets/

2. Cityscapes (urban scenes, heavier setup)
   - Link: https://www.cityscapes-dataset.com/

## How To Choose Quickly (For Friday Submission)

- If you need speed and guaranteed progress, use CIFAR-10 first.
- If you need stronger report quality with real images, use Intel Image Classification.
- If your assignment requires pixel-level output, use Oxford-IIIT Pet segmentation.

## Folder Format For This Skeleton

When using your own images with `dataset.name: imagefolder`, use:

```text
data/raw/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
  test/                  # optional but recommended
    class_a/*.jpg
    class_b/*.jpg
```
