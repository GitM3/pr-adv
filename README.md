# Goal: PhenoBench Semantic Segmentation for plants and weeds.
![Epoch 16 preview](figures/epoch_016.png)
## Overview
This project implements a TensorFlow U-Net for semantic leaf segmentation on [PhenoBench](https://www.phenobench.org/benchmarks.html) dataset
, following the PhenoBench [devkit data pipeline](https://github.com/PRBonn/phenobench/blob/main/phenobench_tutorial.ipynb) and general TensorFlow segmentation [tutorial](https://www.tensorflow.org/tutorials/images/segmentation). The tutorial uses a MobileNetV2 backbone, but as a novel contribution this project implements [MobileNetV3Small](https://arxiv.org/pdf/1905.02244) for a lightweight encoder with better [performance](https://arxiv.org/html/2505.03303v1), with U-Net skip connections and a decoder modeled after the [pix2pix](https://github.com/tensorflow/examples/blob/b5a8265e0b655001eaa859e7bd8ac9b4e03f3ce8/tensorflow_examples/models/pix2pix/pix2pix.py#L220C66-L220C71) upsampling blocks. Evaluation uses the PhenoBench devkit for metrics and early stopping to prevent overfitting.

The segmentation classes: `1 crop, 2 weed, 3 partial crop, 4 partial weeds`
The model:
![MobileNetV3 U-Net wiring](figures/model_v3.png)
## Performance
The training curves show training and validation loss decreasing overall. Training pixel accuracy decreases while validation pixel accuracy increases.

![Training curves](figures/training_curves.png)

- `python validate.py  --weights weights/2026_01_07_11_17/weights_epoch_010.weights.h5`
Semantic metrics:
  soil: 98.9
  crop: 90.45
  weed: 46.84
  mIoU: 78.73

## Architecture

Using 5 connections (Inspected model.summary())
- activation (256x256)
- re_lu (128x128)
- re_lu_3 (64x64)
- activation_2 (32x32)
- activation_17 

- `python validate.py  --weights weights/2026_01_07_11_17/weights_epoch_010.weights.h5`

- Results for no augmentation and 500 Samples:
Semantic metrics:
  soil: 97.6
  crop: 76.06
  weed: 20.47
  mIoU: 64.71

## Upgrading to MobileNetV3
  <img width="820" height="390" alt="image" src="https://github.com/user-attachments/assets/3223227d-056f-4130-8c16-7df070c02427" />
 1) Used check_model to check outputs at downsampling layers.
 2) Match upscaling layers to same spatial resolution. 256->128->64->32->16
 3) Normalize input to [-1,1] (Dataset was 0,1)

## Sources
- PhenoBench benchmark: https://www.phenobench.org/benchmarks.html
- PhenoBench tutorial notebook: https://github.com/PRBonn/phenobench/blob/main/phenobench_tutorial.ipynb
- Segmentation overview: https://arxiv.org/pdf/1910.07655
- TensorFlow segmentation tutorial (MobileNet U-Net): https://www.tensorflow.org/tutorials/images/segmentation
- MobileNetV3 balance discussion: https://arxiv.org/html/2505.03303v1
- Pix2pix upsample block: https://github.com/tensorflow/examples/blob/b5a8265e0b655001eaa859e7bd8ac9b4e03f3ce8/tensorflow_examples/models/pix2pix/pix2pix.py#L220C66-L220C71
- MobileNetV3 reference: https://ieeexplore.ieee.org/document/9522652
- MobileNetV3 original paper: https://arxiv.org/pdf/1905.02244
- Early stopping overview: https://cyborgcodes.medium.com/what-is-early-stopping-in-deep-learning-eeb1e710a3cf
