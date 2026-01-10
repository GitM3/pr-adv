# Goal: PhenoBench Semantic Segmentation for plants and weeds.
![Epoch 16 preview](figures/epoch_016.png)
## Overview
This project implements a TensorFlow U-Net for semantic leaf segmentation on [PhenoBench](https://www.phenobench.org/benchmarks.html) dataset
, following the PhenoBench [devkit data pipeline](https://github.com/PRBonn/phenobench/blob/main/phenobench_tutorial.ipynb) and general TensorFlow segmentation [tutorial](https://www.tensorflow.org/tutorials/images/segmentation). The tutorial uses a MobileNetV2 backbone, but as a novel contribution this project implements [MobileNetV3Small](https://arxiv.org/pdf/1905.02244) for a lightweight encoder with better [performance](https://arxiv.org/html/2505.03303v1), with U-Net skip connections and a decoder modeled after the [pix2pix](https://github.com/tensorflow/examples/blob/b5a8265e0b655001eaa859e7bd8ac9b4e03f3ce8/tensorflow_examples/models/pix2pix/pix2pix.py#L220C66-L220C71) upsampling blocks. Evaluation uses the PhenoBench devkit for metrics and early stopping to prevent overfitting.

The segmentation classes: `1 crop, 2 weed, 3 partial crop, 4 partial weeds`
## Architecture

![MobileNetV3 U-Net wiring](figures/model_v3.png)

The inputs to the UNet is normalised to `[-1,1]` to match MobileNetV3 pre-processing expectations. The backbone however is freezed during training for transfer learning. 

Since the model follows UNet architecture, skip connections are made at downsampling strides from the encoder to the decorder (tasked with upsampling). To find the correct layers, this article was [referenced](https://ieeexplore.ieee.org/document/9522652) (See figure below) but in the end using the keras `model.summary()` to identify the downsampling layers and matching them in the upsampling ladder (16→32→64→128→256) worked. 

  <img width="820" height="390" alt="image" src="https://github.com/user-attachments/assets/3223227d-056f-4130-8c16-7df070c02427" />

The upsampling blocks uses transposed convolution, batch-norm and then ReLu based on pix2pix suggested in the tutorial.

Finally for validation, the phenobench requires converting the predicted mask to a specific format and saved in a prediction directory that the devkit then uses to calculate validation metrics. 

The PhenoBench evaluation uses per-class Intersection-over-Union (IoU) for soil, crop, and weed, plus mean IoU (mIoU). Scores are reported as percentages.
```math
IoU_c = \\frac{|P_c \\cap G_c|}{|P_c \\cup G_c|} = \\frac{TP_c}{TP_c + FP_c + FN_c}
```

```math
mIoU = \\frac{1}{C} \\sum_{c=1}^{C} IoU_c
```

Where `P_c` and `G_c` are the predicted and ground-truth pixels for class `c`, and `C=3` for soil, crop, weed.
## Results
The training curves show training and validation loss decreasing overall. Training pixel accuracy decreases while validation pixel accuracy increases.

![Training curves](figures/training_curves.png)


The validation can be done by supplying the checkpoint weights to the validation script:
- `python validate.py  --weights weights/2026_01_07_11_17/weights_epoch_010.weights.h5`
The final semantic metrics are:
```
  soil: 98.9
  crop: 90.45
  weed: 46.84
  mIoU: 78.73
```
Interestingly, the effect of augmentation and using a smaller training set can be seen (these results were taken during implementing the validation script):
```
Semantic metrics for no augmentation and 500 training samples:
  soil: 97.6
  crop: 76.06
  weed: 20.47
  mIoU: 64.71
```
This showcases the importance of augmentation and a large dataset.
This project's model performs worse when compared to the model implemented by the dataset authors where their metrics are:
```
 soil: 99.28
 crop: 94.30
 weed: 64.37
 mIoU: 85.97
```
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
