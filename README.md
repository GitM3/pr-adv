# Goal
- Implement semantic leaf segmentation model.
- PhenoBench: https://www.phenobench.org/benchmarks.html
## Milestones
- [X] Load dataset into tensorflow format.
- [X] Implement basic U-Net.
- [X] Overfit 5 images to see if works
- [X] Train for 1-2 epochs check functionality, etc.
- [X] Visualise training.
- [X] Evaluate using eval toolkit
- [ ] Try MobileNetV3 integration
## DataLoading
- https://github.com/PRBonn/phenobench/blob/main/phenobench_tutorial.ipynb
- Classes: `1 crop, 2 weed, 3 partial crop, 4 partial weeds`
## Segmentation Models
- [Overview](https://arxiv.org/pdf/1910.07655)
Sticking with the tensorflow implementation [tutorial](https://www.tensorflow.org/tutorials/images/segmentation), I choose the MobileNet, but I choose V3 as other light convolution nets seem not provide that much benefit (Here, it says MobileNetV3 gives best balance, https://arxiv.org/html/2505.03303v1)

Using 5 connections (Inspected model.summary())
- activation (256x256)
- re_lu (128x128)
- re_lu_3 (64x64)
- activation_2 (32x32)
- activation_17 
The above did not work, I need to inspect model more carefully.

Upsampling from their pix2pix example implemented [here](https://github.com/tensorflow/examples/blob/b5a8265e0b655001eaa859e7bd8ac9b4e03f3ce8/tensorflow_examples/models/pix2pix/pix2pix.py#L220C66-L220C71).
# Validation
Using devkit:
- `python validate.py  --weights weights/2026_01_07_11_17/weights_epoch_010.weights.h5`

- Results for no augmentation and 500 Samples:
Semantic metrics:
  soil: 97.6
  crop: 76.06
  weed: 20.47
  mIoU: 64.71

  # Upgrading to MobileNetV3
  <img width="820" height="390" alt="image" src="https://github.com/user-attachments/assets/3223227d-056f-4130-8c16-7df070c02427" />
 1) Used check_model to check outputs at downsampling layers.
 2) Match upscaling layers to same spatial resolution. 256->128->64->32->16
 3) Normalize input to [-1,1] (Dataset was 0,1)

 - This as reference: https://ieeexplore.ieee.org/document/9522652
- And original paper: https://arxiv.org/pdf/1905.02244
- Early stopping: https://cyborgcodes.medium.com/what-is-early-stopping-in-deep-learning-eeb1e710a3cf
