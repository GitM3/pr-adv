# Goal
- Implement semantic leaf segmentation model.
- PhenoBench: https://www.phenobench.org/benchmarks.html
## Milestones
- [ ] Load dataset into tensorflow format.
- [ ] Implement basic U-Net.
- [ ] Overfit 5 images to see if works
- [ ] Train for 1-2 epochs check functionality, etc.
- [ ] Visualise training.
- [ ] Evaluate using eval toolkit
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

Upsampling from their pix2pix example implemented [here](https://github.com/tensorflow/examples/blob/b5a8265e0b655001eaa859e7bd8ac9b4e03f3ce8/tensorflow_examples/models/pix2pix/pix2pix.py#L220C66-L220C71).

