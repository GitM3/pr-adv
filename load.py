from phenobench import PhenoBench
import matplotlib.pyplot as plt
from phenobench.visualization import draw_semantics
import numpy as np
import os
import tensorflow as tf
from pprint import pprint

DATA_DIR = os.path.expanduser("~/Development/08_ADV/PhenoBench")
IMAGE_SIZE = (512, 512)
OUTPUT_CHANNELS = 5
SKIP_LAYER_NAMES = [
    "activation",     # 256x256
    "re_lu",          # 128x128
    "re_lu_3",        # 64x64
    "activation_2",   # 32x32
    "activation_17",  # 16x16
]

def _iter_samples(data, image_key="image", mask_key="semantics"):
    for idx in range(len(data)):
        sample = data[idx]
        image = np.asarray(sample[image_key], dtype=np.uint8)
        mask = np.asarray(sample[mask_key], dtype=np.uint16)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        yield image, mask

def _preprocess_sample(image, mask, image_size):
    image = tf.ensure_shape(image, (None, None, 3))
    mask = tf.ensure_shape(mask, (None, None))
    mask = tf.expand_dims(mask, axis=-1)
    image = tf.image.resize(image, image_size, method="bilinear")
    mask = tf.image.resize(
        mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.int32)
    return image, mask


def create_dataset(
    root_dir,
    split="train",
    mask_key="semantics",
    image_size=IMAGE_SIZE,
    shuffle=False, # TODO: Remember to toggle again after testing
    seed=69,
):
    data = PhenoBench(root_dir, split=split, target_types=[mask_key])
    output_signature = (
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(None, None), dtype=tf.uint16),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: _iter_samples(data, "image", mask_key),
        output_signature=output_signature,
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(1000, len(data)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        lambda image, mask: _preprocess_sample(image, mask, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset

# From pix2pix, no dropout, batch norm
def _upsample(filters, size):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding="same",
                kernel_initializer=initializer,
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ]
    )

def unet_model(output_channels, image_size=IMAGE_SIZE):
    inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(image_size[0], image_size[1], 3), include_top=False
    )
    #base_model.summary() 
    base_model_outputs = [base_model.get_layer(name).output for name in SKIP_LAYER_NAMES]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    up_stack = [
        _upsample(512, 3),
        _upsample(256, 3),
        _upsample(128, 3),
        _upsample(64, 3),
    ]

    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding="same"
    )
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def display_training_sample(dataset):
    for image, mask in dataset.take(1):
        if image.shape.rank == 4:
            image = image[0]
            mask = mask[0]

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(tf.keras.utils.array_to_img(image))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("True Mask")
        plt.imshow(tf.squeeze(mask), cmap="viridis", vmin=0)
        plt.axis("off")
        plt.show()


dataset = create_dataset(DATA_DIR)
display_training_sample(dataset)
model = unet_model(OUTPUT_CHANNELS, IMAGE_SIZE)
model.summary()
