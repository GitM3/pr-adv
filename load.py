import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from phenobench import PhenoBench
from phenobench.visualization import draw_semantics

DATA_DIR = os.path.expanduser("./PhenoBench")
IMAGE_SIZE = (512, 512)
OUTPUT_CHANNELS = 5
MOBILENETV2_SKIP_LAYER_NAMES = [
    "block_1_expand_relu",  
    "block_3_expand_relu", 
    "block_6_expand_relu",
    "block_13_expand_relu",
    "block_16_project", 
]
MOBILENETV3_SKIP_LAYER_NAMES = [
    "activation", 
    "expanded_conv_project_bn",
    "expanded_conv_2_add", 
    "expanded_conv_7_add",
    "activation_17",
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
    shuffle=False,  # TODO: Remember to toggle again after testing
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


def unet_model(output_channels, image_size=IMAGE_SIZE, backbone="mobilenetv2"):
    inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))

    if backbone == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(image_size[0], image_size[1], 3),
            include_top=False
        )
        skip_layer_names = MOBILENETV2_SKIP_LAYER_NAMES
        encoder_inputs = inputs
    elif backbone == "mobilenetv3":
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=(image_size[0], image_size[1], 3),
            include_top=False,
            include_preprocessing=False,
        )
        skip_layer_names = MOBILENETV3_SKIP_LAYER_NAMES
        encoder_inputs = tf.keras.layers.Rescaling(
            2.0, offset=-1.0, name="mobilenetv3_rescale" # 0,1 to -1,1
        )(inputs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}") # TODO: try others if there is time

    base_model_outputs = [
        base_model.get_layer(name).output for name in skip_layer_names
    ]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    up_stack = [
        _upsample(512, 3),
        _upsample(256, 3),
        _upsample(128, 3),
        _upsample(64, 3),
    ]

    skips = down_stack(encoder_inputs)
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


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def create_train_val(
    root_dir,
    image_size=IMAGE_SIZE,
    batch_size=8,
    buffer_size=1000,
    train_limit=None,
    val_limit=None,
    seed=42,
):
    train = create_dataset(
        root_dir,
        split="train",
        image_size=image_size,
        shuffle=False,
        seed=seed,
    )
    val = create_dataset(
        root_dir,
        split="val",
        image_size=image_size,
        shuffle=False,
        seed=seed,
    )

    if train_limit is not None:
        train = train.take(train_limit)
    if val_limit is not None:
        val = val.take(val_limit)

    train = (
        train.cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .repeat()
        .map(Augment(seed=seed), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val = val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train, val


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


if __name__ == "__main__":
    dataset = create_dataset(DATA_DIR)
    display_training_sample(dataset)
    model = unet_model(OUTPUT_CHANNELS, IMAGE_SIZE)
    model.summary()
