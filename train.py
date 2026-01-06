import datetime
import math
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from load import (
    DATA_DIR,
    IMAGE_SIZE,
    OUTPUT_CHANNELS,
    create_dataset,
    unet_model,
)


SAMPLES = 10
BATCH_SIZE = 5
EPOCHS = 25
PREVIEW_DIR = os.path.join(
    "output", f"{datetime.datetime.now():%Y_%m_%d_%H_%M}-preview"
)


class PreviewCallback(tf.keras.callbacks.Callback):
    def __init__(self, sample, output_dir):
        super().__init__()
        self.sample_image, self.sample_mask = sample
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model(self.sample_image, training=False)
        pred_mask = tf.argmax(pred, axis=-1)

        image = tf.keras.utils.array_to_img(self.sample_image[0])
        true_mask = tf.squeeze(self.sample_mask[0])
        pred_mask = tf.squeeze(pred_mask[0])
        true_unique = tf.unique(tf.reshape(true_mask, [-1])).y.numpy().tolist()
        pred_unique = tf.unique(tf.reshape(pred_mask, [-1])).y.numpy().tolist()
        pred_min = float(tf.reduce_min(pred).numpy())
        pred_max = float(tf.reduce_max(pred).numpy())
        pred_mean = float(tf.reduce_mean(pred).numpy())
        print(
            f"epoch {epoch + 1} debug:"
            f" true_unique={sorted(true_unique)}"
            f" pred_unique={sorted(pred_unique)}"
            f" logits(min/mean/max)={pred_min:.4f}/{pred_mean:.4f}/{pred_max:.4f}"
        )

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax in axes:
            ax.axis("off")
        axes[0].set_title("Input")
        axes[0].imshow(image)
        axes[1].set_title("True")
        axes[1].imshow(true_mask, cmap="viridis", vmin=0)
        axes[2].set_title("Pred")
        axes[2].imshow(pred_mask, cmap="viridis", vmin=0)

        path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}.png")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, depth=OUTPUT_CHANNELS)
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2])
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice)


def combo_loss(y_true, y_pred):
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return 0.5 * ce(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)


def main():
    dataset = create_dataset(
        DATA_DIR, split="train", shuffle=False, image_size=IMAGE_SIZE
    ).take(SAMPLES)
    dataset = dataset.cache().batch(BATCH_SIZE).repeat()

    preview_sample = next(
        iter(
            create_dataset(
                DATA_DIR, split="train", shuffle=False, image_size=IMAGE_SIZE
            )
            .take(1)
            .batch(1)
        )
    )

    model = unet_model(OUTPUT_CHANNELS, IMAGE_SIZE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=combo_loss,
    )

    steps_per_epoch = max(1, math.ceil(SAMPLES / BATCH_SIZE))

    model.fit(
        dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[PreviewCallback(preview_sample, PREVIEW_DIR)],
        verbose=2,
    )


if __name__ == "__main__":
    main()
