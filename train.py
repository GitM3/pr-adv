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
    create_train_val,
    unet_model,
)


TRAIN_SAMPLES = 1407
VAL_SAMPLES = 772
BATCH_SIZE = 8
EPOCHS = 30
PREVIEW_DIR = os.path.join(
    "output", f"{datetime.datetime.now():%Y_%m_%d_%H_%M}-preview"
)
WEIGHTS_DIR = os.path.join(
    "weights", f"{datetime.datetime.now():%Y_%m_%d_%H_%M}"
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

        true_mask = tf.squeeze(self.sample_mask, axis=-1)
        true_unique = tf.unique(tf.reshape(true_mask[0], [-1])).y.numpy().tolist()
        pred_unique = tf.unique(tf.reshape(pred_mask[0], [-1])).y.numpy().tolist()
        pred_min = float(tf.reduce_min(pred).numpy())
        pred_max = float(tf.reduce_max(pred).numpy())
        pred_mean = float(tf.reduce_mean(pred).numpy())
        print(
            f"epoch {epoch + 1} debug:"
            f" true_unique={sorted(true_unique)}"
            f" pred_unique={sorted(pred_unique)}"
            f" logits(min/mean/max)={pred_min:.4f}/{pred_mean:.4f}/{pred_max:.4f}"
        )

        rows = int(self.sample_image.shape[0] or 1)
        rows = min(2, rows)
        fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
        if rows == 1:
            axes = [axes]
        for row in range(rows):
            for ax in axes[row]:
                ax.axis("off")
            if row == 0:
                axes[row][0].set_title("Input")
                axes[row][1].set_title("True")
                axes[row][2].set_title("Pred")
            image = tf.keras.utils.array_to_img(self.sample_image[row])
            axes[row][0].imshow(image)
            axes[row][1].imshow(true_mask[row], cmap="viridis", vmin=0)
            axes[row][2].imshow(pred_mask[row], cmap="viridis", vmin=0)

        path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}.png")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def compute_steps(sample_count, batch_size):
    return max(1, math.ceil(sample_count / batch_size))


def save_training_curves(history, output_dir):
    metrics = history.history
    epochs = range(1, len(metrics.get("loss", [])) + 1)
    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, metrics.get("loss", []), label="train")
    if "val_loss" in metrics:
        axes[0].plot(epochs, metrics["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if "pixel_acc" in metrics or "val_pixel_acc" in metrics:
        axes[1].plot(epochs, metrics.get("pixel_acc", []), label="train")
        if "val_pixel_acc" in metrics:
            axes[1].plot(epochs, metrics["val_pixel_acc"], label="val")
        axes[1].set_title("Pixel Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
    else:
        axes[1].axis("off")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path)
    plt.close(fig)


def main():
    train, val = create_train_val(
        DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        train_limit=TRAIN_SAMPLES,
        val_limit=VAL_SAMPLES,
    )

    preview_sample = next(
        iter(
            create_dataset(
                DATA_DIR, split="train", shuffle=False, image_size=IMAGE_SIZE
            )
            .take(2)
            .batch(2)
        )
    )

    model = unet_model(OUTPUT_CHANNELS, IMAGE_SIZE, backbone="mobilenetv3")
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="pixel_acc")],
    )

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(WEIGHTS_DIR, "weights_epoch_{epoch:03d}.weights.h5"),
        save_weights_only=True,
        monitor="val_loss",
        save_best_only=False,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        restore_best_weights=True,
    )

    steps_per_epoch = compute_steps(TRAIN_SAMPLES, BATCH_SIZE)
    validation_steps = compute_steps(VAL_SAMPLES, BATCH_SIZE)

    history = model.fit(
        train,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val,
        validation_steps=validation_steps,
        callbacks=[PreviewCallback(preview_sample, PREVIEW_DIR), checkpoint, early_stop],
        verbose=2,
    )
    save_training_curves(history, PREVIEW_DIR)


if __name__ == "__main__":
    main()
