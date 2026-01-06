import tensorflow as tf
model = tf.keras.applications.MobileNetV3Small(input_shape=(512,512,3),include_top=False)
model.summary()
for l in model.layers:
    print(f"Layer name: {l.name}")
    print(f"Layer output shape: {l.output_shape}")
