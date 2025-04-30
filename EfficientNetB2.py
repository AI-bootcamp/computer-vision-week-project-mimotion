!pip install kagglehub tensorflow

import kagglehub
from pathlib import Path

# Download data
path = kagglehub.dataset_download("trushraut18/colour-classification")
data_dir = Path(path) / "Data"  # adjust if you had to dive into "Data"
print("Data directory:", data_dir)

import tensorflow as tf

batch_size = 32
img_size = (260, 260)

# datasets
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir / "train",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
)
raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir / "validation",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
)

# class names
class_names = raw_train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    raw_train_ds
    .shuffle(512)           # shuffle buffer
    .cache()
    .prefetch(AUTOTUNE)
    .ignore_errors()
)

val_ds = (
    raw_val_ds
    .cache()
    .prefetch(AUTOTUNE)
    .ignore_errors()
)

from tensorflow.keras import layers, applications, Sequential

# Data augmentation pipeline
data_augmentation = Sequential([
    layers.Rescaling(1./255, input_shape=(*img_size, 3)),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.RandomFlip(),
    layers.RandomRotation(0.1),
])

# Load & freeze EfficientNetB2 base
base_model = applications.EfficientNetB2(
    input_shape=(*img_size, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Build end-to-end model
model = Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax"),
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# compute steps per epoch manually
steps_per_epoch = 2700 // batch_size   
validation_steps = 300 // batch_size   

history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=5,
)
# Unfreeze last ~30 layers of EfficientNetB2
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

fine_history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=3,
)
loss, accuracy = model.evaluate(val_ds, steps=validation_steps)
print(f"Validation accuracy after fine-tuning: {accuracy:.4%}")


'''

Getting stuck at ~33 % ==> the network isn’t actually learning anything, one ofthe causes is
-Augmentations that destroy color information
-Only training the top Dense layer
-A frozen backbone + only 5 epochs on the head often isn’t enough 
-Too much regularization
-Learning rate not tuned

'''
