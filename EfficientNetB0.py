!pip install kagglehub tensorflow

import kagglehub
from pathlib import Path

# Download the “Colour Classification” dataset
path = kagglehub.dataset_download("trushraut18/colour-classification")
data_dir = Path(path)

print("Data directory:", data_dir)

import tensorflow as tf

batch_size = 32
img_size   = (224, 224)

# Load data
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
)
raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
)

# class names
class_names = raw_train_ds.class_names
print("Classes:", class_names)

train_ds = raw_train_ds.ignore_errors()
val_ds   = raw_val_ds.ignore_errors()

# Check batchs
print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val   batches:", tf.data.experimental.cardinality(val_ds).numpy())

from tensorflow.keras import layers, applications, Sequential

data_augmentation = Sequential([
    layers.Rescaling(1./255, input_shape=(*img_size, 3)),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.RandomFlip(),
    layers.RandomRotation(0.1),
])

base_model = applications.EfficientNetB0(
    input_shape=(*img_size, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

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

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
)

loss, accuracy = model.evaluate(val_ds)
print(f"Validation accuracy: {accuracy:.4%}")

'''
Epoch 1/5
     84/Unknown 278s 3s/step - accuracy: 0.3319 - loss: 1.1317/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least steps_per_epoch * epochs batches. You may need to use the .repeat() function when building your dataset.
  self._interrupted_warning()
84/84 ━━━━━━━━━━━━━━━━━━━━ 303s 3s/step - accuracy: 0.3318 - loss: 1.1316 - val_accuracy: 0.3333 - val_loss: 1.1050
Epoch 2/5
84/84 ━━━━━━━━━━━━━━━━━━━━ 242s 3s/step - accuracy: 0.3357 - loss: 1.1153 - val_accuracy: 0.3333 - val_loss: 1.1155
Epoch 3/5
84/84 ━━━━━━━━━━━━━━━━━━━━ 254s 3s/step - accuracy: 0.3400 - loss: 1.1152 - val_accuracy: 0.3333 - val_loss: 1.1053
Epoch 4/5
84/84 ━━━━━━━━━━━━━━━━━━━━ 244s 3s/step - accuracy: 0.3212 - loss: 1.1235 - val_accuracy: 0.3333 - val_loss: 1.1500
Epoch 5/5
84/84 ━━━━━━━━━━━━━━━━━━━━ 250s 3s/step - accuracy: 0.3293 - loss: 1.1260 - val_accuracy: 0.3333 - val_loss: 1.1174

very low accuracy cause the model is classifying, which produces low Accuracy, so we have to switch to a detection model, not a classification model.
'''

