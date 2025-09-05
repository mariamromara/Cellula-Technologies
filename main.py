### Import block
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import pandas as pd
#sub import
from tensorflow.python.keras.utils.version_utils import callbacks
from torchvision import datasets, transforms
from keras.callbacks import ModelCheckpoint

### Load images
# used later
disease_list = ("CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT")
img_size = (64,64)
batch_size = 32
# data acquisition
dataset_training = tf.keras.utils.image_dataset_from_directory(
    "Teeth_Dataset/Training",
    image_size=img_size,
    batch_size=batch_size)
dataset_validation = tf.keras.utils.image_dataset_from_directory(
    "Teeth_Dataset/Validation",
    image_size=img_size,
    batch_size=batch_size)
dataset_testing = tf.keras.utils.image_dataset_from_directory(
    "Teeth_Dataset/Testing",
    image_size=img_size,
    batch_size=batch_size)
# normalization
normalization_layer = tf.keras.layers.Rescaling(1./255)
dataset_training = dataset_training.map(lambda x, y: (normalization_layer(x), y))
dataset_validation = dataset_validation.map(lambda x, y: (normalization_layer(x), y))
dataset_testing = dataset_testing.map(lambda x, y: (normalization_layer(x), y))

### Building a model
#build
def classifier():
    teeth_classifier = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=7, activation='softmax')    ])
    return teeth_classifier
#Callbacks
callbacks = ModelCheckpoint("teeth_classifier.keras", save_best_only=True)
#complier
teeth_classifier = classifier()
teeth_classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

###Training the model
history = teeth_classifier.fit(x=dataset_training, epochs=100, validation_data=dataset_validation,
                               callbacks=[callbacks])
test_loss, test_accuracy = teeth_classifier.evaluate(dataset_testing)
print("Test Accuracy:", test_accuracy)
# visualization
loss_accuracy_metrics_df = pd.DataFrame(teeth_classifier.history.history)
loss_accuracy_metrics_df.plot(figsize=(10,5))
plt.show()


###Testing
