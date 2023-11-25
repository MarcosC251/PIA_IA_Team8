from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
import glob
from PIL import Image

folder_abdomen = './AbdomenCT'
folder_breast = './BreastMRI'
folder_cxr = './CXR'
folder_chest = './ChestCT'
folder_hand = './Hand'
folder_head = './HeadCT'

image_files_abdomen = glob.glob(folder_abdomen + '/*.jpeg')
image_files_breast = glob.glob(folder_breast + '/*.jpeg')
image_files_cxr = glob.glob(folder_cxr + '/*.jpeg')
image_files_chest = glob.glob(folder_chest + '/*.jpeg')
image_files_hand = glob.glob(folder_hand + '/*.jpeg')
image_files_head = glob.glob(folder_head + '/*.jpeg')

images_abdomen = []
labels_abdomen = []
images_breast = []
labels_breast = []
images_cxr = []
labels_cxr = []
images_chest = []
labels_chest = []
images_hand = []
labels_hand = []
images_head = []
labels_head = []
class_names = ['abdomen', 'breast','cxr', 'chest', 'hand', 'head']
# Resto de tu código hasta antes de la parte de procesamiento de imágenes...

images = []
for image in images_abdomen:
    images.append(np.array(image))

for image in images_breast:
    images.append(np.array(image))

for image in images_cxr:
    images.append(np.array(image))

for image in images_chest:
    images.append(np.array(image))

for image in images_hand:
    images.append(np.array(image))

for image in images_head:
    images.append(np.array(image))

labels = np.concatenate((np.ones(len(images_abdomen)),
                         np.ones(len(images_breast)),
                         np.ones(len(images_cxr)),
                         np.ones(len(images_chest)),
                         np.ones(len(images_hand)),
                         np.ones(len(images_head))))

# Split de los datos de entrenamiento y prueba
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Redimensionamiento de las imágenes
new_width = 64
new_height = 64

train_images_rgb = []
train_labels_filtered = []
for image, label in zip(train_images, train_labels):
    resized_image = cv2.resize(image, (new_width, new_height))
    if resized_image.shape == (new_height, new_width, 3):
        train_images_rgb.append(resized_image)
        train_labels_filtered.append(label)

test_images_rgb = []
test_labels_filtered = []
for image, label in zip(test_images, test_labels):
    resized_image = cv2.resize(image, (new_width, new_height))
    if resized_image.shape == (new_height, new_width, 3):
        test_images_rgb.append(resized_image)
        test_labels_filtered.append(label)

# Conversión a matrices numpy
train_images_rgb = np.array(train_images_rgb)
train_labels_filtered = np.array(train_labels_filtered)
test_images_rgb = np.array(test_images_rgb)
test_labels_filtered = np.array(test_labels_filtered)

# Definición del modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(new_height, new_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(train_images_rgb, train_labels_filtered, epochs=10,
                    validation_data=(test_images_rgb, test_labels_filtered))
