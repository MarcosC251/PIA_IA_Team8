"""
Team Members:
Dario Contreras Solis               1953740
Marcos Ezequiel Cervantes Castro    1953168
Javier Osmar Covarrubias Bautista   1958896
Jaime Jimenez Suárez                2077627
Manuel Paredes Sánchez              1953821
 """



import tensorflow as tf
from keras import layers, models
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
import glob
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage.color import rgb2gray
from tensorflow import keras

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


AMOUNT_PER_CLASS = 1500

TOTAL_IMAGES = AMOUNT_PER_CLASS * 6

# Limitar los bucles a 100 elementos
for image_file in image_files_abdomen[:AMOUNT_PER_CLASS]:
    image = Image.open(image_file)
    images_abdomen.append(image)
    labels_abdomen.append("abdomen")

for image_file in image_files_breast[:AMOUNT_PER_CLASS]:
    image = Image.open(image_file)
    images_breast.append(image)
    labels_breast.append("breast")

for image_file in image_files_cxr[:AMOUNT_PER_CLASS]:
    image = Image.open(image_file)
    images_cxr.append(image)
    labels_cxr.append("cxr")

for image_file in image_files_chest[:AMOUNT_PER_CLASS]:
    image = Image.open(image_file)
    images_chest.append(image)
    labels_chest.append("chest")

for image_file in image_files_hand[:AMOUNT_PER_CLASS]:
    image = Image.open(image_file)
    images_hand.append(image)
    labels_hand.append("hand")



for image_file in image_files_head[:AMOUNT_PER_CLASS]:
    image = Image.open(image_file)
    images_head.append(image)
    labels_head.append("head")


plt.imshow(images_hand[40], cmap='gray')
plt.show()

plt.figure(figsize=(10, 10))

for i in range(600):
    plt.subplot(30, 20, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    images_to_show = 100

    if(i < images_to_show):
        plt.imshow(images_abdomen[i], cmap='gray')
        plt.xlabel(class_names[0])
    elif(i < images_to_show*2):
        plt.imshow(images_breast[i-images_to_show], cmap='gray')
        plt.xlabel(class_names[1])
    elif(i < images_to_show*3):
        plt.imshow(images_cxr[i-images_to_show * 2], cmap='gray')
        plt.xlabel(class_names[2])
    elif(i < images_to_show*4):
        plt.imshow(images_chest[i-images_to_show * 3], cmap='gray')
        plt.xlabel(class_names[3])
    elif(i < images_to_show*5):
        plt.imshow(images_hand[i-images_to_show * 4], cmap='gray')
        plt.xlabel(class_names[4])
    else:
        plt.imshow(images_head[i-images_to_show * 5], cmap='gray')
        plt.xlabel(class_names[5])
        breakpoint

plt.show()

# Encontrar la minima resolucion en ambas carpetas

min_width_abdomen = min(image.width for image in images_abdomen)
min_height_abdomen = min(image.height for image in images_abdomen)

min_width_breast = min(image.width for image in images_breast)
min_height_breast = min(image.height for image in images_breast)

min_width_cxr = min(image.width for image in images_cxr)
min_height_cxr = min(image.height for image in images_cxr)

min_width_chest = min(image.width for image in images_chest)
min_height_chest = min(image.height for image in images_chest)

min_width_hand = min(image.width for image in images_hand)
min_height_hand = min(image.height for image in images_hand)

min_width_head = min(image.width for image in images_head)
min_height_head = min(image.height for image in images_head)


print("Minima resolucion de abdomen: " + str(min_width_abdomen) + "x" + str(min_height_abdomen))
print("Minima resolucion de breast: " + str(min_width_breast) + "x" + str(min_height_breast))
print("Minima resolucion de cxr: " + str(min_width_cxr) + "x" + str(min_height_cxr))
print("Minima resolucion de chest: " + str(min_width_chest) + "x" + str(min_height_chest))
print("Minima resolucion de hand: " + str(min_width_hand) + "x" + str(min_height_hand))
print("Minima resolucion de head: " + str(min_width_head) + "x" + str(min_height_head))

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


labels = np.concatenate((np.ones(len(images_abdomen)), np.ones(len(images_breast)), np.ones(len(images_cxr)), np.ones(len(images_chest)), np.ones(len(images_hand)), np.ones(len(images_head))),axis=0)


train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

new_width = 64
new_height = 64

train_images_rgb = []
train_labels_filtered = []

for image, label in zip(train_images, train_labels):
    resized_image = cv2.resize(image, (new_width, new_height))
    train_images_rgb.append(resized_image)
    train_labels_filtered.append(label)

test_images_rgb = []
test_labels_filtered = []

for image, label in zip(test_images, test_labels):
    resized_image = cv2.resize(image, (new_width, new_height))
    test_images_rgb.append(resized_image)
    test_labels_filtered.append(label)

train_images_rgb = np.expand_dims(train_images_rgb, axis=-1)
test_images_rgb = np.expand_dims(test_images_rgb, axis=-1)

# Convertir a formato de arrays y normalizar durante el entrenamiento con ImageDataGenerator
train_images_rgb = np.array(train_images_rgb)
train_labels_filtered = np.array(train_labels_filtered)
test_images_rgb = np.array(test_images_rgb)
test_labels_filtered = np.array(test_labels_filtered)


train_labels_one_hot = to_categorical(train_labels_filtered, num_classes=6)
test_labels_one_hot = to_categorical(test_labels_filtered, num_classes=6)


datagen = ImageDataGenerator(rescale=1.0/255.0)

batch_size = 32

train_images_normalized = datagen.flow(train_images_rgb, train_labels_one_hot, batch_size=batch_size)
test_images_normalized = datagen.flow(test_images_rgb, test_labels_one_hot, batch_size=batch_size)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(new_height, new_width, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Ajustar el modelo a los datos de entrenamiento
history = model.fit(train_images_normalized, epochs=10, validation_data=test_images_normalized)

test_loss, test_acc = model.evaluate(test_images_normalized, verbose=2)
print('Accuracy en los datos de prueba:', test_acc)

# Graficar la precisión y la pérdida durante el entrenamiento
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'r', label='Precisión de entrenamiento')
plt.plot(epochs, val_accuracy, 'b', label='Precisión de validación')
plt.title('Precisión de entrenamiento y validación')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Pérdida de entrenamiento')
plt.plot(epochs, val_loss, 'b', label='Pérdida de validación')
plt.title('Pérdida de entrenamiento y validación')
plt.legend()
plt.show()



# Revisamos la predicción general y observamos su tipo

# In[159]:


predictions=model.predict(test_images_normalized)


# In[160]:


predictions[0]


# Hacemos una revisión del modelo

# In[161]:


# Guardar el modelo
model.save('modelo.h5')

# Cargar el modelo
loaded_model = tf.keras.models.load_model('modelo.h5')

# Realizar predicciones con el modelo cargado
predictions = loaded_model.predict(test_images_normalized)

# Convertir las predicciones a etiquetas (0 o 1)
predicted_labels = np.round(predictions).flatten()

# Imprimir las etiquetas predichas y las etiquetas reales
print('Etiquetas predichas:', predicted_labels)
print('Etiquetas reales:', test_labels_filtered)

def show_labels(labels, predicted_labels, images, ncols=5):
    n = len(labels)
    nrows = (n + ncols - 1) // ncols
    figsize = (20, 5 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.tight_layout()

    for i, ax in enumerate(axes.flat):
        if i < n:
            label = labels[i]
            predicted_label = predicted_labels[i]
            class_name = class_names[int(label)]
            predicted_class_name = class_names[int(predicted_label)]
            percentage = round(predictions[i][0] * 100, 2)
            is_correct = label == predicted_label

            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'True label: {class_name}\nPredicted: {predicted_class_name} ({percentage}%)\nCorrect: {"True" if is_correct else "False"}', color='green' if is_correct else 'red')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.show()



# Mostrar las etiquetas de forma visual y con porcentajes
show_labels(test_labels_filtered, predicted_labels, test_images_rgb)