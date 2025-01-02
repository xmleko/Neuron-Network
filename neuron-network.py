import os
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import shutil
import plotly.graph_objects as go

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import itertools

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

np.set_printoptions(precision=6, suppress=True)

zip_file = 'dataset.zip'
output_dir = 'dataset'

if not os.path.exists(output_dir):  
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Dataset został rozpakowany do folderu: {output_dir}")
else:
    print(f"Folder {output_dir} już istnieje.")

"""Przygotowanie zbioru"""

base_dir = './dataset/dataset'
raw_no_of_files = {}
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y','Z']
for i in classes:
    raw_no_of_files[i] = len(os.listdir(os.path.join(base_dir, i)))

raw_no_of_files.items()


data_dir = './images'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
for directory in (train_dir, test_dir):
    if not os.path.exists(directory):
        os.mkdir(directory)

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
train_dirs = {}
test_dirs = {}
for letter in letters:
    train_dirs[letter] = os.path.join(train_dir, letter)
    test_dirs[letter] = os.path.join(test_dir, letter)

for directory in itertools.chain(train_dirs.values(), test_dirs.values()):
    if not os.path.exists(directory):
        os.mkdir(directory)

fnames = {}
for i in letters:
  fnames[i] = os.listdir(os.path.join(base_dir, i))

#fnames["A"]

min_size = min(len(fnames["A"]), len(fnames["B"]), len(fnames["C"]), len(fnames["D"]), len(fnames["E"]), len(fnames["F"]), len(fnames["G"]),len(fnames["H"]),len(fnames["I"]),
                   len(fnames["J"]),len(fnames["K"]),len(fnames["L"]),len(fnames["M"]),len(fnames["N"]),len(fnames["O"]),len(fnames["P"]),len(fnames["Q"]),len(fnames["R"]),
                   len(fnames["S"]),len(fnames["T"]),len(fnames["U"]),len(fnames["V"]),len(fnames["W"]),len(fnames["X"]),len(fnames["Y"]),len(fnames["Z"]))

print(min_size)

train_size = int(np.floor(0.85 * min_size))
test_size = min_size - train_size

train_idx = train_size
test_idx = train_size + test_size



for letter in letters:
    for i, fname in enumerate(fnames[letter]):
        src = os.path.join(base_dir, letter, fname)
        if i <= train_idx:  
            dst = os.path.join(train_dirs[letter], fname)
        elif train_idx < i < test_idx:  
            dst = os.path.join(test_dirs[letter], fname)

        shutil.copyfile(src, dst)

print('A - zbiór treningowy', len(os.listdir(train_dirs["A"])))
print('A - zbiór testowy', len(os.listdir(test_dirs["A"])))

print(train_dirs["A"].count)

"""Eksploracja danych"""

idx = 6
class_letter = "A"
# np. fnames["A"] = ["img1.jpg", "img2.jpg", ...]

names_mapping = fnames[class_letter]

if idx < len(names_mapping):
    img_path = os.path.join(data_dir + "/train", class_letter, names_mapping[idx])

    if os.path.exists(img_path):
        # Ładowanie obrazka
        img = image.load_img(img_path)

        # Wyświetl obrazek
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.grid(False)
        plt.axis(False)
        plt.show()
        # Kształt obrazka
        img_array = np.array(img)
        print(img_array.shape)
    else:
        print(f"Plik nie istnieje: {img_path}")
else:
    print(f"Nieprawidłowy indeks: {idx}")

"""Augmentacja danych

"""

train_datagen = ImageDataGenerator(
    rotation_range=5,      # zakres kąta o który losowo zostanie wykonany obrót obrazów
    rescale=1./255.,
    width_shift_range=0.05,  # pionowe przekształcenia obrazu
    height_shift_range=0.05, # poziome przekształcenia obrazu
  #  shear_range=0.2,        # zares losowego przycianania obrazu
    zoom_range=0.1,         # zakres losowego przybliżania obrazu
    horizontal_flip=False,   # losowe odbicie połowy obrazu w płaszczyźnie poziomej
    vertical_flip=False,
    fill_mode='nearest'     # strategia wypełniania nowo utworzonych pikseli, które mogą powstać w wyniku przekształceń
)

test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                   target_size=(128, 128),
                                                   batch_size=32,
                                                   shuffle=True,
                                                   class_mode='categorical')


test_generator = test_datagen.flow_from_directory(directory=test_dir,
                                                   target_size=(128, 128),
                                                   batch_size=32,
                                                   class_mode='categorical')


def display_augmented_images(directory, idx, train_datagen):

   # Wyświetla przykładowe obrazy uzyskane za pomocą augmentacji danych.
    directory = './images/train/J'
    fnames = [os.path.join(directory, fname) for fname in os.listdir(directory)]

    img_path = fnames[idx]
    img = image.load_img(img_path, target_size=(32, 32))

    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)  

    i = 1
    plt.figure(figsize=(16, 8))
    for batch in train_datagen.flow(x, batch_size=1):  
        plt.subplot(3, 4, i)
        plt.grid(False)
        plt.imshow(image.array_to_img(batch[0]))  
        i += 1
        if i > 12:  # Pokaż tylko 12 obrazów
            break

    plt.show()


display_augmented_images(fnames["A"], 6, train_datagen = train_datagen)

"""Budowa modelu"""

batch_size = 1
import math

steps_per_epoch = math.ceil(train_generator.samples / train_generator.batch_size)
test_steps = math.ceil(test_generator.samples / test_generator.batch_size)

# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
# conv_base.trainable = False

# def print_layers(model):
#     for layer in model.layers:
#         print(f'layer_name: {layer.name:13} trainable: {layer.trainable}')

# print_layers(conv_base)

# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False

# print_layers(conv_base)


# model = Sequential()
# model.add(layers.Flatten())
# model.add(layers.Dense(units=256, activation='relu', kernel_regularizer='l2'))
# model.add(Dropout(0.5))
# model.add(layers.Dense(units=26, activation='softmax'))


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())


model.add(Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(units=256, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(layers.Dense(units=26, activation='softmax'))



adam = Adam(learning_rate=1e-5)  
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-5),
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# EarlyStopping - zatrzymuje trening, jeśli nie ma poprawy przez 3 epoki
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitorujemy dokładność walidacyjną
    patience=5,               # Pozwól na 3 epoki bez poprawy
    restore_best_weights=True  
)


checkpoint = ModelCheckpoint(
    'best_model.keras',           
    monitor='val_accuracy',    
    save_best_only=True,
    mode='max',
    verbose=1
)



history = model.fit(
    train_generator,                   # generator treningowy danych
    steps_per_epoch=steps_per_epoch,    # ilość wsadów na epokę
    epochs=50,                          # liczba epok
    validation_data=test_generator,     # dane walidacyjne
    validation_steps=test_steps,        # ilość wsadów walidacyjnych
    callbacks=[early_stopping, checkpoint]  # Dodanie callbacków
)


# import torch

# print("Number of GPU: ", torch.cuda.device_count())
# print("GPU Name: ", torch.cuda.get_device_name())


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)







# # Załaduj model
# model = load_model('best_model.h5')

# # Ścieżka do obrazu
# image_path = './a.jpg'

# # Wstępne przetwarzanie obrazu
# img = load_img(image_path, target_size=(128, 128))  # Dopasowanie do rozmiaru wejścia modelu
# img_array = img_to_array(img)  # Zamiana na tablicę NumPy
# img_array = img_array / 255.   # Normalizacja (do zakresu 0-1)
# img_array = np.expand_dims(img_array, axis=0)  # Dodanie wymiaru wsadu (batch_size=1)

# # Predykcja modelu
# predictions = model.predict(img_array)

# # Znalezienie najbardziej prawdopodobnej klasy
# predicted_class_index = np.argmax(predictions)  # Indeks z najwyższym prawdopodobieństwem

# # Wyświetlenie wyników
# print(f"Przewidywana klasa: {predicted_class_index}")