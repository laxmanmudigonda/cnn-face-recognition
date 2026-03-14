import os
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

dataset = "dataset"

data = []
labels = []

for person in os.listdir(dataset):

    person_path = os.path.join(dataset, person)

    # Skip files like pairs.txt
    if not os.path.isdir(person_path):
        continue

    for img in os.listdir(person_path):

        img_path = os.path.join(person_path, img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, (100,100))

        data.append(image)
        labels.append(person)

data = np.array(data) / 255.0
data = data.reshape(-1,100,100,1)

le = LabelEncoder()
labels = le.fit_transform(labels)

labels = to_categorical(labels)

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(labels.shape[1],activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(data, labels, epochs=10)

model.save("face_model.h5")

print("Model trained successfully!")