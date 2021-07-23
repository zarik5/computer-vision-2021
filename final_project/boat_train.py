import os
import sys
import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, preprocessing as preproc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


# This function is used to import the selected images from the COCCO dataset with YOLO labels format
def load_images_from_folder(names_selected, folder_images, folder_txt):
    dataset = []
    labels = []
    for image_name in names_selected:
        try:
            img = cv2.imread(os.path.join(folder_images, image_name))
            rows, cols = img.shape[:2]
            if img is not None:
                txt_name = image_name.rpartition(".")[0] + ".txt"
                with open(os.path.join(folder_txt, txt_name), "r") as reader:
                    line = reader.readline()
                    while line != "":
                        values = line.split(" ")
                        x, y, width, height = (
                            float(values[1]) * cols,
                            float(values[2]) * rows,
                            float(values[3]) * cols,
                            float(values[4]) * rows,
                        )
                        cropped_img = img[
                            int(y - height / 2) : int(y + height / 2),
                            int(x - width / 2) : int(x + width / 2),
                        ]
                        resized_img = cv2.resize(cropped_img, (224, 224))
                        dataset.append(resized_img)
                        if int(values[0]) == 9:
                            labels.append((1.0, 0.0))
                        else:
                            labels.append((0.0, 1.0))
                        line = reader.readline()

        except:
            print("Image error")
    return np.array(dataset), np.array(labels)



# The model defined is a custom version obtained starting from the VGG16 net with 2 outputs, defining boat and not boat
def model():
    K.set_learning_phase(0)
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=8,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=1000,activation="relu"))
    model.add(Dense(units=1000,activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=2, activation="softmax"))

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    
    model.summary()   
    return model

# Function used to freeze the model and convert from Keras to Tensorflow, the format recognized by the DNN module of OpenCV 
def save_tensorflow_model(keras_model, dir, file_name):
    converted_model = tf.function(lambda x: keras_model(x))
    converted_model = converted_model.get_concrete_function(
        x=[tf.TensorSpec(inp.shape, tf.float32) for inp in keras_model.inputs]
    )

    frozen_func = convert_variables_to_constants_v2(converted_model)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(frozen_func.graph, dir, name=file_name, as_text=False)

# Training function
def train():
    images_dir = sys.argv[1]
    labels_dir = sys.argv[2]
    model_dir = sys.argv[3]
    epoch_max = int(sys.argv[4])



    print("\nCompiling model...")
    boat_model = model()


    print("\nTraining model...")
    # Due to the large size of the database is decided to import the images only when are needed for the training to avoid possible crashes
    for i in range(0, epoch_max):
        names = os.listdir()
        n = int(len(names)/10)
        random.shuffle(names)
        for j in range(0, len(names), n):
            dataset, labels = load_images_from_folder(names[j:j+n],images_dir, labels_dir)
            boat_model.fit(dataset, labels,epochs=1)

    print("\nSaving model...")
    save_tensorflow_model(boat_model, model_dir, "boat_model.pb")
    print("\nDone")


if __name__ == "__main__":
    train()
