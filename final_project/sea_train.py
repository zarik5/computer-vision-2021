import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, preprocessing as preproc
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

BATCH_SIZE = 128


def load_images_tensor(names):
    tensors = []
    for name in names:
        image = preproc.image.load_img(name, color_mode="rgb")

        tensors.append(
            preproc.image.img_to_array(
                image,
                dtype="float32",
                data_format="channels_last",
            )
        )

    return np.stack(tensors) / 255


class Dataset(keras.utils.Sequence):
    def __init__(self, dataset_dir, context_levels_count):
        classes_file = open(dataset_dir + "/classes.txt", "r")
        lines = classes_file.readlines()
        self.dataset_dir = dataset_dir
        self.names_and_classes = [l.split() for l in lines]
        self.context_levels_count = context_levels_count

    def __len__(self):
        return int(len(self.names_and_classes) / BATCH_SIZE)

    def get_input_batch(self, file_name_prefixes):
        inputs = []

        inputs.append(
            load_images_tensor(
                [
                    f"{self.dataset_dir}/{prefix}_focus.png"
                    for prefix in file_name_prefixes
                ]
            )
        )

        for i in range(self.context_levels_count):
            inputs.append(
                load_images_tensor(
                    [
                        f"{self.dataset_dir}/{prefix}_context_{i}.png"
                        for prefix in file_name_prefixes
                    ]
                )
            )

        return inputs

    def __getitem__(self, idx):
        # At the beginning of an epoch, shuffle the entries. Keras can already do this between
        # batches but it cannot do shuffling inside batches.
        if idx == 0:
            np.random.shuffle(self.names_and_classes)

        pairs = self.names_and_classes[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
        names = [pair[0] for pair in pairs]
        classes = [pair[1] for pair in pairs]

        inputs = self.get_input_batch(names)
        outputs = np.array(classes).astype(np.float)

        return (inputs, outputs)


def model(focus_input_shape, context_input_shape, context_levels_count):
    inputs = []

    # Process the small window around the pixel to be classified. No max-pooling should be used.
    focus_input = keras.Input(
        shape=focus_input_shape, dtype="float32", name="focus_input"
    )
    inputs.append(focus_input)

    focus_features = layers.Conv2D(32, 3, activation="relu")(focus_input)
    focus_features = layers.Conv2D(64, 3, activation="relu")(focus_features)
    focus_features = layers.Dropout(0.25)(focus_features)
    focus_output = layers.Flatten()(focus_features)
    # focus_features = layers.Dense(2048, activation="relu")(focus_features)
    # focus_output = layers.Dropout(0.5)(focus_features)

    # Create a reusable pipeline for context recognition at multiple scales. Weights will be shared.
    # Input: 32x32x3, Output: 4x4x3
    context_input = keras.Input(shape=context_input_shape, name=f"context_input")
    context_features = layers.Conv2D(16, 3, activation="relu")(context_input)
    context_features = layers.MaxPooling2D(pool_size=(2, 2))(context_features)
    context_features = layers.Conv2D(32, 3, activation="relu")(context_features)
    context_features = layers.MaxPooling2D(pool_size=(2, 2))(context_features)
    context_features = layers.Conv2D(64, 5, activation="relu")(context_features)
    # context_features = layers.MaxPooling2D(pool_size=(2, 2))(context_features)
    context_features = layers.Dropout(0.25)(context_features)
    context_output = layers.Flatten()(context_features)
    context_pipeline = keras.Model(
        context_input, context_output, name="context_pipeline"
    )

    keras.utils.plot_model(context_pipeline, "context_pipeline.png", show_shapes=True)

    # Apply the context pipeline for input windows at multiple scales
    context_outputs = []
    for i in range(context_levels_count):
        context_input = keras.Input(
            shape=context_input_shape, dtype="float32", name=f"context_input{i}"
        )
        inputs.append(context_input)

        context_outputs.append(context_pipeline(context_input))

    context_output = layers.concatenate(context_outputs)

    # Merge context and focus branches and feed them to dense layers
    x = layers.concatenate([focus_output, context_output])
    x = layers.Dense(1000)(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, output)

    keras.utils.plot_model(model, "complete_model.png", show_shapes=True)

    # binary classification (sea or non-sea)
    model.compile(
        optimizer=keras.optimizers.Nadam(),
        loss=keras.losses.binary_crossentropy,
        metrics=["accuracy"],
    )

    return model


def save_tensorflow_model(keras_model, dir, file_name):
    converted_model = tf.function(lambda x: keras_model(x))
    converted_model = converted_model.get_concrete_function(
        x=[tf.TensorSpec(inp.shape, tf.float32) for inp in keras_model.inputs]
    )

    frozen_func = convert_variables_to_constants_v2(converted_model)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(frozen_func.graph, dir, name=file_name, as_text=False)


def train():
    focus_side = int(sys.argv[1]) 
    focus_input_shape = (focus_side, focus_side, 3)
    context_side = int(sys.argv[2])
    context_input_shape = (context_side, context_side, 3)
    context_levels_count = int(sys.argv[3])
    dataset_dir = sys.argv[4]
    model_dir = sys.argv[5]

    print("\nCompiling model...")
    compiled_model = model(focus_input_shape, context_input_shape, context_levels_count)

    print("\nIndexing dataset...")
    train_dataset = Dataset(dataset_dir + "/train", context_levels_count)
    test_dataset = Dataset(dataset_dir + "/test", context_levels_count)

    compiled_model.fit(
        train_dataset,
        epochs=1,
        shuffle=False,
    )

    score = compiled_model.evaluate(test_dataset)

    print("\nLoss:", score[0])
    print("Accuracy:", score[1])

    save_tensorflow_model(compiled_model, model_dir, "model.pb")


if __name__ == "__main__":
    train()
