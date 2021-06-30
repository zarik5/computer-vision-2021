import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, preprocessing as preproc

# Parameters
FOCUS_INPUT_SHAPE = (9, 9, 3)
CONTEXT_INPUT_SHAPE = (32, 32, 3)
CONTEXT_LEVELS_COUNT = 4

BATCH_SIZE = 128

HELP_MESSAGE = """
sea_train.py: Train a model to classify the central pixel of windows of pixels as sea or non-sea.

USAGE:
python sea_segmentation.py <in_dataset_dir> <out_model_path>

"""


class Dataset(keras.utils.Sequence):
    def __init__(self, dataset_dir):
        classes_file = open(dataset_dir + "/classes.txt", "r")
        lines = classes_file.readlines()
        self.dataset_dir = dataset_dir
        self.names_and_classes = [l.split() for l in lines]

    def __len__(self):
        return int(len(self.names_and_classes) / BATCH_SIZE)

    def load_images_tensor(self, names):
        tensors = []
        for name in names:
            image = preproc.image.load_img(
                f"{self.dataset_dir}/{name}.png", color_mode="rgb"
            )

            tensors.append(
                preproc.image.img_to_array(
                    image,
                    dtype="float32",
                    data_format="channels_last",
                )
            )

        return np.stack(tensors) / 255

    def __getitem__(self, idx):
        # At the beginning of an epoch, shuffle the entries. Keras can already do this between
        # batches but it cannot do shuffling inside batches.
        if idx == 0:
            np.random.shuffle(self.names_and_classes)

        pairs = self.names_and_classes[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
        names = [pair[0] for pair in pairs]
        classes = [pair[1] for pair in pairs]

        inputs = []

        inputs.append(self.load_images_tensor([f"{name}_focus" for name in names]))

        for i in range(CONTEXT_LEVELS_COUNT):
            inputs.append(
                self.load_images_tensor([f"{name}_context_{i}" for name in names])
            )

        return (inputs, np.array(classes).astype(np.float))


def model():
    inputs = []

    # Process the small window around the pixel to be classified. No max-pooling should be used.
    focus_input = keras.Input(shape=(9, 9, 3), dtype="float32", name="focus_input")
    inputs.append(focus_input)

    focus_features = layers.Conv2D(32, 3, activation="relu")(focus_input)
    focus_features = layers.Conv2D(64, 3, activation="relu")(focus_features)
    focus_features = layers.Dropout(0.25)(focus_features)
    focus_output = layers.Flatten()(focus_features)
    # focus_features = layers.Dense(2048, activation="relu")(focus_features)
    # focus_output = layers.Dropout(0.5)(focus_features)

    # Create a reusable pipeline for context recognition at multiple scales. Weights will be shared.
    # Input: 32x32x3, Output: 4x4x3
    context_input = keras.Input(shape=CONTEXT_INPUT_SHAPE, name=f"context_input")
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
    for i in range(CONTEXT_LEVELS_COUNT):
        context_input = keras.Input(
            shape=CONTEXT_INPUT_SHAPE, dtype="float32", name=f"context_input{i}"
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


def main():
    if len(sys.argv) != 3:
        print(HELP_MESSAGE)
        return

    dataset_dir = sys.argv[1]
    model_path = sys.argv[2]

    print("\nCompiling model...")
    compiled_model = model()

    print("\nIndexing dataset...")
    train_dataset = Dataset(dataset_dir + "/train")
    test_dataset = Dataset(dataset_dir + "/test")

    compiled_model.fit(
        train_dataset,
        epochs=1,
        shuffle=False,
    )

    score = compiled_model.evaluate(test_dataset)

    print("\nLoss:", score[0])
    print("Accuracy:", score[1])

    compiled_model.save(model_path, include_optimizer=False)


if __name__ == "__main__":
    main()
