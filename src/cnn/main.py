import csv
from pathlib import Path
from typing import Literal

import numpy as np
import tqdm
from PIL import Image

from src.activation_function import ReLU, Softmax
from src.layers.dense import PerceptronLayer
from src.layers.vision import ConvolutionalLayer, FlattenLayer, MaxPoolingLayer
from src.loss_function import MSE, LossFunction
from src.network import NeuralNetwork

# image_path, label
Dataset = list[tuple[Path, str]]
SPORTS = ["basketball"]  # , "football", "tennis", "volleyball", "golf"]
NUM_CLASSES = 100
INPUT_SIZE = (224, 224)  # Default input size
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.01
LOSS_FUNCTION = MSE()


def train(
    nn: NeuralNetwork,
    training_data: Dataset,
    num_classes: int = NUM_CLASSES,
    input_size: tuple[int, int] = INPUT_SIZE,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    loss_function: LossFunction = LOSS_FUNCTION,
) -> list[float]:
    """training loop for the neural network"""

    # Process all images and labels
    all_images = []
    all_labels = []

    for image_path, label in tqdm.tqdm(training_data, desc="Processing training data"):
        if not image_path.exists() or not image_path.suffix.endswith(".jpg"):
            continue

        # load image as
        img_array = preprocess_image(image_path=image_path, input_size=input_size)
        all_images.append(img_array)

        # Load label (assuming label file contains class index)
        one_hot = np.zeros(num_classes)
        one_hot[int(label)] = 1
        all_labels.append(one_hot)

    # Convert to numpy arrays
    print(
        f"Converting to numpy arrays: {len(all_images)} images, {len(all_labels)} labels"
    )
    X = np.array(all_images)
    y = np.array(all_labels)

    # Train the neural network
    num_samples = X.shape[0]
    losses: list[float] = []

    # Use full batch if batch_size is None
    batch_size = num_samples if batch_size is None else batch_size

    for epoch in tqdm.tqdm(range(epochs), desc="Training"):
        epoch_loss = 0.0

        # Shuffle data for each epoch
        indices = np.random.permutation(num_samples)
        x_shuffled = X[indices]
        y_shuffled = y[indices]

        # Mini-batch training
        for i in tqdm.tqdm(
            range(0, num_samples, batch_size),
            desc=f"Training epoch {epoch} batch",
        ):
            x_batch = x_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            # Forward pass
            y_pred = nn.forward(x_batch)

            # Calculate loss
            batch_loss = loss_function.function(y_batch, y_pred)
            epoch_loss += batch_loss * len(x_batch) / num_samples

            # Calculate gradients
            gradient = loss_function.derivative(y_batch, y_pred)

            # Backward pass
            nn.backward(gradient, learning_rate)

        losses.append(epoch_loss)

    return losses


def predict(
    nn: NeuralNetwork, image_path: Path, input_size: tuple[int, int] = INPUT_SIZE
) -> str:
    """predict the image label using the neural network"""
    assert image_path.exists()
    img_array = preprocess_image(image_path, input_size=input_size)
    prediction = nn.forward(img_array)
    return str(np.argmax(prediction))


def accuracy(
    nn: NeuralNetwork, test_data: Dataset, input_size: tuple[int, int] = INPUT_SIZE
) -> float:
    """calculate the accuracy of the neural network on the test data"""
    correct = 0
    total = 0
    for image_path, label in tqdm.tqdm(test_data, desc="Calculating accuracy"):
        prediction = predict(nn, image_path, input_size=input_size)
        if prediction == label:
            correct += 1
        total += 1
    return correct / total


def preprocess_image(
    image_path: Path,
    input_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """preprocess the image"""
    img = Image.open(image_path)
    img = img.convert("RGB")  # Ensure the image is in RGB format
    img = img.resize(input_size)  # Resize to match network input
    img_array = np.array(img)

    # Check if the image has 3 channels (RGB)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = img_array.transpose(2, 0, 1)  # CHW format
    elif len(img_array.shape) == 2:  # Grayscale image
        img_array = np.expand_dims(img_array, axis=0)  # Add channel dimension
    else:  # Handle other cases like RGBA
        img_array = img_array[:, :, :3].transpose(
            2, 0, 1
        )  # Take first 3 channels and transpose

    return img_array / 255.0  # normalize


def load_data(
    stage: Literal["train", "test", "valid"],
    sport_names: list[str] | None = None,
) -> Dataset:
    """
    load the data from CSV
    optionally filter for a specific sport category
    """
    # read the csv file
    csv_data = Path("data/imgcls/sports.csv").read_text()
    csv_reader = csv.reader(csv_data.splitlines())

    # Filter by stage (train/test/valid)
    dataset = [row for row in csv_reader if row[3] == stage]

    # Filter by sport name if provided
    if sport_names:
        dataset = [row for row in dataset if row[2] in sport_names]

    X = tqdm.tqdm(
        [Path(f"data/imgcls/{row[1]}") for row in dataset],
        desc=f"Loading {stage} data",
    )
    y = [row[0] for row in dataset]  # get the class id, not the label

    return list(zip(X, y))


def main(input_size: tuple[int, int] = INPUT_SIZE) -> None:
    """Convolutional Neural Network with shallower architecture"""

    # Calculate pooling output dimensions
    h, w = input_size
    pool1_h, pool1_w = h // 4, w // 4
    pool2_h, pool2_w = pool1_h // 4, pool1_w // 4

    # define network architecture with reduced depth for faster execution
    nn = NeuralNetwork(
        layers=[
            # First convolutional block
            ConvolutionalLayer(
                input_shape=(3, h, w), kernel_size=3, num_filters=32, padding=1
            ),  # Output: (32, h, w)
            MaxPoolingLayer(
                input_shape=(32, h, w), pool_size=4
            ),  # Output: (32, h/4, w/4)
            # Second convolutional block
            ConvolutionalLayer(
                input_shape=(32, pool1_h, pool1_w),
                kernel_size=3,
                num_filters=64,
                padding=1,
            ),  # Output: (64, h/4, w/4)
            MaxPoolingLayer(
                input_shape=(64, pool1_h, pool1_w), pool_size=4
            ),  # Output: (64, h/16, w/16)
            FlattenLayer(),
            # Fully connected layers
            PerceptronLayer(
                input_size=64 * pool2_h * pool2_w, output_size=256, activation=ReLU()
            ),
            PerceptronLayer(
                input_size=256, output_size=NUM_CLASSES, activation=Softmax()
            ),
        ]
    )

    # training loop with single sport
    training_data = load_data("train", sport_names=SPORTS)
    train(
        nn=nn,
        training_data=training_data,
        num_classes=NUM_CLASSES,
        input_size=input_size,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        loss_function=LOSS_FUNCTION,
    )

    # testing loop
    testing_data = load_data("test", sport_names=SPORTS)
    acc = accuracy(nn, testing_data, input_size=input_size)
    print(f"Accuracy: {acc:.2f}")


if __name__ == "__main__":
    main(input_size=(64, 64))
