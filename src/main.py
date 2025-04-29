from pathlib import Path

import numpy as np
from PIL import Image

from src.activation_function import ReLU, Softmax
from src.network import ConvolutionalLayer, Layer, MaxPoolingLayer, NeuralNetwork
from src.loss_function import MSE


def train(nn: NeuralNetwork, train_data: list[tuple[Path, Path]]) -> None:
    """training loop for the neural network"""

    # Process all images and labels
    all_images = []
    all_labels = []

    for image_path, label_path in train_data:
        assert image_path.exists()
        assert label_path.exists()
        assert image_path.stem == label_path.stem

        # Load and preprocess image (normalize to [0,1])
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize to match network input
        img_array = np.array(img).transpose(2, 0, 1) / 255.0  # CHW format, normalized
        all_images.append(img_array)

        # Load label (assuming label file contains class index)
        label_text = label_path.read_text().strip()

        # Convert label to one-hot encoding (assuming 10 classes)
        label_idx = int(label_text)
        one_hot = np.zeros(10)
        one_hot[label_idx] = 1
        all_labels.append(one_hot)

    # Convert to numpy arrays
    X = np.array(all_images)
    y = np.array(all_labels)

    # Train the neural network
    nn.train(
        x=X, y=y, epochs=50, learning_rate=0.01, loss_function=MSE(), batch_size=32
    )


def predict(nn: NeuralNetwork, image_path: Path) -> str:
    """predict the image label using the neural network"""
    assert image_path.exists()
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match network input
    img_array = np.array(img).transpose(2, 0, 1) / 255.0  # CHW format, normalized
    prediction = nn.forward(img_array)
    return str(np.argmax(prediction))


def accuracy(nn: NeuralNetwork, test_data: list[tuple[Path, Path]]) -> float:
    """calculate the accuracy of the neural network on the test data"""
    correct = 0
    total = 0
    for image_path, label_path in test_data:
        prediction = predict(nn, image_path)
        if prediction == label_path.stem:
            correct += 1
        total += 1
    return correct / total


def load_train_data() -> list[tuple[Path, Path]]:
    """
    load the data from the data/car/train/images and data/car/train/labels
    returns a list of tuples of the form (image_path, label_path)
    """

    train_X = sorted(list(Path("data/car/train/images").glob("*.jpg")))
    train_y = sorted(list(Path("data/car/train/labels").glob("*.txt")))
    zipped = list(zip(train_X, train_y))
    return zipped


def load_test_data() -> list[tuple[Path, Path]]:
    """
    load the data from the data/car/test/images and data/car/test/labels
    returns a list of tuples of the form (image_path, label_path)
    """

    test_X = sorted(list(Path("data/car/test/images").glob("*.jpg")))
    test_y = sorted(list(Path("data/car/test/labels").glob("*.txt")))
    zipped = list(zip(test_X, test_y))
    return zipped


def main() -> None:
    """Convolutional Neural Network"""

    # load data
    train_data = load_train_data()

    # define network architecture
    nn = NeuralNetwork(
        layers=[
            # conv & pooling layers
            ConvolutionalLayer(
                input_shape=(3, 224, 224), kernel_size=3, num_filters=32
            ),
            MaxPoolingLayer(input_shape=(32, 222, 222), pool_size=2),
            ConvolutionalLayer(
                input_shape=(32, 111, 111), kernel_size=3, num_filters=64
            ),
            MaxPoolingLayer(input_shape=(64, 109, 109), pool_size=2),
            ConvolutionalLayer(
                input_shape=(64, 54, 54), kernel_size=3, num_filters=128
            ),
            MaxPoolingLayer(input_shape=(128, 52, 52), pool_size=10),
            # fully connected layers
            Layer(input_size=128 * 5 * 5, output_size=64, activation=ReLU()),
            Layer(input_size=64, output_size=10, activation=Softmax()),
        ]
    )

    # training loop
    train(nn, train_data)
    print("Training complete: ", nn)

    # testing loop
    test_data = load_test_data()
    acc = accuracy(nn, test_data)
    print(f"Accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()
