import numpy as np

from src.network import NeuralNetwork
from src.activation_function import ReLU, Sigmoid


def test_neural_network() -> None:
    """Test that the neural network can be created and forward pass works."""
    # Create a neural network
    network = NeuralNetwork()

    # Add layers (input_size, output_size, activation)
    network.add_layer(input_size=3, output_size=4, activation=ReLU())  # Hidden layer
    network.add_layer(input_size=4, output_size=2, activation=Sigmoid())  # Output layer

    # Forward pass through the network
    input_data = np.array([0.1, 0.2, 0.3])
    output = network.forward(input_data)

    # Assert output has correct shape
    assert output.shape == (2,)
    # Sigmoid activation bounds values between 0 and 1
    assert np.all(output >= 0) and np.all(output <= 1)


def test_network_consistency() -> None:
    """Test that the network produces the same output for the same input."""
    network = NeuralNetwork()
    network.add_layer(input_size=2, output_size=3, activation=ReLU())
    network.add_layer(input_size=3, output_size=1, activation=Sigmoid())

    input_data = np.array([0.5, 0.7])
    output1 = network.forward(input_data)
    output2 = network.forward(input_data)

    np.testing.assert_array_almost_equal(output1, output2)


def test_xor_problem() -> None:
    """Test that the network can learn the XOR problem."""
    # XOR problem inputs and expected outputs
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create a network with appropriate architecture for XOR
    network = NeuralNetwork()
    network.add_layer(input_size=2, output_size=4, activation=ReLU())
    network.add_layer(input_size=4, output_size=1, activation=Sigmoid())

    # Train the network on XOR data
    # Typically we would call something like:
    # network.train(X, y, epochs=1000, learning_rate=0.1)

    # For now, we'll just check the forward pass
    for i in range(len(X)):
        output = network.forward(X[i])
        # Just verify output shape until training is implemented
        assert output.shape == (1,)

    # Once training is implemented, we would add:
    # After training, check predictions
    # for i in range(len(X)):
    #     prediction = network.forward(X[i])
    #     np.testing.assert_array_almost_equal(prediction, y[i], decimal=1)
