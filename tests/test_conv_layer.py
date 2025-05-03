import numpy as np

from src.activation_function import ReLU
from src.layers.vision import ConvolutionalLayer


def test_conv_layer_forward_shape():
    """Test that convolutional layer produces outputs with the expected shape."""
    # Initialize layer
    input_shape = (3, 32, 32)  # (channels, height, width)
    kernel_size = 3
    num_filters = 16
    stride = 1
    padding = 1

    conv_layer = ConvolutionalLayer(
        input_shape=input_shape,
        kernel_size=kernel_size,
        num_filters=num_filters,
        stride=stride,
        padding=padding,
        activation=ReLU(),
    )

    # Create test input (batch of 2 images)
    batch_size = 2
    test_input = np.random.rand(batch_size, *input_shape)

    # Calculate expected output dimensions
    expected_height = (input_shape[1] - kernel_size + 2 * padding) // stride + 1
    expected_width = (input_shape[2] - kernel_size + 2 * padding) // stride + 1
    expected_shape = (batch_size, num_filters, expected_height, expected_width)

    # Run forward pass
    output = conv_layer.forward(test_input)

    # Check shape
    assert output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {output.shape}"
    )

    # Check that values are non-negative (ReLU activation)
    assert np.all(output >= 0), (
        "Expected all values to be non-negative with ReLU activation"
    )

    print(
        f"✅ Convolutional layer forward pass shape test passed. Output shape: {output.shape}"
    )
    return True


def test_conv_layer_padding():
    """Test that padding is correctly applied."""
    # Initialize layer with padding
    input_shape = (3, 16, 16)
    kernel_size = 3
    num_filters = 8
    padding = 1

    conv_layer = ConvolutionalLayer(
        input_shape=input_shape,
        kernel_size=kernel_size,
        num_filters=num_filters,
        padding=padding,
    )

    # Set all filter weights to 1 for easy verification
    conv_layer.filters.fill(1.0 / (kernel_size * kernel_size * input_shape[0]))
    conv_layer.biases.fill(0.0)

    # Create test input with known values
    batch_size = 1
    test_input = np.ones((batch_size, *input_shape))

    # Run forward pass
    output = conv_layer.forward(test_input)

    # With all 1s input, unit weights, and zero bias, all output values should be 1
    expected_output_shape = (batch_size, num_filters, input_shape[1], input_shape[2])
    assert output.shape == expected_output_shape, (
        f"Expected shape {expected_output_shape}, got {output.shape}"
    )

    # Check center values (should all be close to 1)
    assert np.allclose(output[:, :, 1:-1, 1:-1], 1.0, atol=1e-5), (
        "Center values should be close to 1"
    )

    print(f"✅ Convolutional layer padding test passed. Output shape: {output.shape}")
    return True


def test_conv_layer_backward():
    """Test that convolutional layer backpropagation updates parameters correctly."""
    # Initialize layer with deterministic values
    input_shape = (3, 8, 8)  # (channels, height, width)
    kernel_size = 3
    num_filters = 4
    stride = 1
    padding = 1

    conv_layer = ConvolutionalLayer(
        input_shape=input_shape,
        kernel_size=kernel_size,
        num_filters=num_filters,
        stride=stride,
        padding=padding,
    )

    # Set fixed initial weights for reproducibility
    np.random.seed(42)
    conv_layer.filters = (
        np.random.randn(num_filters, input_shape[0], kernel_size, kernel_size) * 0.1
    )
    conv_layer.biases = np.zeros(num_filters)

    # Store initial weights for comparison
    initial_filters = conv_layer.filters.copy()
    initial_biases = conv_layer.biases.copy()

    # Create test input with deterministic values
    np.random.seed(100)
    batch_size = 2
    test_input = np.random.rand(batch_size, *input_shape)

    # Forward pass
    output = conv_layer.forward(test_input)

    # Create a meaningful gradient (instead of all ones)
    # Using random values ensures we'll have significant updates
    np.random.seed(200)
    dummy_gradient = np.random.rand(*output.shape) * 2 - 1  # Values between -1 and 1
    # Ensure dummy_gradient is explicitly a numpy ndarray
    dummy_gradient = np.asarray(dummy_gradient, dtype=np.float64)

    # Backward pass with a larger learning rate
    learning_rate = 0.1
    input_gradient = conv_layer.backward(
        gradient=dummy_gradient, learning_rate=learning_rate
    )

    # Check that filters and biases were updated
    assert not np.allclose(conv_layer.filters, initial_filters, atol=1e-6), (
        "Filters should be updated during backward pass"
    )
    assert not np.allclose(conv_layer.biases, initial_biases, atol=1e-6), (
        "Biases should be updated during backward pass"
    )

    # Check input gradient shape
    assert input_gradient.shape == test_input.shape, (
        f"Expected input gradient shape {test_input.shape}, got {input_gradient.shape}"
    )

    # Manually verify a simple case to ensure gradient calculation works
    # Reset the layer with simple values
    simple_layer = ConvolutionalLayer(
        input_shape=(1, 3, 3),  # Single channel small input
        kernel_size=2,
        num_filters=1,
        stride=1,
        padding=0,
    )

    # Set simple weights
    simple_layer.filters = np.ones((1, 1, 2, 2))
    simple_layer.biases = np.zeros(1)

    # Simple input
    simple_input = np.ones((1, 1, 3, 3))

    # Forward pass
    simple_output = simple_layer.forward(simple_input)

    # Expected shape: (1, 1, 2, 2) with all values = 4 + 0 = 4 before activation
    assert simple_output.shape == (1, 1, 2, 2)

    # Simple gradient (all ones)
    simple_gradient = np.ones_like(simple_output)

    # Backward pass
    simple_layer.backward(simple_gradient, learning_rate=0.1)

    # Since gradient is all ones and input is all ones, each filter weight should
    # be reduced by learning_rate * 1 * 1 / batch_size = 0.1
    assert np.allclose(simple_layer.filters, 0.9), (
        f"Expected filters to be reduced to 0.9, got {simple_layer.filters}"
    )

    print("✅ Convolutional layer backward pass test passed")
    return True
