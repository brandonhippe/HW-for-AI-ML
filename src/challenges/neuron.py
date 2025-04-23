import os
from dataclasses import dataclass, field
from math import ceil
from typing import Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.random import PCG64, Generator
from numpy.typing import NDArray


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    length = os.get_terminal_size().columns - 10
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def plot_animation(obj: Union['Neuron', 'NeuralNetwork'], training_data, training_steps, function, output_file, duration=5, fps=20):
    tot_frames = duration * fps
    interval = duration * 1000 / tot_frames

    epochs_per_frame = ceil(training_steps / tot_frames)
    training_steps = max(training_steps, tot_frames * epochs_per_frame)

    print(f"Total frames: {tot_frames}, Epochs per frame: {epochs_per_frame}")
    print(f"Training steps: {training_steps}, Duration: {duration} seconds, FPS: {fps}")

    fig, ax = plt.subplots()
    pos = ax.imshow(np.zeros((100, 100)), cmap='Blues')
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    colorbar = fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_title(f"Output while training {function.__name__} (Epoch {epochs_per_frame} of {training_steps})")
    colorbar.set_label("Output")

    def update(frame):
        printProgressBar(frame + 1, tot_frames)
        for _ in range(epochs_per_frame):
            for input_vec in training_data:
                desired = np.array([function(*input_vec)])
                obj(input_vec, desired)

        Z = np.array([[obj(np.array([a, b])) for a in x] for b in y])
        pos.set_data(Z)
        pos.set_clim(vmin=Z.min(), vmax=Z.max())  # Autoscale the colorbar
        ax.set_title(f"Output while training {function.__name__} (Epoch {(frame + 1) * epochs_per_frame} of {training_steps})")
        return pos,

    ani = FuncAnimation(fig, update, frames=tot_frames, interval=interval, blit=False)
    ani.save(output_file, writer='ffmpeg', fps=fps)  # Save the animation as a video file
    print(f"Animation saved to {output_file}")


def sigmoid(x: NDArray) -> NDArray:
    """
    Sigmoid activation function.
    """
    assert np.min(x) >= -1000, "Input to sigmoid must be >= -1000"
    assert np.max(x) <= 1000, "Input to sigmoid must be <= 1000"
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: NDArray) -> NDArray:
    """
    Derivative of the sigmoid function.
    """
    assert np.min(x) >= 0, "Input to sigmoid derivative must be non-negative"
    assert np.max(x) <= 1, "Input to sigmoid derivative must be less than or equal to 1"
    return x * (1 - x)

def relu(x: NDArray) -> NDArray:
    """
    ReLU activation function.
    """
    return np.maximum(0, x)

def relu_derivative(x: NDArray) -> NDArray:
    """
    Derivative of the ReLU function.
    """
    return np.where(x > 0, 1, 0)


@dataclass
class NeuralNetwork:
    num_inputs: int
    num_outputs: int
    hidden_layers: Iterable[int]
    alpha: float = 0.01
    neurons: List[List['Neuron']] = field(default_factory=list)

    def __post_init__(self) -> None:
        layer_order = [1, self.num_inputs] + self.hidden_layers + [self.num_outputs]
        for p_layer_size, layer_size in zip(layer_order[:-1], layer_order[1:]):
            layer = [Neuron(p_layer_size, self.alpha) for _ in range(layer_size)]
            self.neurons.append(layer)

        print(f"Neural network layers: {list(map(lambda n: len(n), self.neurons))}")
    
    def __call__(self, input_vec: NDArray, desired: Optional[NDArray] = None, printActivation: Optional[bool]=False) -> NDArray:
        """
        Perform a forward pass and optionally update weights using backpropagation if desired output is provided.
        """
        inputs = input_vec[:]
        activations = [inputs]  # Store activations for each layer

        # Forward pass
        for i, layer in enumerate(self.neurons):
            if i == 0:
                layer_outputs = np.array(list(map(lambda e: e[1](np.array([inputs[e[0]]])), enumerate(layer)))) # Compute outputs for the current layer
            else:
                layer_outputs = np.array(list(map(lambda n: n(inputs), layer)))
            inputs = layer_outputs[:]  # Update inputs for the next layer
            activations.append(inputs)

        if printActivation:
            for i, activation in enumerate(activations):
                print(f"Layer {i} activation: {activation}")

        # If desired output is provided, perform backpropagation
        if desired is not None:
            self.update_weights(activations, desired)

        return inputs  # Final output of the network

    def update_weights(self, activations: List[NDArray], desired: NDArray) -> None:
        """
        Perform backpropagation to update weights and biases in all layers.
        """
        # Compute the error at the output layer
        delta = desired - activations[-1]  # Shape: (num_outputs,)
        deltas = [delta]  # Store deltas for each layer

        # Backpropagate the error through the hidden layers
        for l in range(len(self.neurons) - 1, 0, -1):  # Iterate backward through layers
            layer = self.neurons[l]
            prev_layer = self.neurons[l - 1]
            delta = np.zeros(len(prev_layer))
            for i, neuron in enumerate(prev_layer):
                for j, next_neuron in enumerate(layer):
                    # Ensure all components are scalars
                    weight_ij = next_neuron.weights[i, 0]  # Scalar weight
                    activation_i = neuron.output # Scalar activation
                    delta[i] += deltas[0][j] * weight_ij * neuron.activation_function_derivative(activation_i) # Scalar delta
            deltas.insert(0, delta) # Insert at the beginning to match layer order

        # Update weights and biases for each layer
        for l, layer in enumerate(self.neurons):
            for i, neuron in enumerate(layer):
                neuron.update_weights(deltas[l][i])

    def plot_animation(self, training_data, training_steps, function, output_file):
        plot_animation(self, training_data, training_steps, function, output_file)


@dataclass
class Neuron:
    num_inputs: int
    alpha: float = 0.01
    weights: NDArray = field(init=False)
    bias: float = field(init=False, default=0)
    input: NDArray = field(init=False, default=None)
    pre_activation: float = field(init=False, default=None)
    output: float = field(init=False, default=None)
    activation_function: callable = field(default=sigmoid)
    activation_function_derivative: callable = field(default=sigmoid_derivative)

    def __post_init__(self) -> None:
        # Randomly initialize weights and biases
        rng = Generator(PCG64())
        self.weights = rng.normal(0, np.sqrt(2 / self.num_inputs), (self.num_inputs, 1))

    def __call__(self, input_vec: NDArray, desired: Optional[NDArray]=None) -> float:
        """
        Perform a forward pass and optionally update weights if desired output is provided.
        """
        assert input_vec.shape == (self.num_inputs,), f"Input vector must have shape ({self.num_inputs},), got {input_vec.shape}"

        # Ensure input_vec is a column vector
        self.input = input_vec.reshape(-1, 1)  # Shape: (num_inputs, 1)

        # Compute the output using the sigmoid activation function
        # Multiply and accumulate
        self.pre_activation = self.input.T @ self.weights + self.bias  # Shape: (1, num_outputs)
        
        # Apply activation function
        self.output = self.activation_function(self.pre_activation).item()  # Shape: (num_outputs,)

        # If desired output is provided, update weights
        if desired is not None:
            delta = desired - self.output
            self.update_weights(delta)

        return self.output
    
    def update_weights(self, delta: NDArray) -> None:
        """
        Update weights and biases using the delta (error) and learning rate.
        """
        # Update weights and biases
        d_pred = delta * self.activation_function_derivative(self.output)  # Shape: (num_outputs,)
        error = d_pred * self.input  # Shape: (num_inputs, num_outputs)
        self.weights += self.alpha * error  # Shape: (num_inputs, num_outputs)
        self.bias += self.alpha * d_pred

    def plot_animation(self, training_data, training_steps, function, output_file):
        plot_animation(self, training_data, training_steps, function, output_file)

if __name__ == "__main__":
    def test_training(obj):
        print("Before training:")
        outputs = []
        for input_vec, desired_output in training_data:
            output = obj(input_vec)
            outputs.append(output)
            print(f"\tInput: {input_vec}, Expected: {desired_output}, Output: {output}")

        for epoch in range(training_steps):
            for input_vec, desired_output in training_data:
                obj(input_vec, desired_output)

        # Test the network
        print("After training:")
        outputs_after_training = []
        for input_vec, desired_output in training_data:
            output = obj(input_vec)
            outputs_after_training.append(output)
            print(f"\tInput: {input_vec}, Expected: {desired_output}, Output: {output}")

            assert abs(desired_output.item() - output) < training_threshold, "Network did not learn correctly"
    
    default_alpha = 0.35
    training_steps = 10000
    training_threshold = 0.5
    print("Testing Neuron and NeuralNetwork classes")

    # Test Neuron
    print("Testing Neuron class")
    neuron = Neuron(num_inputs=2, alpha=default_alpha)
    outputs = []

    training_data = [
        (np.array([0.5, 0.8]), np.array([1])),
        (np.array([0.2, 0.3]), np.array([0]))
    ]
    test_training(neuron)

    # Test NeuralNetwork
    print("Testing NeuralNetwork class")
    nn = NeuralNetwork(num_inputs=2, num_outputs=1, hidden_layers=[3], alpha=default_alpha)
    outputs = []

    print("Testing NAND function")
    training_data = [
        (np.array([0, 0]), np.array([1])),
        (np.array([0, 1]), np.array([1])),
        (np.array([1, 0]), np.array([1])),
        (np.array([1, 1]), np.array([0]))
    ]
    test_training(nn)

    nn = NeuralNetwork(num_inputs=2, num_outputs=1, hidden_layers=[3], alpha=default_alpha)

    print("Testing XOR function")
    training_data = [
        (np.array([0, 0]), np.array([0])),
        (np.array([0, 1]), np.array([1])),
        (np.array([1, 0]), np.array([1])),
        (np.array([1, 1]), np.array([0]))
    ]
    test_training(nn)
