from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Iterable, Tuple, List
from numpy.typing import NDArray


@dataclass
class NeuralNetwork:
    num_inputs: int
    num_outputs: int
    neuron_layers: Iterable[int]
    alpha: float = 0.01
    neurons: List[List['Neuron']] = field(default_factory=list)

    def __post_init__(self) -> None:
        inputs = self.num_inputs
        for layer_size in self.neuron_layers:
            layer = [Neuron(inputs, layer_size, self.alpha) for _ in range(layer_size)]
            self.neurons.append(layer)
            inputs = layer_size
    
    def __call__(self, input_vec: NDArray, desired: Optional[NDArray]=None) -> NDArray:
        inputs = input_vec
        for layer in self.neurons:
            outputs = []
            for neuron in layer:
                output, _ = neuron(inputs)
                outputs.append(output)
                inputs = np.array(outputs).T  # Update inputs for the next layer

        if desired is not None:
            for layer in reversed(self.neurons):
                for neuron in layer:
                    _, desired = neuron(inputs, desired)
                    inputs = np.array([desired]).T

        return inputs


@dataclass
class Neuron:
    num_inputs: int
    num_outputs: int
    alpha: float = 0.01
    weights: NDArray = field(init=False)
    bias: NDArray = field(init=False)
    output: NDArray = field(init=False, default=None)

    def __post_init__(self) -> None:
        rng = np.random.default_rng()
        self.weights = rng.uniform(-1, 1, (self.num_inputs, self.num_outputs))
        self.bias = rng.uniform(-1, 1, (self.num_outputs))
    
    def __call__(self, input_vec: NDArray, desired: Optional[NDArray]=None) -> Tuple[Optional[NDArray]]:
        self.output = 1 / (1 + np.exp(-(input_vec @ self.weights + self.bias)))
        if desired is not None:
            desired = self.update_weights(input_vec, desired)

        return self.output, desired
    
    def update_weights(self, input_vec: NDArray, desired: NDArray) -> NDArray:
        if self.output is None:
            raise ValueError("Output is not computed yet. Call the neuron with input first.")
        
        delta = desired - self.output
        self.weights += self.alpha * np.outer(input_vec, delta)
        self.bias += self.alpha * delta

        return self(input_vec)
