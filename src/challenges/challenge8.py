import argparse
from collections.abc import Iterable
from itertools import product
from pathlib import Path

import numpy as np
from neuron import NeuralNetwork


def xor(a, b):
    return round(a) ^ round(b)


def nand(a, b):
    return int(not (round(a) & round(b)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neuron and generate an animation.")
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Learning rate for the neuron (default: 0.35)"
    )
    parser.add_argument(
        "--hidden_layers",
        nargs="+",
        type=list[int],
        help="List of integers representing the number of neurons in each hidden layer",
        default=[2]
    )
    args = parser.parse_args()
    args.hidden_layers = [int(x[0] if isinstance(x, Iterable) else x) for x in args.hidden_layers]
    print(f"Hidden layers: {args.hidden_layers}")

    training_steps = args.steps
    print(f"Training steps: {training_steps}")

    for function in [nand, xor]:
        nn = NeuralNetwork(2, 1, args.hidden_layers, args.alpha)
        print(f"Training function: {function.__name__}")
        training_data = list(map(lambda i: np.array(i), product(range(2), repeat=2)))

        nn.plot_animation(training_data, training_steps, function, str(Path(Path(__file__).parent, f"{function.__name__}_challenge8.mp4")))
