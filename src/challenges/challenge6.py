import argparse
from itertools import product
from pathlib import Path

import numpy as np
from neuron import Neuron


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
    args = parser.parse_args()

    training_steps = args.steps
    print(f"Training steps: {training_steps}")

    for function in [nand, xor]:
        neuron = Neuron(2, args.alpha)
        print(f"Training function: {function.__name__}")
        training_data = list(map(lambda i: np.array(i), product(range(2), repeat=2)))

        neuron.plot_animation(training_data, training_steps, function, str(Path(Path(__file__).parent, f"{function.__name__}_challenge7.mp4")))
