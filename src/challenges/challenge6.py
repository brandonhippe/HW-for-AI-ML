import argparse
from neuron import Neuron
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def plot_animation(neuron, training_data, training_steps, function, output_file=None):
    if output_file is None:
        output_file = Path(Path(__file__).parent, f"{function.__name__}_challenge7.mp4")

    fig, ax = plt.subplots()
    pos = ax.imshow(np.zeros((100, 100)), cmap='Blues')
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    colorbar = fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_title("Output of the neuron during training")

    def update(frame):
        for (a, b) in training_data:
            input_vec = np.array([a, b])
            desired = np.array([function(a, b)])
            neuron(input_vec, desired)

        Z = np.array([[neuron(np.array([a, b]))[0] for a in x] for b in y])
        pos.set_data(Z)
        pos.set_clim(vmin=Z.min(), vmax=Z.max())  # Autoscale the colorbar
        ax.set_title(f"Training Progress: {frame + 1}/{training_steps}")
        return pos,

    ani = FuncAnimation(fig, update, frames=training_steps, interval=50, blit=False)
    ani.save(output_file, writer='ffmpeg', fps=20)  # Save the animation as a video file
    print(f"Animation saved to {output_file}")


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
        neuron = Neuron(2, 1, args.alpha)
        print(f"Training function: {function.__name__}")
        training_data = list(product(range(2), repeat=2))

        plot_animation(neuron, training_data, training_steps, function)
