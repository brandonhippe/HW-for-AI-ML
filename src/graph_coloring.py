import csv
import random
import re
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


def verify_coloring(adj_mat: NDArray[np.bool_], coloring: NDArray[np.int_]) -> bool:
    """
    Verify if the given coloring is valid for the graph represented by the adjacency matrix.

    Parameters
    ----------
    adj_mat : NDArray[np.bool_]
        Adjacency matrix of the graph.
    coloring : NDArray[np.int_]
        Array representing the coloring of the graph. Any uncolored nodes should have a value of 0, and are not considered in the verification.

    Returns
    -------
    bool
        True if the coloring is valid, False otherwise.
    """
    n = adj_mat.shape[0]
    for i, coloring_i in filter(lambda e: e[1] != 0, enumerate(coloring[:-1])):
        for j, coloring_j in filter(lambda e: e[1] != 0, enumerate(coloring[i+1:], i + 1)):
            if adj_mat[i, j] and coloring_i == coloring_j:
                return False
    return True


def generate_coloring(adj_mat: NDArray[np.bool_], num_colors: int, initial_coloring: Optional[NDArray[np.int_]]=None) -> NDArray[np.int_]:
    """
    Generate a possible graph coloring using a recursive algorithm.

    Parameters
    ----------
    adj_mat : NDArray[np.bool_]
        Adjacency matrix of the graph.
    num_colors : int
        Number of available colors.
    initial_coloring : Optional[NDArray[np.int_]]
        Initial coloring of the graph. If None, all nodes are initialized with no color (0).
        Any uncolored nodes should initially have a value of 0.
        The length of the initial coloring must match the number of nodes in the graph.
        If provided, the algorithm will attempt to color the graph starting from this initial coloring.

    Returns
    -------
    NDArray[np.int_]
        Array representing the coloring of the graph. Each element is the color assigned to the corresponding node (1 or larger).
    """
    n = adj_mat.shape[0]
    coloring = initial_coloring[:] if initial_coloring is not None else np.zeros((n,), dtype=np.int_)  # Initialize all nodes with no color (-1)
    assert coloring.shape == (n,), "Initial coloring must have the same length as the number of nodes."

    try:
        node = next(filter(lambda e: e[1] == 0, enumerate(coloring)))[0]
    except StopIteration:
        return coloring
    
    # Find colors of adjacent nodes
    adjacent_colors = set(filter(None, (coloring[neighbor] for neighbor in range(n) if adj_mat[node, neighbor])))

    # Assign the first available color
    for color in set(range(1, num_colors + 1)) - adjacent_colors:
        coloring[node] = color
        
        if verify_coloring(adj_mat, coloring):
            coloring = generate_coloring(adj_mat, num_colors, coloring)
            if not np.any(coloring == 0):
                return coloring
            
        coloring[node] = 0

    return coloring


def parse_sudokus(sudoku_str: str) -> List[NDArray[np.int_]]:
    """
    Parse a string representation of multiple Sudoku puzzles into a list of numpy arrays.

    Parameters
    ----------
    sudoku_str : str
        String representation of Sudoku puzzles, where each puzzle is separated by a newline.

    Returns
    -------
    list[NDArray[np.int_]]
        List of numpy arrays representing the Sudoku puzzles.
    """


class GraphSAGESudokuSolver(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        GraphSAGE-based model for solving Sudoku puzzles.

        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dim : int
            Dimension of hidden layers.
        output_dim : int
            Number of output classes (e.g., 9 for Sudoku).
        """
        super(GraphSAGESudokuSolver, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, 2*hidden_dim)
        self.fc = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of the GraphSAGE model.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge index in COO format.

        Returns
        -------
        torch.Tensor
            Output logits for each node.
        """
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x


def train_graphsage_sudoku_solver(sudoku_adj: NDArray[np.bool_], sudokus: List[NDArray[np.int_]], num_colors: int, epochs: int = 100):
    """
    Train a GraphSAGE-based GNN to solve Sudoku puzzles iteratively by predicting one cell at a time.

    Parameters
    ----------
    sudoku_adj : NDArray[np.bool_]
        Adjacency matrix for the Sudoku graph.
    sudokus : List[NDArray[np.int_]]
        List of Sudoku puzzles as numpy arrays.
    num_colors : int
        Number of available colors (9 for Sudoku).
    epochs : int
        Number of training epochs.

    Returns
    -------
    GraphSAGESudokuSolver
        Trained GraphSAGE model.
    """
    # Convert adjacency matrix to edge index format
    edge_index = torch.tensor(np.array(np.nonzero(sudoku_adj)), dtype=torch.long)

    # Prepare training data
    num_nodes = sudoku_adj.shape[0]
    x = torch.eye(num_nodes, dtype=torch.float)  # Node features (identity matrix)
    y_list = []

    for sudoku in sudokus:
        y = sudoku.flatten() - 1  # Convert 1-9 to 0-8 for training, keep 0 as is for unfilled cells
        y_list.append(torch.tensor(y, dtype=torch.long))

    y = torch.stack(y_list)  # Shape: (num_puzzles, num_nodes)

    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index)

    # Initialize the model, loss function, and optimizer
    model = GraphSAGESudokuSolver(input_dim=num_nodes, hidden_dim=64, output_dim=num_colors)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Use reduction='none' to compute loss per element
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sudoku_idx, sudoku in enumerate(sudokus):
            optimizer.zero_grad()

            # Flatten the Sudoku grid and initialize node features
            solution = sudoku.flatten()
            x = torch.eye(num_nodes, dtype=torch.float)

            # Iteratively predict one cell at a time
            while np.any(solution == 0):  # While there are unfilled cells
                out = model(x, edge_index)  # Predict logits for all nodes
                probabilities = torch.softmax(out, dim=1)  # Convert logits to probabilities

                # Find the most confident unfilled cell
                unfilled_indices = np.where(solution == 0)[0]
                unfilled_probs = probabilities[unfilled_indices]
                max_confidence, max_index = torch.max(unfilled_probs.max(dim=1).values, dim=0)
                selected_cell = unfilled_indices[max_index.item()]

                # Compute loss for the selected cell
                target = y[sudoku_idx, selected_cell]
                if target == -1:  # Skip if the cell is unfilled in the target
                    continue
                loss = criterion(out[selected_cell].unsqueeze(0), target.unsqueeze(0))
                total_loss += loss.item()

                # Backpropagate and update weights
                loss.backward()
                optimizer.step()

                # Assign the most likely value to the selected cell
                predicted_value = torch.argmax(probabilities[selected_cell]).item() + 1  # Convert 0-8 back to 1-9
                solution[selected_cell] = predicted_value

                # Update the node features to reflect the new prediction
                x[selected_cell] = 0  # Reset the one-hot encoding
                x[selected_cell, predicted_value - 1] = 1  # Update with the predicted value

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss}")

    return model


def solve_sudoku_with_graphsage(model: GraphSAGESudokuSolver, sudoku_adj: NDArray[np.bool_], sudoku: NDArray[np.int_]) -> NDArray[np.int_]:
    """
    Use a trained GraphSAGE model to iteratively solve a Sudoku puzzle by predicting one cell at a time.

    Parameters
    ----------
    model : GraphSAGESudokuSolver
        Trained GraphSAGE model.
    sudoku_adj : NDArray[np.bool_]
        Adjacency matrix for the Sudoku graph.
    sudoku : NDArray[np.int_]
        Sudoku puzzle as a numpy array.

    Returns
    -------
    NDArray[np.int_]
        Solved Sudoku puzzle as a numpy array.
    """
    # Convert adjacency matrix to edge index format
    edge_index = torch.tensor(np.array(np.nonzero(sudoku_adj)), dtype=torch.long)

    # Create node features (one-hot encoding for simplicity)
    num_nodes = sudoku_adj.shape[0]
    x = torch.eye(num_nodes, dtype=torch.float)

    # Flatten the Sudoku grid for easier manipulation
    solution = sudoku.flatten()

    model.eval()
    while np.any(solution == 0):  # While there are unfilled cells
        with torch.no_grad():
            out = model(x, edge_index)  # Predict logits for all nodes
            probabilities = torch.softmax(out, dim=1)  # Convert logits to probabilities

        # Find the most confident unfilled cell
        unfilled_indices = np.where(solution == 0)[0]
        unfilled_probs = probabilities[unfilled_indices]
        max_confidence, max_index = torch.max(unfilled_probs.max(dim=1).values, dim=0)
        selected_cell = unfilled_indices[max_index.item()]

        # Assign the most likely value to the selected cell
        predicted_value = torch.argmax(probabilities[selected_cell]).item() + 1  # Convert 0-8 back to 1-9
        solution[selected_cell] = predicted_value

        # Update the node features to reflect the new prediction
        x[selected_cell] = 0  # Reset the one-hot encoding
        x[selected_cell, predicted_value - 1] = 1  # Update with the predicted value

    return solution.reshape((9, 9))


def ix_to_rc(ix):
    r = ix // 9
    c = ix % 9
    return r, c

def rc_to_ix(r, c):
    return r * 9 + c

def rc_to_box(r, c):
    box = (c // 3) + (3 * (r // 3))
    return box

def box_to_rc(box, ix):
    r = 3 * (box // 3) + (ix // 3)
    c = 3 * (box % 3) + (ix % 3)
    return r, c


def main():
    adj_mat = np.array([
        [False, True, True, False],
        [True, False, True, True],
        [True, True, False, True],
        [False, True, True, False]
    ], dtype=np.bool_)

    num_colors = 3
    coloring = generate_coloring(adj_mat, num_colors)
    print("Generated Coloring:", coloring)
    assert verify_coloring(adj_mat, coloring), "The generated coloring is not valid."
    print("The generated coloring is valid.")

    sudoku_filename = "sudoku.txt"
    # sudoku_filename = "sudoku.csv"
    sudoku_path = Path(__file__).parent / "coloring_tests" / sudoku_filename

    with open(sudoku_path, encoding="UTF-8") as f:
        if sudoku_path.suffix == ".csv":
            sudoku_csv = csv.DictReader(f)
            sudoku_strings = [row["quizzes"] for row in sudoku_csv]
            # convert_sudoku = lambda sudoku: np.array(list(map(int, sudoku)), dtype=np.int_).reshape((9, 9))
            # sudokus = [convert_sudoku(sudoku_strings[0])]
            sudokus = [np.array(list(map(int, sudoku)), dtype=np.int_).reshape((9, 9)) for sudoku in sudoku_strings]
        else:
            grid_split = re.split(r"Grid \d+\n", f.read().strip())
            sudokus = []
            for grid_str in grid_split:
                grid_str = grid_str.strip()
                if not grid_str:
                    continue
                
                grid = np.array([np.array(list(map(int, iter(line.strip())))) for line in grid_str.splitlines()], dtype=np.int_)
                sudokus.append(grid)

    assert all(grid.shape == (9, 9) for grid in sudokus), "Each Sudoku puzzle must be a 9x9 grid."
    print("Sudoku puzzles parsed successfully.")

    sudoku_adj = np.array([[False] * 81 for _ in range(81)], dtype=np.bool_)

    for ix in range(len(sudoku_adj)):
        r, c = ix_to_rc(ix)
        box = rc_to_box(r, c)
        for cell_ix in range(9):
            sudoku_adj[ix, rc_to_ix(r, cell_ix)] = True
            sudoku_adj[ix, rc_to_ix(cell_ix, c)] = True
            sudoku_adj[ix, rc_to_ix(*box_to_rc(box, cell_ix))] = True

    print("Created Sudoku Adjacency Matrix")
    r = random.randint(0, 8)
    c = random.randint(0, 8)
    print(f"Adjacency list for Row {r + 1} Col {c + 1}")
    print(sudoku_adj[rc_to_ix(r, c)][:].reshape((9, 9)))

    tot_time = 0

    for ix, sudoku in enumerate(sudokus, 1):
        start_t = perf_counter()
        solution = generate_coloring(sudoku_adj, 9, initial_coloring=sudoku.reshape(81,))
        tot_time += perf_counter() - start_t
        print(f"Grid {ix} solution:")
        print(solution.reshape((9, 9)))

    print(f"Average solve time: {tot_time / len(sudokus):.3f}s")

    tot_time = 0

    # Train the GNN
    num_colors = 9
    tot_time = 0
    valid = 0
    start_t = perf_counter()
    model = train_sudoku_solver(sudoku_adj, sudokus, num_colors, epochs=100_000)
    tot_time += perf_counter() - start_t

    # Solve Sudoku puzzles
    for ix, sudoku in enumerate(sudokus, 1):
        start_t = perf_counter()
        solution = solve_sudoku_with_gnn(model, sudoku_adj, sudoku)
        tot_time += perf_counter() - start_t
        if verify_coloring(sudoku_adj, solution[:].reshape(81,)):
            print(f"Grid {ix} solution:")
            valid += 1
        else:
            print(f"Invalid solution for grid {ix}!")
        print(solution)

    print(f"Average solve time: {tot_time / len(sudokus):.3f}s")

    # Train the GraphSAGE model
    num_colors = 9
    tot_time = 0

    for ix, sudoku in enumerate(sudokus, 1):
        start_t = perf_counter()
        solution = generate_coloring(sudoku_adj, 9, initial_coloring=sudoku.reshape(81,))
        tot_time += perf_counter() - start_t
        print(f"Grid {ix} solution:")
        print(solution.reshape((9, 9)))

    print(f"Average solve time (including training): {tot_time / len(sudokus):.3f}s")

    
if __name__ == "__main__":
    main()
