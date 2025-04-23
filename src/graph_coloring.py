import random
from typing import Optional, List
from time import perf_counter

import numpy as np
from numpy.typing import NDArray


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
        node = filter(lambda e: e[1] == 0, enumerate(coloring)).__next__()[0]
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
    import re
    grid_split = re.split(r"Grid \d+\n", sudoku_str.strip())
    grids = []
    for grid_str in grid_split:
        grid_str = grid_str.strip()
        if not grid_str:
            continue
        
        grid = np.array([np.array(list(map(int, iter(line.strip())))) for line in grid_str.splitlines()], dtype=np.int_)
        assert grid.shape == (9, 9), "Each Sudoku puzzle must be a 9x9 grid."
        grids.append(grid)

    return grids


if __name__ == "__main__":
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


    with open("coloring_tests/sudokus.txt", "r") as f:
        sudokus = parse_sudokus(f.read())

    print("Sudoku puzzles parsed successfully.")

    sudoku_adj = np.array([[False] * 81 for _ in range(81)], dtype=np.bool_)

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
