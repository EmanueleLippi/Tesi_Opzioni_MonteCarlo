import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def plot_paths(
    t: np.ndarray,
    X: np.ndarray,
    n_paths: int = 10,
    title: str = "Traiettorie simulate",
    xlabel: str = "Tempo",
    ylabel: str = "Valore",
    save_path: Optional[str] = None,
):
    """
    t: (N+1,)
    X: (M, N+1)
    """
    Np = min(n_paths, X.shape[0])
    plt.figure(figsize=(10, 6))
    plt.plot(t, X[:Np].T, lw=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
