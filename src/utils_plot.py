import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def plot_paths(
    t,
    X,
    n_paths: int = 10,
    title: str = "Traiettorie simulate",
    xlabel: str = "Tempo",
    ylabel: str = "Valore",
    save_path: Optional[str] = None,
    alpha: float = 0.9,
):
    """
    t: array-like di shape (N+1,)
    X: array-like di shape (M, N+1)
    """
    n_show = min(n_paths, X.shape[0])

    plt.figure(figsize=(10, 6))
    plt.plot(t, X[:n_show].T, lw=1, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
