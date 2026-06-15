import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# -----------------------
# Matplotlib LaTeX config
# -----------------------
rcParams.update(
    {
        # Use LaTeX for text rendering
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        # Font sizes (tuned for papers)
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # Line styling
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        # Grid
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        # Figure size (single column IEEE ~3.5in)
        "figure.figsize": (3.5, 2.4),
        "figure.dpi": 300,
        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

if __name__ == "__main__":
    # Example data
    x = np.linspace(0, 10, 300)
    y = np.sin(x)
    y2 = np.cos(x)

    fig, ax = plt.subplots()

    ax.plot(x, y, label=r"$\sin(x)$")
    ax.plot(x, y2, label=r"$\cos(x)$", linestyle="--")

    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Amplitude")
    ax.set_title(r"Example Signal Dynamics")

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(
        "plotting/plots/example_plot.pdf"
    )  # Always save as vector PDF for papers
