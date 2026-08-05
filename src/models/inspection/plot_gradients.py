import torch.nn as nn
from matplotlib import pyplot as plt


def plot_gradient_flow(model: nn.Module) -> plt.Figure:
    names, means, maxs = [], [], []

    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        names.append(n)
        means.append(p.grad.abs().mean().item())
        maxs.append(p.grad.abs().max().item())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(names))

    ax.bar(x, means, color="steelblue", label="mean")
    ax.bar(x, maxs, color="coral", alpha=0.6, label="max")

    for i, m in enumerate(means):
        if m == 0:
            ax.text(i, ax.get_ylim()[0] * 2, "∅", ha="center", fontsize=7, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("Gradient Magnitude (log scale)")
    ax.set_title("Gradient Flow Per Layer")
    ax.set_yscale("log")
    ax.legend()

    plt.tight_layout()
    return fig