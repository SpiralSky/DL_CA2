import pandas as pd
import torch
from pandas.io.formats.style_render import StylerRenderer
from torch.utils.data import DataLoader


def get_class_statistics(dataloader: DataLoader, classes: list[str]) -> StylerRenderer:
    sums = {c: torch.zeros(3) for c in classes}
    brightness_vals = {c: [] for c in classes}
    counts = {c: 0 for c in classes}

    for batch_images, batch_labels in dataloader:
        for img, label in zip(batch_images, batch_labels):
            class_name: str = classes[int(label.item())]
            sums[class_name] += img.mean(dim=(1, 2))
            brightness_vals[class_name].append(img.mean().item())
            counts[class_name] += 1

    rows = []
    for c in classes:
        n = counts[c]
        r, g, b = (sums[c] / n).tolist()
        brightness = torch.tensor(brightness_vals[c])
        rows.append({
            "class": c,
            "mean_r": r,
            "mean_g": g,
            "mean_b": b,
            "brightness_mean": brightness.mean().item(),
            "brightness_std": brightness.std().item(),
        })

    df = pd.DataFrame(rows).set_index("class")

    return df.style.background_gradient(
        subset=["mean_r"], cmap="Reds"
    ).background_gradient(
        subset=["mean_g"], cmap="Greens"
    ).background_gradient(
        subset=["mean_b"], cmap="Blues"
    ).background_gradient(
        subset=["brightness_mean", "brightness_std"], cmap="Greys"
    ).format(precision=3)