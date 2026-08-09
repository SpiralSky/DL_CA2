import gc

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.autoencoders.AbstractVAE import AbstractVAE


@torch.no_grad()
def calculate_class_fid(
    model: AbstractVAE,
    dataloader: DataLoader,
    class_labels: list[str],
    *,
    num_images: int | None = None,
    chunk_size: int | None = None,
) -> pd.DataFrame:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    device = next(model.parameters()).device
    model.eval()

    if chunk_size is None:
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3 if torch.cuda.is_available() else 0

        chunk_size = 8 if free_gb >= 6 else 4 if free_gb >= 3 else 2

    results = []

    # Overall FID
    fid = FrechetInceptionDistance(feature=2048).to(device)
    count = 0

    for images, *_ in dataloader:
        images = images.to(device)
        recon, *_ = model(images)

        real = images.clamp(0, 1).mul(255).to(torch.uint8)
        fake = recon.clamp(0, 1).mul(255).to(torch.uint8)

        fid.update(real, real=True)
        fid.update(fake, real=False)

        count += images.size(0)

        if num_images and count >= num_images:
            break

    results.append(
        {
            "class": "all",
            "FID": float(fid.compute().cpu()),
            "samples": count,
        }
    )

    del fid
    gc.collect()
    torch.cuda.empty_cache()

    # Per-class FID
    class_results = {}

    for start in range(0, len(class_labels), chunk_size):
        ids = range(start, min(start + chunk_size, len(class_labels)))

        fids = {
            i: FrechetInceptionDistance(feature=2048).to(device)
            for i in ids
        }

        counts = {i: 0 for i in ids}

        for images, labels, *_ in dataloader:
            images, labels = images.to(device), labels.to(device)

            recon, *_ = model(images)

            real = images.clamp(0, 1).mul(255).to(torch.uint8)
            fake = recon.clamp(0, 1).mul(255).to(torch.uint8)

            for i in labels.unique():
                i = int(i)

                if i not in fids:
                    continue

                mask = labels == i

                fids[i].update(real[mask], real=True)
                fids[i].update(fake[mask], real=False)

                counts[i] += int(mask.sum())

        for i in ids:
            class_results[i] = {
                "class": class_labels[i],
                "FID": float(fids[i].compute().cpu()) if counts[i] else float("nan"),
                "samples": counts[i],
            }

        del fids

        gc.collect()
        torch.cuda.empty_cache()

    results.extend(class_results[i] for i in range(len(class_labels)))

    return pd.DataFrame(results)