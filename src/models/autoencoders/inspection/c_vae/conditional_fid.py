import gc

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance

from src.models.autoencoders.BetaConditionalVAE import BetaConditionalVAE


@torch.no_grad()
def calculate_conditional_class_fid(
    model: BetaConditionalVAE,
    dataloader: DataLoader,
    class_labels: list[str],
    *,
    num_images: int = 5000,
    batch_size: int = 64,
) -> pd.DataFrame:

    device = next(model.parameters()).device
    model.eval()

    results = []

    for class_id, class_name in enumerate(class_labels):

        fid = FrechetInceptionDistance(
            feature=2048
        ).to(device)

        count = 0

        # Collect real images of this class
        real_images = []

        for images, labels in dataloader:
            mask = labels == class_id

            if mask.any():
                real_images.append(images[mask])

            if sum(x.size(0) for x in real_images) >= num_images:
                break

        real_images = torch.cat(real_images)[:num_images]

        # Update real images in batches
        for batch in real_images.split(batch_size):
            fid.update(
                batch.to(device)
                .clamp(0, 1)
                .mul(255)
                .byte(),
                real=True,
            )

        del real_images
        gc.collect()
        torch.cuda.empty_cache()

        # Generate and update fake images in batches
        remaining = num_images

        while remaining > 0:
            current = min(batch_size, remaining)

            labels = torch.full(
                (current,),
                class_id,
                device=device,
                dtype=torch.long,
            )

            generated = model.sample(
                current,
                labels,
                device=device,
            )

            fid.update(
                generated.clamp(0, 1)
                .mul(255)
                .byte(),
                real=False,
            )

            count += current
            remaining -= current

            del generated, labels

        results.append(
            {
                "class": class_name,
                "FID": float(fid.compute().cpu()),
                "samples": count,
            }
        )

        del fid
        gc.collect()
        torch.cuda.empty_cache()

    return pd.DataFrame(results)