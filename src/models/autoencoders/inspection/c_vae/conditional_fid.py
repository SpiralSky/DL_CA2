import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance
from src.models.autoencoders.VAE import VAE


@torch.no_grad()
def calculate_conditional_class_fid(
    model: VAE,
    dataloader: DataLoader,
    class_labels: list[str],
    *,
    num_images: int = 5000,
) -> pd.DataFrame:

    device = next(model.parameters()).device
    model.eval()

    results = []

    for class_id, class_name in enumerate(class_labels):

        labels = torch.full(
            (num_images,),
            class_id,
            device=device,
            dtype=torch.long,
        )

        generated = model.sample(
            num_images,
            labels,
            device=device,
        )

        # collect real images of same class
        real_images = []

        for images, batch_labels in dataloader:
            mask = batch_labels == class_id

            if mask.any():
                real_images.append(
                    images[mask]
                )

            if sum(x.size(0) for x in real_images) >= num_images:
                break

        real_images = torch.cat(real_images)[:num_images]
        real_images = real_images.to(device)

        fid = FrechetInceptionDistance(
            feature=2048
        ).to(device)

        fid.update(
            real_images.clamp(0,1)
            .mul(255)
            .byte(),
            real=True,
        )

        fid.update(
            generated.clamp(0,1)
            .mul(255)
            .byte(),
            real=False,
        )

        results.append(
            {
                "class": class_name,
                "FID": float(fid.compute().cpu()),
                "samples": num_images,
            }
        )

    return pd.DataFrame(results)