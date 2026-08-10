import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from src.models.util.TrainingHistory import TrainingHistory
from training.callbacks.EarlyStopping import EarlyStopping


class VectorQuantizer(nn.Module):
    """
    VQ-VAE codebook layer (van den Oord et al., 2017).
    Maps each continuous latent vector to its nearest embedding in a learned
    codebook, using a straight-through estimator so gradients can still flow
    back through the (non-differentiable) argmin lookup.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        """
        :param num_embeddings: Codebook size (number of discrete codes).
        :param embedding_dim: Dimensionality of each code / the latent vector.
        :param commitment_cost: Weight on the encoder "commitment" term.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor):
        """
        :param z: Continuous encoder output, shape (batch_size, embedding_dim).
        :return: (quantized, vq_loss, encoding_indices)
        """
        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z @ self.embedding.weight.t()
        )
        encoding_indices = distances.argmin(dim=1)
        quantized = self.embedding(encoding_indices)

        codebook_loss = functional.mse_loss(quantized, z.detach())
        commitment_loss = functional.mse_loss(quantized.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: forward pass uses `quantized`,
        # backward pass copies gradients straight through to `z`.
        quantized = z + (quantized - z).detach()

        return quantized, vq_loss, encoding_indices


class VQVAE(nn.Module):
    """
    Conditional VQ-VAE. Standalone (does not share a base class with the
    Gaussian VAEs elsewhere in this codebase) since it has no mu/logvar,
    reparameterization, or KL divergence -- the continuous encoder output
    is snapped to a learned codebook instead.

    NOTE: expects `encoder(images, embeddings)` to return a single
    continuous latent tensor of shape (batch_size, latent_dim), unlike the
    (mu, logvar) tuple used by the Gaussian VAEs' encoders.
    """
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim: int,
        num_classes: int,
        num_embeddings: int = 512,
        label_embed_dim: int = 16,
        commitment_cost: float = 0.25,
    ):
        """
        :param encoder: Module mapping (images, label_embeddings) -> continuous latent, shape (batch_size, latent_dim).
        :param decoder: Module mapping (quantized_latent, label_embeddings) -> reconstruction.
        :param latent_dim: Latent / codebook embedding dimension.
        :param num_classes: Number of conditioning classes.
        :param num_embeddings: Codebook size (number of discrete codes).
        :param label_embed_dim: Dimensionality of the label embedding.
        :param commitment_cost: Weight on the encoder "commitment" term in the VQ loss.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        self.label_embeddings = nn.Embedding(num_classes, label_embed_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        """
        :param images: Image batch, shape (batch_size, C, H, W).
        :param labels: Class labels, shape (batch_size,) or (batch_size, 1).
        :return: (reconstruction, vq_loss, encoding_indices)
        """
        labels = labels.squeeze().long()
        embeddings = self.label_embeddings(labels)
        z = self.encoder(images, embeddings)
        quantized, vq_loss, encoding_indices = self.quantizer(z)
        recon = self.decoder(quantized, embeddings)
        return recon, vq_loss, encoding_indices

    @torch.no_grad()
    def sample(self, num_samples: int, labels: torch.Tensor, device: torch.device | None = None):
        """
        Draws random codebook indices and decodes them. NOTE: there's no
        learned prior over codes here, so this is a known simplification --
        expect incoherent samples compared to a proper VQ-VAE prior
        (e.g. PixelCNN over code indices).
        """
        device = device or next(self.parameters()).device
        indices = torch.randint(0, self.quantizer.num_embeddings, (num_samples,), device=device)
        z = self.quantizer.embedding(indices)
        y = self.label_embeddings(labels.to(device))
        return self.decoder(z, y)

    @staticmethod
    def reconstruction_loss(recon: torch.Tensor, x: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
        """
        :param recon: Reconstructed images.
        :param x: Target images.
        :param loss_type: Either "mse" or "bce".
        """
        if loss_type == "mse":
            return functional.mse_loss(recon, x, reduction="sum") / x.size(0)
        if loss_type == "bce":
            return functional.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
        raise ValueError(f"Unknown recon_loss_type '{loss_type}', expected 'mse' or 'bce'")

    def get_loss(self, recon, images, vq_loss, *, recon_loss_type: str = "mse") -> dict[str, torch.Tensor]:
        recon_loss = self.reconstruction_loss(recon, images, recon_loss_type)
        return {
            "loss": recon_loss + vq_loss,
            "reconstruction": recon_loss,
            "vq_loss": vq_loss,
        }

    def run_epoch(
        self,
        loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        grad_clip_norm: float,
        recon_loss_type: str,
        train: bool,
    ) -> dict[str, float]:
        self.train() if train else self.eval()

        totals = {"loss": 0.0, "reconstruction": 0.0, "vq_loss": 0.0}
        num_batches = 0

        with torch.enable_grad() if train else torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)

                if train:
                    optimizer.zero_grad()

                recon, vq_loss, _ = self(images, labels)
                losses = self.get_loss(recon, images, vq_loss, recon_loss_type=recon_loss_type)

                if train:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

                for key in totals:
                    totals[key] += losses[key].item()
                num_batches += 1

        return {key: value / num_batches for key, value in totals.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        max_epochs: int,
        lr: float,
        grad_clip_norm: float,
        recon_loss_type: str = "mse",
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> TrainingHistory:
        """
        Self-contained training loop -- doesn't share fit()/run_epoch() with
        the Gaussian VAEs elsewhere in the project, since there's no KL
        term/warmup/beta to plumb through here.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        if early_stopping is None:
            early_stopping = EarlyStopping(
                monitor=("val", "loss"),
                start_epoch=30,
            )

        history = TrainingHistory(max_epochs)

        for epoch in range(1, max_epochs + 1):
            train_metrics = self.run_epoch(
                train_loader, device, optimizer, grad_clip_norm, recon_loss_type, train=True
            )
            val_metrics = self.run_epoch(
                val_loader, device, optimizer, grad_clip_norm, recon_loss_type, train=False
            )

            scheduler.step(val_metrics["loss"])

            message = history.update(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                extra_metrics={"lr": optimizer.param_groups[0]["lr"]},
            )
            print(message)

            if stop_message := early_stopping.on_epoch_end(epoch, history, self):
                print(stop_message)
                break

        return history