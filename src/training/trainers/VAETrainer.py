import torch

from src.models.autoencoders.losses.models.vae_loss import vae_loss
from src.training.trainers.Trainer import Trainer


class VAETrainer(Trainer):
    def __init__(
            self,
            model,
            optimizer,
            *,
            scheduler=None,
            callbacks=None,
            grad_clip_norm=1.0,
            loss_fn=vae_loss,
            recon_loss_type="mse",
            free_bits=0.0,
            kl_warmup_epochs=0,
    ):
        super().__init__(
            model,
            optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            grad_clip_norm=grad_clip_norm,
        )

        self.loss_fn = loss_fn
        self.recon_loss_type = recon_loss_type
        self.free_bits = free_bits
        self.kl_warmup_epochs = kl_warmup_epochs

    def run_epoch(
        self,
        loader,
        device,
        *,
        train: bool,
        epoch: int,
    ):
        self.model.train() if train else self.model.eval()

        totals = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "kl_divergence": 0.0,
        }

        batches = 0

        with torch.enable_grad() if train else torch.no_grad():

            for batch in loader:

                losses = self.run_batch(
                    batch,
                    device,
                    train=train,
                    epoch=epoch,
                )

                for key in totals:
                    totals[key] += losses[key].item()

                batches += 1

        return {
            key: value / batches
            for key, value in totals.items()
        }

    def run_batch(
            self,
            batch,
            device,
            *,
            train: bool,
            epoch: int,
    ):
        images = batch[0].to(device)

        if train:
            self.optimizer.zero_grad()

        recon, mu, logvar = self.model(images)

        losses = self.loss_fn(
            recon,
            images,
            mu,
            logvar,
            recon_loss_type=self.recon_loss_type,
            free_bits=self.free_bits,
            kl_weight=self.get_kl_weight(epoch),
        )

        if train:
            losses["loss"].backward()

            self.clip_gradients()

            self.optimizer.step()

        return losses

    def get_kl_weight(
            self,
            epoch: int,
    ) -> float:

        if self.kl_warmup_epochs <= 0:
            return 1.0

        return min(
            epoch / self.kl_warmup_epochs,
            1.0,
        )