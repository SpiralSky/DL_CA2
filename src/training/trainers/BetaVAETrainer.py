from src.models.autoencoders.losses.models import vae_loss
from src.training.trainers.VAETrainer import VAETrainer


class BetaVAETrainer(VAETrainer):
    """
    Trainer for conditional beta-VAE models.
    """

    def __init__(
        self,
        model,
        optimizer,
        *,
        beta: float = 1.0,
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
            loss_fn=loss_fn,
            recon_loss_type=recon_loss_type,
            free_bits=free_bits,
            kl_warmup_epochs=kl_warmup_epochs,
        )

        self.beta = beta

    def run_batch(
        self,
        batch,
        device,
        *,
        train: bool,
        epoch: int,
    ):
        """
        Runs a single conditional VAE batch.

        :param batch: Batch containing images and labels.
        :param device: Training device.
        :param train: Whether to update gradients.
        :param epoch: Current epoch.
        :return: Loss dictionary.
        """

        images = batch[0].to(device)
        labels = batch[1].to(device)

        if train:
            self.optimizer.zero_grad()

        recon, mu, logvar = self.model(images, labels)

        losses = self.loss_fn(
            recon,
            images,
            mu,
            logvar,
            recon_loss_type=self.recon_loss_type,
            free_bits=self.free_bits,
            kl_weight=self.get_kl_weight(epoch) * self.beta,
        )

        if train:
            losses["loss"].backward()

            self.clip_gradients()

            self.optimizer.step()

        return losses