import time

import torch

from src.models.util.TrainingHistory import TrainingHistory
from src.training.callbacks.Callback import CallbackSignal


class Trainer:
    """
    Base trainer class.

    Handles training loop mechanics independent of model type.
    """

    def __init__(
        self,
        model,
        optimizer,
        *,
        scheduler=None,
        callbacks=None,
        grad_clip_norm: float | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.grad_clip_norm = grad_clip_norm


    def run_epoch(
        self,
        loader,
        device: torch.device,
        *,
        train: bool,
        epoch: int,
    ) -> dict[str, float]:
        """
        Runs one epoch.

        Implemented by subclasses.
        """
        return {}


    def _run_callbacks(
        self,
        hook: str,
        **kwargs,
    ) -> CallbackSignal:

        signal = CallbackSignal()

        for callback in self.callbacks:

            callback_fn = getattr(
                callback,
                hook,
                None,
            )

            # noinspection calling-non-callable
            result = callback_fn(
                trainer=self,
                **kwargs,
            )

            if result is not None:
                signal.stop_training = (
                    signal.stop_training or result.stop_training
                )

        return signal


    def fit(
        self,
        train_loader,
        val_loader,
        device: torch.device,
        max_epochs: int,
    ) -> TrainingHistory:

        history = TrainingHistory(max_epochs)


        self._run_callbacks(
            "on_train_start",
        )


        for epoch in range(1, max_epochs + 1):

            start = time.time()


            signal = self._run_callbacks("on_epoch_start", epoch=epoch, history=history)

            if signal.stop_training:
                break


            train_metrics = self.run_epoch(train_loader, device, train=True, epoch=epoch)
            val_metrics = self.run_epoch(val_loader, device, train=False, epoch=epoch)


            if self.scheduler is not None:
                self.scheduler.step(val_metrics["loss"])


            metrics = {
                "train": train_metrics,
                "val": val_metrics,
            }


            message = history.update(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                extra_metrics={
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "time": time.time() - start,
                },
            )

            print(message)

            signal = self._run_callbacks("on_epoch_end", epoch=epoch, history=history, metrics=metrics)

            if signal.stop_training:
                break

        self._run_callbacks(
            "on_train_end",
            history=history,
        )

        return history


    def clip_gradients(self) -> None:
        """
        Clips model gradients if gradient clipping is enabled.
        """
        if self.grad_clip_norm is None:
            return

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.grad_clip_norm,
        )