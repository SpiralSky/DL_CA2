from typing import Any
from torch import nn

from src.models.util.TrainingHistory import TrainingHistory


class Callback:
    """
    Base Callback Class.
    """

    def __init__(self, start_epoch: int = 1) -> None:
        """
        Creates a new callback.

        :param start_epoch: Epoch to start callback at.
        """
        self.start_epoch = start_epoch

    def on_epoch_end(
        self,
        epoch: int,
        history: TrainingHistory,
        model: nn.Module,
    ) -> Any:
        """
        Intended to be called every epoch.

        :param epoch: Current epoch value.
        :param history: Training history of the model.
        :param model: Model.
        :return:
        """
        return