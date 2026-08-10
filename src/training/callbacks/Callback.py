from dataclasses import dataclass

@dataclass
class CallbackSignal:
    """
    Signal returned by callbacks to control trainer behaviour.
    """

    stop_training: bool = False

# noinspection unused-parameter
class Callback:
    """
    Base callback class.

    Callbacks may override any hook.
    """

    def on_train_start(self, trainer) -> CallbackSignal | None:
        return None


    def on_epoch_start(self, trainer, epoch: int, history) -> CallbackSignal | None:
        return None


    def on_epoch_end(self, trainer, epoch: int, history, metrics: dict[str, dict[str, float]]) -> CallbackSignal | None:
        return None


    def on_train_end(self, trainer, history) -> CallbackSignal | None:
        return None