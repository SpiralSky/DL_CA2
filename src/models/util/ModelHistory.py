from typing import Any, Mapping


class ModelHistory:
    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def update(
        self,
        *,
        epoch: int,
        max_epochs: int,
        train_metrics: Mapping[str, Any],
        val_metrics: Mapping[str, Any],
        extra_metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:

        logs = {
            "epoch": epoch,
            "max_epochs": max_epochs,

            **{
                f"train_{key}": value
                for key, value in self._format_metrics(train_metrics).items()
            },

            **{
                f"val_{key}": value
                for key, value in self._format_metrics(val_metrics).items()
            },
        }

        if extra_metrics is not None:
            logs.update(extra_metrics)

        self.history.append(logs)

        return logs

    @staticmethod
    def _format_metrics(
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            ("loss" if key == "total" else key): value
            for key, value in metrics.items()
        }

    def format_epoch(
        self,
        index: int = -1,
        *,
        metrics: list[str] | None = None,
    ) -> str:

        logs = self.history[index]

        if metrics is None:
            metrics = [
                key
                for key in logs
                if key not in {"epoch", "max_epochs", "gradients"}
            ]

        values = [
            self._format_value(key, logs[key])
            for key in metrics
            if key in logs
        ]

        message = f"Epoch {logs['epoch']}/{logs['max_epochs']}"

        if values:
            message += " | " + " ".join(values)

        return message

    @staticmethod
    def _format_value(
        key: str,
        value: Any,
    ) -> str:

        display_key = (
            key
            .removeprefix("train_")
            .removeprefix("val_")
        )

        if key == "lr":
            return f"lr={value:.2e}"

        if key == "time":
            return f"time={value:.1f}s"

        if isinstance(value, float):
            return f"{display_key}={value:.4f}"

        return f"{display_key}={value}"

    def values(self, key: str) -> list[Any]:
        return [
            entry[key]
            for entry in self.history
            if key in entry
        ]

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.history[index]