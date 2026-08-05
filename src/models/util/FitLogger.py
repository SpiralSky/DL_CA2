from typing import Mapping, Any


class FitLogger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history: list[dict[str, Any]] = []

    def log(
        self,
        *,
        epoch: int,
        max_epochs: int,
        train: Mapping[str, Any],
        val: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:

        logs: dict[str, Any] = {
            "epoch": epoch,
            "max_epochs": max_epochs,
            **{
                ("loss" if k == "total" else k): v
                for k, v in train.items()
            },
            **{
                f"val_{'loss' if k == 'total' else k}": v
                for k, v in val.items()
            },
        }

        if extra:
            logs.update(extra)

        self.history.append(logs)

        if self.verbose:
            self.print(logs)

        return logs

    def print(self, logs: Mapping[str, Any]) -> None:
        train = []
        val = []
        extra = []

        for key, value in logs.items():
            if key in {"epoch", "max_epochs"}:
                continue

            item = self.format_value(key, value)

            if key.startswith("val_"):
                val.append(item)
            elif key in {"lr", "time"}:
                extra.append(item)
            else:
                train.append(item)

        message = f"Epoch {logs['epoch']}/{logs['max_epochs']}"

        if train:
            message += " | " + " ".join(train)

        if val:
            message += " | " + " ".join(val)

        if extra:
            message += " | " + " ".join(extra)

        print(message)

    def format_value(self, key: str, value: Any) -> str:
        if key == "lr":
            return f"lr={value:.2e}"

        if key == "time":
            return f"time={value:.1f}s"

        if isinstance(value, float):
            return f"{key}={value:.4f}"

        return f"{key}={value}"