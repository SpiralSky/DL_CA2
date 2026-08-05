from typing import Mapping, Any, Iterable


class FitLogger:
    def __init__(
        self,
        verbose: bool = True,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
    ):
        self.verbose = verbose
        self.include = set(include) if include else None
        self.exclude = set(exclude) if exclude else set()

    def log(
        self,
        *,
        epoch: int,
        max_epochs: int,
        train: Mapping[str, Any],
        val: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:

        logs = {
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

        if self.verbose:
            self.print(logs)

        return logs

    def should_log(self, key: str) -> bool:
        if key in self.exclude:
            return False

        if self.include is not None:
            return key in self.include

        return True

    def print(self, logs: Mapping[str, Any]) -> None:
        parts = [
            f"Epoch {logs['epoch']}/{logs['max_epochs']}"
        ]

        for group in ("train", "val", "extra"):
            metrics = []

            for key, value in logs.items():
                if not self.should_log(key):
                    continue

                if group == "train" and key.startswith("val_"):
                    continue

                if group == "val" and not key.startswith("val_"):
                    continue

                if group == "extra" and key not in {"lr", "time"}:
                    continue

                metrics.append(self.format_value(key, value))

            if metrics:
                parts.append(" ".join(metrics))

        print(" | ".join(parts))

    def format_value(self, key: str, value: Any) -> str:
        if key == "lr":
            return f"{key}={value:.2e}"

        if key == "time":
            return f"{key}={value:.1f}s"

        if isinstance(value, float):
            return f"{key}={value:.4f}"

        return f"{key}={value}"