from typing import Any, Mapping, Literal

import pandas as pd


class TrainingHistory:
    """
    A class to store history during training.
    Also provides formatting functions.
    """
    def __init__(self, max_epochs: int) -> None:
        """
        :param max_epochs: Max epochs for this history.
        """
        self.max_epochs: int = max_epochs
        self.train: pd.DataFrame = pd.DataFrame()
        self.val: pd.DataFrame = pd.DataFrame()
        self.extra: pd.DataFrame = pd.DataFrame()

    def update(
        self,
        *,
        epoch: int,
        train_metrics: Mapping[str, Any],
        val_metrics: Mapping[str, Any],
        extra_metrics: Mapping[str, Any] | None = None,
    ) -> str:
        """
        Adds a new entry to training history with metrics.
        :param epoch: Current training epoch (1-indexed).
        :param train_metrics: Training Metrics
        :param val_metrics: Validation Metrics
        :param extra_metrics: Extra Metrics
        :return:
        """
        self.train.loc[epoch, list(train_metrics.keys())] = dict(train_metrics)
        self.val.loc[epoch, list(val_metrics.keys())] = dict(val_metrics)

        if extra_metrics:
            self.extra.loc[epoch, list(extra_metrics.keys())] = dict(extra_metrics)

        return self.format_epoch(epoch)

    def _resolve_epoch(self, epoch: int) -> int:
        return epoch if epoch > 0 else self.train.index[epoch]

    def format_epoch(
        self,
        epoch: int = -1,
        *,
        metrics: list[str] | None = None,
    ) -> str:
        """
        Formats metrics for the epoch and outputs them by classes.
        :param epoch: Epoch number (1-indexed). Negative values index
        from the end (e.g. -1 is the most recent epoch).
        :param metrics: Selective Metrics to output.
        If specified, matches all metrics with the same name in
        train, validation and extra metrics.
        :return:
        """
        epoch = self._resolve_epoch(epoch)
        message = f"Epoch {epoch}/{self.max_epochs}"

        for label, df in (("train", self.train), ("val", self.val), ("extra", self.extra)):
            if epoch not in df.index:
                continue

            row: dict[str, Any] = df.loc[epoch].dropna().to_dict()

            if metrics is not None:
                row = {k: v for k, v in row.items() if k in metrics}

            values: list[str] = [self._format_value(k, v) for k, v in row.items()]

            if values:
                message += f" | {label}: " + " ".join(values)

        return message

    @staticmethod
    def _format_value(
        key: str,
        value: Any,
    ) -> str:
        """
        Formats value decimal point precision and outputs in a specific format.
        Intended to only be called internally by TrainingHistory.
        :param key: Name of the metric (e.g. loss).
        :param value: Value of the metric.
        :return: Formatted value: key=value
        """
        if key == "lr":
            return f"lr={value:.2e}"

        if key == "time":
            return f"time={value:.1f}s"

        if isinstance(value, float):
            return f"{key}={value:.4f}"

        return f"{key}={value}"

    def values(
        self,
        key: str,
        split: Literal["train", "val", "extra"],
    ) -> list[Any]:
        """
        Gets all values of a specific metric from a specific split.
        :param key: Name of the metric.
        :param split: Which metrics group to read from.
        :return: List of values of the metric.
        """
        df = getattr(self, split)
        if key in df.columns:
            return df[key].dropna().tolist()
        return []

    # TODO: Fix docstring
    def epochs(
        self,
        key: str,
        split: Literal["train", "val", "extra"],
    ) -> list[int]:
        """
        Gets the epoch numbers corresponding to `values(key, split)`, i.e.
        the epochs at which `key` has a non-null entry in `split`. Kept in
        lockstep with `values()` (same dropna) so the two line up even if
        a metric has gaps (missing epochs) relative to others in the split.
        :param key: Name of the metric.
        :param split: Which metrics group to read from.
        :return: List of epoch numbers (the df index) where the metric has a value.
        """
        df = getattr(self, split)
        if key in df.columns:
            return df[key].dropna().index.tolist()
        return []

    def __len__(self) -> int:
        return len(self.train)

    def __getitem__(self, index: int) -> dict[str, Any]:
        epoch = self._resolve_epoch(index)
        return {
            "epoch": epoch,
            "max_epochs": self.max_epochs,
            "train": self.train.loc[epoch].dropna().to_dict(),
            "val": self.val.loc[epoch].dropna().to_dict(),
            "extra": (
                self.extra.loc[epoch].dropna().to_dict()
                if epoch in self.extra.index
                else {}
            ),
        }

    def __str__(self) -> str:
        return "\n".join(self.format_epoch(epoch) for epoch in self.train.index)