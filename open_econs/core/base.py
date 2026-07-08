from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd
class BaseModel(ABC):
    formula: str = ""
    data_shape: tuple[int, int] = (0, 0)
    cov_type: str = ""
    call: dict[str, Any] = {}
    timestamp: datetime = datetime.fromtimestamp(0)
    package_version: str = ""

    _frozen: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} results are immutable after fit(). "
                f"Cannot set '{name}'. Create a new estimate instead."
            )
        super().__setattr__(name, value)

    def _freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)

    # ── abstract interface ──────────────────────────────────────────

    @abstractmethod
    def tidy(self) -> pd.DataFrame:
        """Coefficient or effect table, one row per term (R-broom style)."""

    @abstractmethod
    def summary(self) -> str:
        """Pretty-printed terminal string."""

    # ── optional stubs (loud) ───────────────────────────────────────

    def predict(self, newdata: pd.DataFrame | None = None) -> pd.Series:
        raise NotImplementedError(
            f"{type(self).__name__} does not support predict(). "
            "predict() is only defined for LinearModel-family results."
        )

    def plot(self) -> None:
        raise NotImplementedError(
            "plot() is not implemented in this version of open-econs. "
            "Planned for a future release; matplotlib will be an optional "
            "extra, not a hard dependency. In the meantime, call .tidy() "
            "and plot the DataFrame yourself."
        )

    def export(self, path: str) -> None:
        import json

        if not path.endswith(".json"):
            raise NotImplementedError(
                f"export() to '{path.rsplit('.', 1)[-1]}' is not supported in "
                "this version. Only .json export is available in v0.1. "
                "Export with a '.json' extension, or call .tidy() and save "
                "the DataFrame yourself."
            )
        data = self.to_dict()
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)

    def to_dict(self) -> dict[str, Any]:
        d = self.tidy().to_dict(orient="records")
        return {
            "formula": self.formula,
            "nobs": self.data_shape[0],
            "cov_type": self.cov_type,
            "call": {k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v) for k, v in self.call.items()},
            "timestamp": str(self.timestamp),
            "package_version": self.package_version,
            "results": d,
        }

    def __repr__(self) -> str:
        return self.summary()