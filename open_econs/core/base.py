from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """Abstract base for all result models.

    Attributes set during ``__init__`` are *best-effort* immutable after
    ``_freeze()`` is called.  Normal attribute assignment (``r.x = ...``)
    and deletion (``del r.x``) are blocked, but ``object.__setattr__`` /
    ``object.__delattr__`` — which bypass Python-level attribute control —
    cannot be prevented without a C extension.  Consumers should treat
    results as read-only and create a fresh estimate if modification is
    needed.
    """

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

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} results are immutable after fit(). "
                f"Cannot delete '{name}'. Create a new estimate instead."
            )
        super().__delattr__(name)

    def _freeze(self) -> None:
        self._frozen = True
        # Note: object.__setattr__(self, ...) still works.
        # Python cannot prevent this without a C extension.

    def _is_frozen(self) -> bool:
        return self._frozen

    # ── abstract interface ──────────────────────────────────────────

    @abstractmethod
    def tidy(self) -> pd.DataFrame:
        """Coefficient or effect table, one row per term (R-broom style)."""

    @abstractmethod
    def summary(self) -> str:
        """Pretty-printed terminal string."""

    # ── optional stubs (loud) ───────────────────────────────────────

    def predict(self, newdata: pd.DataFrame | None = None) -> pd.Series:
        """Predict on ``newdata`` using the fitted model.

        Not every estimator supports in-sample or out-of-sample prediction;
        results that do not (e.g. most causal/nonlinear estimators) raise
        ``NotImplementedError`` describing which family supports it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support predict(). "
            "predict() is only defined for LinearModel-family results."
        )

    def plot(self) -> None:
        """Render a diagnostic/coefficient plot for this result.

        Not implemented in this version. ``matplotlib`` is intentionally an
        optional extra, not a hard dependency. Call ``.tidy()`` and plot the
        returned DataFrame yourself in the meantime.
        """
        raise NotImplementedError(
            "plot() is not implemented in this version of open-econs. "
            "Planned for a future release; matplotlib will be an optional "
            "extra, not a hard dependency. In the meantime, call .tidy() "
            "and plot the DataFrame yourself."
        )

    def export(self, path: str) -> None:
        """Persist the result to disk.

        Writes ``.json`` (full ``to_dict()`` payload) or ``.csv`` (the
        ``.tidy()`` coefficient table). Other extensions raise
        ``NotImplementedError``. Serialisation uses ``default=str`` so numpy
        scalars and timestamps survive the round-trip.
        """
        import json

        if path.endswith(".json"):
            data = self.to_dict()
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2, default=str)
        elif path.endswith(".csv"):
            self.tidy().to_csv(path, index=False)
        else:
            raise NotImplementedError(
                f"export() to '{path.rsplit('.', 1)[-1]}' is not supported in "
                "this version. Only .json and .csv export are available. "
                "Export with a '.json' or '.csv' extension, or call .tidy() "
                "and save the DataFrame yourself."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the result.

        Includes ``formula``, ``nobs``, ``cov_type``, the captured ``call``,
        ``timestamp``, ``package_version``, and the ``results`` rows built
        from ``.tidy()``.
        """
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

    def to_latex(self, caption: str = "", label: str = "") -> str:
        """Return a LaTeX ``table`` of the ``.tidy()`` coefficient table.

        Pass ``caption`` / ``label`` to wrap the ``tabular`` in a
        ``\\begin{table}`` environment.
        """
        tex = self.tidy().to_latex(index=False)
        if caption or label:
            head = "\\begin{table}\n\\centering\n"
            mid = ""
            if caption:
                mid += f"\\caption{{{caption}}}\n"
            if label:
                mid += f"\\label{{{label}}}\n"
            tex = head + mid + tex + "\\end{table}\n"
        return tex

    def to_html(self, caption: str = "") -> str:
        """Return an HTML table of the ``.tidy()`` coefficient table.

        Pass ``caption`` to embed a ``<caption>`` element.
        """
        html = self.tidy().to_html(index=False)
        if caption:
            html = html.replace("<table", f'<table\n<caption>{caption}</caption>', 1)
        return html

    def __repr__(self) -> str:
        return self.summary()