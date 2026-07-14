from collections.abc import Sequence


def missing_column_error(name: str, available: Sequence[str]) -> ValueError:
    available_str = ", ".join(sorted(available))
    return ValueError(
        f"Column '{name}' not found in DataFrame. "
        f"Available columns: [{available_str}]"
    )


def cluster_column_error(name: str, available: Sequence[str]) -> ValueError:
    available_str = ", ".join(sorted(available))
    return ValueError(
        f"Cluster column '{name}' not found in DataFrame. "
        f"Available columns: [{available_str}]"
    )


def non_binary_error(name: str, values: Sequence) -> ValueError:
    found = ", ".join(repr(v) for v in values)
    return ValueError(
        f"Column '{name}' must be binary (exactly 2 unique non-null values). "
        f"Found {len(values)} unique values: [{found}]"
    )


def empty_data_error(original_n: int, dropped: int = 0, cols: list[str] | None = None) -> ValueError:
    msg = f"DataFrame has 0 rows after dropping missing values. Started with {original_n} rows"
    if dropped:
        cols_str = ", ".join(cols or ["unknown"])
        msg += f", {dropped} had missing values in: [{cols_str}]"
    msg += "."
    return ValueError(msg)


def singular_matrix_error() -> RuntimeError:
    return RuntimeError(
        "Singular design matrix — check for collinear or duplicated variables in your formula."
    )


def rows_dropped_warning(dropped: int, total: int, cols: list[str]) -> str:
    cols_str = ", ".join(sorted(cols))
    return (
        f"Dropped {dropped} of {total} rows due to missing values in: [{cols_str}]"
    )


class VcovTypeNotSupportedError(ValueError):
    """Raised when a covariance type is not supported for the given model.

    This error is raised when a user requests HC2 or HC3 standard errors on a
    model with absorbed fixed effects (FE).  HC2/HC3 apply leverage-based
    corrections (dividing by ``1 - h_ii``) that are statistically invalid once
    fixed effects have been absorbed — the hat-matrix diagonal ``h_ii`` no
    longer reflects the true projection because the FE projection removes
    degrees of freedom in a way that HC2/HC3's pointwise leverage adjustment
    does not account for.

    Use ``cov_type="HC1"`` instead, which applies the simpler ``N/(N-K)``
    correction and is valid post-absorption.  For clustered standard errors, use
    ``cluster=...`` instead.
    """

    def __init__(self, cov_type: str) -> None:
        msg = (
            f"cov_type={cov_type!r} is not supported for models with absorbed "
            f"fixed effects. The leverage adjustments in HC2/HC3 are invalid "
            f"once fixed effects are absorbed. Use cov_type='HC1' instead "
            f"(or cov_type='CRV1' / cluster=... for clustered SEs). "
            f"See CHANGELOG for details."
        )
        super().__init__(msg)
        self.cov_type = cov_type