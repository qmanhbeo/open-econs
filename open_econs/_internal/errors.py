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