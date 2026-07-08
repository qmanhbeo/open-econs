import warnings

import pandas as pd
from formulaic import Formula

from open_econs._internal.errors import (
    empty_data_error,
    missing_column_error,
    rows_dropped_warning,
)


def parse_formula(
    formula: str,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_columns_in_data(formula, data)

    original_n = len(data)
    result = Formula(formula).get_model_matrix(data, na_action="drop")
    yy: pd.DataFrame = result.lhs
    XX: pd.DataFrame = result.rhs

    dropped = original_n - len(yy)
    if dropped > 0:
        cols = _find_missing_cols(formula, data)
        warnings.warn(rows_dropped_warning(dropped, original_n, cols))

    if len(yy) == 0:
        cols = _find_missing_cols(formula, data)
        raise empty_data_error(original_n, dropped, cols)

    return yy, XX


def _validate_columns_in_data(formula: str, data: pd.DataFrame) -> None:
    f = Formula(formula)
    vars_needed: set[str] = {str(v) for v in f.required_variables}
    missing = sorted(v for v in vars_needed if v not in data.columns)
    if missing:
        raise missing_column_error(missing[0], data.columns.tolist())


def _find_missing_cols(formula: str, data: pd.DataFrame) -> list[str]:
    f = Formula(formula)
    vars_needed: set[str] = {str(v) for v in f.required_variables}
    return sorted(v for v in vars_needed if data[v].isna().any())