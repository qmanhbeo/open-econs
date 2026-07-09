
from open_econs._internal.errors import (
    missing_column_error,
    non_binary_error,
    empty_data_error,
    cluster_column_error,
    rows_dropped_warning,
)


class TestMissingColumnError:
    def test_single_missing(self):
        err = missing_column_error("income", ["age", "education"])
        assert "income" in str(err)
        assert "age" in str(err)
        assert "education" in str(err)

    def test_message_format(self):
        err = missing_column_error("x", ["a", "b", "c"])
        msg = str(err)
        assert "not found" in msg
        assert "Available" in msg


class TestClusterColumnError:
    def test_message(self):
        err = cluster_column_error("mycluster", ["a", "b"])
        assert "mycluster" in str(err)
        assert "a" in str(err) or "Available" in str(err)


class TestNonBinaryError:
    def test_message(self):
        err = non_binary_error("female", [0, 1, 2])
        assert "female" in str(err)
        assert "binary" in str(err)
        assert "3" in str(err) or "0, 1, 2" in str(err)


class TestEmptyDataError:
    def test_simple(self):
        err = empty_data_error(100)
        assert "0 rows" in str(err)
        assert "100" in str(err)

    def test_with_dropped(self):
        err = empty_data_error(100, dropped=25, cols=["age", "income"])
        assert "25" in str(err)
        assert "age" in str(err)


class TestRowsDroppedWarning:
    def test_message(self):
        msg = rows_dropped_warning(5, 100, ["age", "income"])
        assert "5" in msg
        assert "100" in msg
        assert "age" in msg