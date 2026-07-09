from datetime import datetime
from typing import Any

from open_econs._version import __version__


def capture_call(**kwargs: Any) -> dict[str, Any]:
    """Record the calling arguments plus provenance for a result's ``.call``."""
    kwargs["timestamp"] = str(datetime.now())
    kwargs["package_version"] = __version__
    return kwargs
