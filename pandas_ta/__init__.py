from pandas_ta_classic import *  # re-export public API
try:
    # expose version attr if present
    from pandas_ta_classic import version  # noqa: F401
except Exception:  # pragma: no cover
    version = "classic"
