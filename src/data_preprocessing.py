import pandas as pd

from .config import INPUT_RANGES, RAW_FEATURES


def validate_inputs(record: dict) -> list[str]:
    errors = []
    for feature in RAW_FEATURES:
        value = record.get(feature)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            errors.append(f"{feature}: missing value")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{feature}: must be numeric, got {value!r}")
            continue
        lo, hi = INPUT_RANGES[feature]
        span = hi - lo
        if not (lo - 0.2 * span <= number <= hi + 0.2 * span):
            errors.append(
                f"{feature}: {number} outside plausible range [{lo}, {hi}]"
            )
    return errors


def coerce_record(record: dict) -> dict:
    return {feature: float(record[feature]) for feature in RAW_FEATURES}
