from datetime import datetime
from typing import TypeVar

T = TypeVar("T")


def to_datetime(value: str | int) -> int | datetime:
    """Convert a string in ISO format or an integer timestamp to a datetime object or an integer timestamp.

    Args:
        value: A string in ISO format (e.g., "2023-01-01T00:00:00")
            or an integer timestamp (e.g., 1672531200).

    Returns:
        A datetime object if the input is a string,
            or an integer timestamp if the input is an integer.
    """
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    if isinstance(value, int):
        return value
    raise TypeError(f"Unsupported type for datetime conversion: {type(value)}")


def get_corresponding_value(
    current_time: int | str,
    time_span2value: dict[tuple[int | str, int | str], T],
    default_value: T,
) -> T:
    """Get the value corresponding to the current time within the given time spans.

    Args:
        current_time: The current time as an integer timestamp or a string in ISO format.
        time_span2value: A dictionary mapping time spans to values.
            Ex): {
                ("2023-01-01 00:00:00", "2023-02-01 00:00:00"): "Value for January",
                ("2023-02-01 00:00:00", "2023-03-01 00:00:00"): "Value for February",
            }
        default_value: The default value to return if no matching time span is found.

    Returns:
        The value corresponding to the current time, or the default value if no match is found.
    """

    current_dt: int | datetime = to_datetime(current_time)
    for time_span, value in time_span2value.items():
        start_raw, end_raw = time_span
        start_dt: int | datetime = to_datetime(start_raw)
        end_dt: int | datetime = to_datetime(end_raw)
        if (
            isinstance(current_dt, int)
            and isinstance(start_dt, int)
            and isinstance(end_dt, int)
        ):
            if start_dt <= current_dt < end_dt:
                return value
        elif (
            isinstance(current_dt, datetime)
            and isinstance(start_dt, datetime)
            and isinstance(end_dt, datetime)
        ):
            if start_dt <= current_dt < end_dt:
                return value
        else:
            raise TypeError("Inconsistent types for current_time and time spans.")
    return default_value
