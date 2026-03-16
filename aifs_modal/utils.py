"""Utils."""

import datetime


def datetime_to_str(date: datetime.datetime) -> str:
    """Convert datetime to a string."""
    assert date.tzinfo == datetime.UTC
    assert date.minute == date.second == date.microsecond == 0
    assert date.hour in [0, 6, 12, 18]
    return date.strftime("%Y-%m-%d/%Hz")
