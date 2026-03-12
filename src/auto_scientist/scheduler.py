"""Time-window scheduling logic for token budget management."""

import asyncio
from datetime import datetime, time


def parse_schedule(schedule_str: str) -> tuple[time, time]:
    """Parse a schedule string like '22:00-06:00' into start and end times.

    Args:
        schedule_str: Time window in 'HH:MM-HH:MM' format.

    Returns:
        Tuple of (start_time, end_time).

    Raises:
        ValueError: If the format is invalid.
    """
    parts = schedule_str.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid schedule format: {schedule_str!r}. Expected 'HH:MM-HH:MM'.")

    start = time.fromisoformat(parts[0].strip())
    end = time.fromisoformat(parts[1].strip())
    return start, end


def is_within_window(now: datetime, start: time, end: time) -> bool:
    """Check if the current time falls within the scheduled window.

    Handles overnight windows (e.g., 22:00-06:00) where end < start.
    """
    current = now.time()

    if start <= end:
        # Same-day window (e.g., 09:00-17:00)
        return start <= current <= end
    else:
        # Overnight window (e.g., 22:00-06:00)
        return current >= start or current <= end


def seconds_until_window(now: datetime, start: time) -> float:
    """Calculate seconds until the next window opening.

    Args:
        now: Current datetime.
        start: Window start time.

    Returns:
        Seconds until the window opens. 0 if already in the window.
    """
    today_start = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)

    if today_start > now:
        return (today_start - now).total_seconds()
    else:
        # Next occurrence is tomorrow
        from datetime import timedelta

        tomorrow_start = today_start + timedelta(days=1)
        return (tomorrow_start - now).total_seconds()


async def wait_for_window(schedule_str: str | None) -> None:
    """If a schedule is configured and we're outside the window, sleep until it opens.

    Args:
        schedule_str: Schedule string like '22:00-06:00', or None to skip.
    """
    if schedule_str is None:
        return

    start, end = parse_schedule(schedule_str)
    now = datetime.now()

    if is_within_window(now, start, end):
        return

    wait_seconds = seconds_until_window(now, start)
    hours = wait_seconds / 3600
    print(f"Outside scheduled window ({schedule_str}). Sleeping for {hours:.1f} hours...")
    await asyncio.sleep(wait_seconds)
