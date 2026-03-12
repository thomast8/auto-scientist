"""Tests for time-window scheduling logic."""

from datetime import datetime, time
from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.scheduler import is_within_window, parse_schedule, seconds_until_window, wait_for_window


class TestParseSchedule:
    def test_valid_schedule(self):
        start, end = parse_schedule("22:00-06:00")
        assert start == time(22, 0)
        assert end == time(6, 0)

    def test_same_day_schedule(self):
        start, end = parse_schedule("09:00-17:00")
        assert start == time(9, 0)
        assert end == time(17, 0)

    def test_with_spaces(self):
        start, end = parse_schedule(" 22:00 - 06:00 ")
        assert start == time(22, 0)
        assert end == time(6, 0)

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_schedule("invalid")

    def test_single_time(self):
        with pytest.raises(ValueError):
            parse_schedule("22:00")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_schedule("")

    def test_midnight_boundary(self):
        start, end = parse_schedule("00:00-23:59")
        assert start == time(0, 0)
        assert end == time(23, 59)


class TestIsWithinWindow:
    def test_overnight_inside_late(self):
        now = datetime(2026, 1, 1, 23, 0)
        assert is_within_window(now, time(22, 0), time(6, 0))

    def test_overnight_inside_early(self):
        now = datetime(2026, 1, 1, 3, 0)
        assert is_within_window(now, time(22, 0), time(6, 0))

    def test_overnight_outside(self):
        now = datetime(2026, 1, 1, 14, 0)
        assert not is_within_window(now, time(22, 0), time(6, 0))

    def test_same_day_inside(self):
        now = datetime(2026, 1, 1, 12, 0)
        assert is_within_window(now, time(9, 0), time(17, 0))

    def test_same_day_outside(self):
        now = datetime(2026, 1, 1, 20, 0)
        assert not is_within_window(now, time(9, 0), time(17, 0))

    def test_boundary_start(self):
        now = datetime(2026, 1, 1, 22, 0)
        assert is_within_window(now, time(22, 0), time(6, 0))

    def test_boundary_end(self):
        now = datetime(2026, 1, 1, 6, 0)
        assert is_within_window(now, time(22, 0), time(6, 0))


class TestSecondsUntilWindow:
    def test_later_today(self):
        now = datetime(2026, 1, 1, 20, 0)
        secs = seconds_until_window(now, time(22, 0))
        assert secs == 7200  # 2 hours

    def test_tomorrow(self):
        now = datetime(2026, 1, 1, 23, 0)
        secs = seconds_until_window(now, time(22, 0))
        assert secs == pytest.approx(23 * 3600, abs=1)  # ~23 hours

    def test_one_minute_window(self):
        now = datetime(2026, 1, 1, 10, 30)
        assert is_within_window(now, time(10, 30), time(10, 31))
        assert not is_within_window(
            datetime(2026, 1, 1, 10, 32), time(10, 30), time(10, 31),
        )


class TestWaitForWindow:
    @pytest.mark.asyncio
    async def test_none_schedule_returns_immediately(self):
        await wait_for_window(None)

    @pytest.mark.asyncio
    @patch("auto_scientist.scheduler.asyncio.sleep", new_callable=AsyncMock)
    @patch("auto_scientist.scheduler.datetime")
    async def test_inside_window_no_sleep(self, mock_dt, mock_sleep):
        mock_dt.now.return_value = datetime(2026, 1, 1, 23, 0)
        await wait_for_window("22:00-06:00")
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch("auto_scientist.scheduler.asyncio.sleep", new_callable=AsyncMock)
    @patch("auto_scientist.scheduler.datetime")
    async def test_outside_window_sleeps(self, mock_dt, mock_sleep):
        mock_dt.now.return_value = datetime(2026, 1, 1, 14, 0)
        await wait_for_window("22:00-06:00")
        mock_sleep.assert_called_once()
        sleep_secs = mock_sleep.call_args[0][0]
        assert sleep_secs > 0
