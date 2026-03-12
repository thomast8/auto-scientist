"""Tests for time-window scheduling logic."""

from datetime import datetime, time

import pytest

from auto_scientist.scheduler import is_within_window, parse_schedule, seconds_until_window


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
