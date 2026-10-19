from datetime import datetime, timezone

from src.core.scheduler import _cron_field_matches, _parse_cron, _should_run


def test_parse_cron_valid():
    result = _parse_cron("0 */6 * * *")
    assert result == {
        "minute": "0",
        "hour": "*/6",
        "day_of_month": "*",
        "month": "*",
        "day_of_week": "*",
    }


def test_parse_cron_invalid():
    try:
        _parse_cron("bad")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_cron_field_star():
    assert _cron_field_matches("*", 0) is True
    assert _cron_field_matches("*", 23) is True


def test_cron_field_step():
    assert _cron_field_matches("*/6", 0) is True
    assert _cron_field_matches("*/6", 6) is True
    assert _cron_field_matches("*/6", 12) is True
    assert _cron_field_matches("*/6", 3) is False


def test_cron_field_literal():
    assert _cron_field_matches("5", 5) is True
    assert _cron_field_matches("5", 6) is False


def test_cron_field_comma():
    assert _cron_field_matches("1,3,5", 3) is True
    assert _cron_field_matches("1,3,5", 4) is False


def test_cron_field_range():
    assert _cron_field_matches("1-5", 3) is True
    assert _cron_field_matches("1-5", 0) is False
    assert _cron_field_matches("1-5", 6) is False


def test_should_run_every_6_hours():
    # minute=0, hour divisible by 6
    now = datetime(2025, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
    assert _should_run("0 */6 * * *", now) is True

    # minute=0, hour=3 — should not match
    now = datetime(2025, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
    assert _should_run("0 */6 * * *", now) is False

    # minute=5 — should not match
    now = datetime(2025, 1, 15, 6, 5, 0, tzinfo=timezone.utc)
    assert _should_run("0 */6 * * *", now) is False


def test_should_run_every_minute():
    now = datetime(2025, 1, 15, 14, 37, 0, tzinfo=timezone.utc)
    assert _should_run("* * * * *", now) is True


def test_should_run_midnight_daily():
    now = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    assert _should_run("0 0 * * *", now) is True

    now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert _should_run("0 0 * * *", now) is False


def test_should_run_invalid_cron():
    now = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    assert _should_run("bad expression", now) is False
