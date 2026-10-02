import sqlite3

import pytest

from conftest import PLANT_DB, needs_data
from shopfloor_agent.data.build import hours, iso_datetime, us_datetime
from shopfloor_agent.data.sources import SOURCES, Source, verify
from shopfloor_agent.servers.store import ToolError, parse_bound


def test_us_dates_and_durations() -> None:
    assert us_datetime("4/6/16 14:00") == "2016-04-06T14:00:00"
    assert us_datetime("12/31/09 9:05") == "2009-12-31T09:05:00"
    assert us_datetime("") is None
    assert iso_datetime("2010-06-22 14:12:00") == "2010-06-22T14:12:00"
    assert hours("3:00") == 3.0
    assert hours("7:30") == 7.5
    assert hours("40:00:00") == 40.0  # a few rows use hours:minutes:seconds
    assert hours("") is None


def test_date_bounds_include_the_whole_end_period() -> None:
    assert parse_bound("2017", end=False) == "2017-01-01T00:00:00"
    assert parse_bound("2017", end=True) == "2018-01-01T00:00:00"
    assert parse_bound("2020-12", end=True) == "2021-01-01T00:00:00"
    assert parse_bound("2020-06-07", end=True) == "2020-06-08T00:00:00"
    assert parse_bound("2020-06-07T10:30", end=True) == "2020-06-07T10:30:00"
    assert parse_bound(None, end=True) is None
    with pytest.raises(ToolError, match="invalid date"):
        parse_bound("last week", end=False)


def test_sources_are_pinned(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert len({s.filename for s in SOURCES}) == len(SOURCES)
    assert all(len(s.sha256) == 64 and s.size > 0 for s in SOURCES)
    path = tmp_path / "x.csv"
    path.write_bytes(b"abc")
    sha_abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert verify(Source("a/x.csv", 3, sha_abc), path)
    assert not verify(Source("a/x.csv", 3, "0" * 64), path)


@needs_data
def test_plant_db_contents() -> None:
    db = sqlite3.connect(PLANT_DB)
    count = lambda sql: db.execute(sql).fetchone()[0]  # noqa: E731
    assert count("SELECT COUNT(*) FROM equipment") == 11
    assert count("SELECT COUNT(*) FROM work_orders") == 4249
    assert count("SELECT COUNT(*) FROM work_orders WHERE work_type = 'CM'") == 720
    assert count("SELECT COUNT(*) FROM events") == 6256
    assert count("SELECT COUNT(*) FROM alerts") == 1466
    assert count("SELECT COUNT(*) FROM work_orders WHERE finished_at IS NULL") == 0
    # every work order and alert points at known equipment
    assert (
        count(
            "SELECT COUNT(*) FROM work_orders WHERE equipment_id NOT IN "
            "(SELECT equipment_id FROM equipment)"
        )
        == 0
    )
    assert count("SELECT COUNT(*) FROM telemetry") == 28775
    assert count("SELECT MIN(ts) FROM telemetry") == "2020-06-01T00:00:00"
