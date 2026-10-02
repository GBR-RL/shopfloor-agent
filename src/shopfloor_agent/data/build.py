"""Builds the plant database (SQLite) from the pinned AssetOpsBench files.

One file, rebuilt deterministically; evaluation episodes work on copies, so write tools
(new or closed work orders) never leak from one task into the next.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from shopfloor_agent.data.sources import COMMIT, REPO

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE equipment (
    equipment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    asset_class TEXT NOT NULL
);
CREATE TABLE components (
    asset_class TEXT NOT NULL,
    component TEXT NOT NULL,
    explanation TEXT NOT NULL,
    PRIMARY KEY (asset_class, component)
);
CREATE TABLE failure_codes (
    secondary_code TEXT PRIMARY KEY,
    secondary_description TEXT NOT NULL,
    primary_code TEXT NOT NULL,
    primary_description TEXT NOT NULL,
    category TEXT NOT NULL
);
CREATE TABLE failure_modes (
    asset_class TEXT NOT NULL,
    description TEXT NOT NULL
);
CREATE TABLE work_orders (
    wo_id TEXT PRIMARY KEY,
    equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
    description TEXT NOT NULL,
    component TEXT NOT NULL,
    primary_code TEXT,
    secondary_code TEXT,
    work_type TEXT NOT NULL CHECK (work_type IN ('PM', 'CM')),
    priority INTEGER NOT NULL,
    status TEXT NOT NULL,
    reported_at TEXT,
    finished_at TEXT,
    duration_h REAL,
    labor_h REAL
);
CREATE INDEX wo_equipment_finished ON work_orders(equipment_id, finished_at);
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
    event_group TEXT NOT NULL,
    event_category TEXT NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    event_time TEXT NOT NULL,
    note TEXT NOT NULL
);
CREATE INDEX events_equipment_time ON events(equipment_id, event_time);
CREATE TABLE alert_rules (rule_id TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE alert_rule_failure_codes (
    rule_id TEXT NOT NULL REFERENCES alert_rules(rule_id),
    primary_code TEXT NOT NULL,
    PRIMARY KEY (rule_id, primary_code)
);
CREATE TABLE alerts (
    alert_id INTEGER PRIMARY KEY,
    equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
    rule_id TEXT NOT NULL REFERENCES alert_rules(rule_id),
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL
);
CREATE INDEX alerts_equipment_start ON alerts(equipment_id, start_time);
CREATE TABLE anomaly_failure_codes (
    kpi TEXT NOT NULL,
    anomaly_type TEXT NOT NULL,
    category TEXT NOT NULL,
    primary_code TEXT NOT NULL,
    secondary_code TEXT NOT NULL
);
CREATE TABLE sensors (
    sensor_id TEXT PRIMARY KEY,
    equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
    name TEXT NOT NULL
);
CREATE TABLE telemetry (
    sensor_id TEXT NOT NULL REFERENCES sensors(sensor_id),
    ts TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (sensor_id, ts)
);
"""


def us_datetime(value: str) -> str | None:
    """'4/6/16 14:00' (month/day/two-digit year) -> '2016-04-06T14:00:00'."""
    value = value.strip()
    if not value:
        return None
    return datetime.strptime(value, "%m/%d/%y %H:%M").isoformat()


def iso_datetime(value: str) -> str:
    """'2010-06-22 14:12:00' or '2020-06-01T00:15:00' -> second-resolution ISO format."""
    return datetime.fromisoformat(value.strip()).replace(microsecond=0).isoformat()


def hours(value: str) -> float | None:
    """'3:00' (hours:minutes) or '40:00:00' (hours:minutes:seconds) -> hours as a float."""
    value = value.strip()
    if not value:
        return None
    parts = [int(p) for p in value.split(":")]
    return float(sum(p / 60**i for i, p in enumerate(parts)))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in rows]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _insert(db: sqlite3.Connection, table: str, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0])
    db.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    return len(rows)


def build(raw: dict[str, Path], out: Path) -> dict[str, int]:
    """Writes the database to `out` (replacing it) and returns the row count per table."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    try:
        db.executescript(SCHEMA)
        counts = _load(db, raw)
        db.commit()
    finally:
        db.close()
    tmp.replace(out)
    return counts


def _load(db: sqlite3.Connection, raw: dict[str, Path]) -> dict[str, int]:
    work_orders = _rows(raw["all_wo_with_code_component_events.csv"])
    events = _rows(raw["event.csv"])
    alerts = _rows(raw["alert_events.csv"])

    equipment = {r["equipment_id"]: r["equipment_name"] for r in [*work_orders, *events, *alerts]}
    counts = {
        "meta": _insert(
            db,
            "meta",
            [
                {"key": "source", "value": f"https://github.com/{REPO}/tree/{COMMIT}"},
                {"key": "license", "value": "Apache-2.0 (IBM AssetOpsBench sample data)"},
            ],
        ),
        "equipment": _insert(
            db,
            "equipment",
            (
                {"equipment_id": k, "name": v, "asset_class": "chiller"}
                for k, v in sorted(equipment.items())
            ),
        ),
        "components": _insert(
            db,
            "components",
            (
                {
                    "asset_class": r["equipment"],
                    "component": r["component"],
                    "explanation": r["explanation"],
                }
                for r in _rows(raw["component.csv"])
            ),
        ),
        "failure_codes": _insert(
            db,
            "failure_codes",
            (
                {
                    "secondary_code": r["secondary_code"],
                    "secondary_description": r["secondary_code_description"],
                    "primary_code": r["primary_code"],
                    "primary_description": r["primary_code_description"],
                    "category": r["category"],
                }
                for r in _rows(raw["failure_codes.csv"])
            ),
        ),
    }
    modes = yaml.safe_load(raw["failure_modes.yaml"].read_text(encoding="utf-8"))
    counts["failure_modes"] = _insert(
        db,
        "failure_modes",
        ({"asset_class": cls, "description": d} for cls, items in modes.items() for d in items),
    )
    counts["work_orders"] = _insert(
        db,
        "work_orders",
        (
            {
                "wo_id": r["wo_id"],
                "equipment_id": r["equipment_id"],
                "description": r["wo_description"],
                "component": r["collection"],
                "primary_code": r["primary_code"] or None,
                "secondary_code": r["secondary_code"] or None,
                "work_type": "PM" if r["preventive"].upper() == "TRUE" else "CM",
                "priority": int(r["work_priority"]),
                "status": "COMP",  # the history holds completed work only
                "reported_at": None,
                "finished_at": us_datetime(r["actual_finish"]),
                "duration_h": hours(r["duration"]),
                "labor_h": hours(r["actual_labor_hours"]),
            }
            for r in work_orders
        ),
    )
    counts["events"] = _insert(
        db,
        "events",
        (
            {
                "event_id": r["event_id"],
                "equipment_id": r["equipment_id"],
                "event_group": r["event_group"],
                "event_category": r["event_category"],
                "event_type": r["event_type"],
                "description": r["description"],
                "event_time": iso_datetime(r["event_time"]),
                "note": r["note"],
            }
            for r in events
        ),
    )
    counts["alert_rules"] = _insert(
        db,
        "alert_rules",
        ({"rule_id": r["rule_id"], "name": r["rule_name"]} for r in _rows(raw["alert_rule.csv"])),
    )
    mapping = {
        (r["rule_id"], r["primary_code"]) for r in _rows(raw["alert_rule_failure_code_mapping.csv"])
    }
    counts["alert_rule_failure_codes"] = _insert(
        db,
        "alert_rule_failure_codes",
        ({"rule_id": rule, "primary_code": code} for rule, code in sorted(mapping)),
    )
    counts["alerts"] = _insert(
        db,
        "alerts",
        (
            {
                "equipment_id": r["equipment_id"],
                "rule_id": r["rule_id"],
                "start_time": us_datetime(r["start_time"]),
                "end_time": us_datetime(r["end_time"]),
            }
            for r in alerts
        ),
    )
    counts["anomaly_failure_codes"] = _insert(
        db,
        "anomaly_failure_codes",
        (
            {
                "kpi": r["kpi_name"],
                "anomaly_type": r["anomaly_type"],
                "category": r["category"],
                "primary_code": r["primary_code"],
                "secondary_code": r["secondary_code"],
            }
            for r in _rows(raw["anomaly_to_failure_code_mapping.csv"])
        ),
    )
    counts.update(_load_telemetry(db, raw["chiller6_june2020_sensordata_couchdb.json"], equipment))
    return counts


def _load_telemetry(
    db: sqlite3.Connection, path: Path, equipment: dict[str, str]
) -> dict[str, int]:
    by_name = {name: eq_id for eq_id, name in equipment.items()}
    records = json.loads(path.read_text(encoding="utf-8"))
    sensors: dict[str, dict[str, str]] = {}
    points = []
    for rec in records:
        name = rec["asset_id"]
        eq_id = by_name[name]
        ts = iso_datetime(rec["timestamp"])
        for key, value in rec.items():
            if key in ("asset_id", "timestamp", "_id", "_rev") or value is None:
                continue
            sensor_name = key.removeprefix(f"{name} ").strip()
            sensor_id = f"{eq_id}.{_slug(sensor_name)}"
            sensors[sensor_id] = {
                "sensor_id": sensor_id,
                "equipment_id": eq_id,
                "name": sensor_name,
            }
            points.append({"sensor_id": sensor_id, "ts": ts, "value": float(value)})
    return {
        "sensors": _insert(db, "sensors", sorted(sensors.values(), key=lambda s: s["sensor_id"])),
        "telemetry": _insert(db, "telemetry", points),
    }
