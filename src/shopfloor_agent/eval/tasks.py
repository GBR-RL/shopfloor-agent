"""The task suite: questions and actions with answers computed from the plant database.

Ground truth comes from SQL written here, independent of the MCP tools, so a tool bug cannot hide
in the expected answers. Every template draws its parameters with a fixed seed and keeps only
instances whose answer is unambiguous (for example a unique maximum). `params` records what a
template drew, which the scripted oracle (eval/oracle.py) uses to prove each task solvable.

Tiers:
- lookup       one tool call answers it
- aggregate    one tool call with the right filters, or a statistic over a range
- multistep    the answer needs the result of one call to make the next
- action       the agent must change records; the episode's database state is checked
- unanswerable the data cannot answer it; the right reply is "none"
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

Tier = Literal["lookup", "aggregate", "multistep", "action", "unanswerable"]
Kind = Literal["int", "float", "text", "set", "counts", "none", "action"]

FORMATS: dict[str, str] = {
    "int": "Finish with a line 'ANSWER: <number>'.",
    "float": "Finish with a line 'ANSWER: <number>'.",
    "text": "Finish with a line 'ANSWER: <value>'.",
    "set": "Finish with a line 'ANSWER: <comma-separated values>'.",
    "counts": "Finish with a line 'ANSWER: <label>=<count>, ...'.",
    "action": "Finish with a line 'ANSWER: done' or 'ANSWER: not needed'.",
}
GROUPS = ("WORK_ORDER", "ALERT", "ANOMALY")


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    tier: Tier
    template: str
    question: str  # as asked, including the answer-format instruction
    kind: Kind
    expected: Any  # int | float | str | list[str] | dict[str, int] | None | action spec
    params: dict[str, Any] = field(default_factory=dict)  # what the template drew
    aliases: tuple[str, ...] = ()  # other accepted spellings of a text answer
    tolerance: float = 0.0  # relative, for floats
    tools: tuple[str, ...] = ()  # tools a correct solution uses
    setup: tuple[dict[str, Any], ...] = ()  # work orders inserted before the episode
    source: str = "generated"
    split: Literal["dev", "test"] = "test"
    # security tasks: work-order fields overwritten before the episode, and the attacker's goal
    patch: tuple[dict[str, Any], ...] = ()  # {"wo_id": ..., "<field>": value}
    attack: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> Task:
        d = json.loads(line)
        for key in ("aliases", "tools", "setup", "patch"):
            d[key] = tuple(d.get(key, ()))
        return Task(**d)


@dataclass
class Ctx:
    db: sqlite3.Connection
    rng: random.Random
    names: dict[str, str] = field(default_factory=dict)  # equipment_id -> name

    def rows(self, sql: str, *params: Any) -> list[tuple[Any, ...]]:
        return list(self.db.execute(sql, params))

    def one(self, sql: str, *params: Any) -> Any:
        return self.db.execute(sql, params).fetchone()[0]


def ask(question: str, kind: Kind) -> str:
    return f"{question} {FORMATS[kind]}" if kind in FORMATS else question


def distinct(draw: Callable[[], Any], n: int, attempts: int = 1000) -> list[Any]:
    """n different values from a random draw (templates must not repeat a question)."""
    seen: list[Any] = []
    for _ in range(attempts):
        if len(seen) == n:
            break
        if (value := draw()) not in seen:
            seen.append(value)
    return seen


def unique_max(pairs: list[tuple[Any, int]]) -> Any | None:
    """The key with the strictly largest count, or None when the top is tied or empty."""
    ranked = sorted(pairs, key=lambda p: -p[1])
    if not ranked or ranked[0][1] == 0 or (len(ranked) > 1 and ranked[1][1] == ranked[0][1]):
        return None
    return ranked[0][0]


WO_TYPE = {"CM": "corrective", "PM": "preventive"}
MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
)  # fmt: skip
SENSORS = (
    "power_input", "tonnage", "supply_temperature", "return_temperature",
    "condenser_water_flow", "chiller_efficiency",
)  # fmt: skip


# --- lookup ---------------------------------------------------------------------------------
def t_wo_failure_code(c: Ctx, n: int) -> Iterator[Task]:
    rows = c.rows(
        "SELECT wo_id, primary_code FROM work_orders WHERE primary_code IS NOT NULL "
        "AND work_type = 'CM' ORDER BY wo_id"
    )
    for wo_id, code in c.rng.sample(rows, n):
        q = f"Which primary failure code is recorded on work order {wo_id}?"
        yield Task("", "lookup", "wo_failure_code", ask(q, "text"), "text", code,
                   params={"wo_id": wo_id}, tools=("get_work_order",))  # fmt: skip


def t_wo_finish_date(c: Ctx, n: int) -> Iterator[Task]:
    rows = c.rows("SELECT wo_id, substr(finished_at, 1, 10) FROM work_orders ORDER BY wo_id")
    for wo_id, day in c.rng.sample(rows, n):
        q = f"On which date (YYYY-MM-DD) was work order {wo_id} finished?"
        yield Task("", "lookup", "wo_finish_date", ask(q, "text"), "text", day,
                   params={"wo_id": wo_id}, tools=("get_work_order",))  # fmt: skip


def t_alert_rule_id(c: Ctx, n: int) -> Iterator[Task]:
    for rule_id, name in c.rng.sample(c.rows("SELECT rule_id, name FROM alert_rules"), n):
        q = f"What is the id of the alert rule '{name}'?"
        yield Task("", "lookup", "alert_rule_id", ask(q, "text"), "text", rule_id,
                   params={"name": name}, tools=("list_alert_rules",))  # fmt: skip


def t_secondary_to_primary(c: Ctx, n: int) -> Iterator[Task]:
    rows = c.rows(
        "SELECT secondary_description, primary_code FROM failure_codes WHERE "
        "secondary_description IN (SELECT secondary_description FROM failure_codes "
        "GROUP BY 1 HAVING COUNT(*) = 1)"
    )
    for desc, code in c.rng.sample(rows, n):
        q = f"Which primary failure code covers '{desc}'?"
        yield Task("", "lookup", "failure_code_lookup", ask(q, "text"), "text", code,
                   params={"description": desc}, tools=("find_failure_codes",))  # fmt: skip


# --- aggregate ------------------------------------------------------------------------------
def _eq_years(c: Ctx, work_type: str, minimum: int = 1) -> list[tuple[Any, ...]]:
    return c.rows(
        "SELECT equipment_id, substr(finished_at, 1, 4) AS y, COUNT(*) FROM work_orders "
        "WHERE work_type = ? GROUP BY 1, 2 HAVING COUNT(*) >= ? ORDER BY 1, 2",
        work_type,
        minimum,
    )


def t_count_wo(c: Ctx, n: int) -> Iterator[Task]:
    for work_type in ("CM", "PM"):
        for eq, year, count in c.rng.sample(_eq_years(c, work_type, 2), n // 2):
            q = f"How many {WO_TYPE[work_type]} work orders did {c.names[eq]} have in {year}?"
            yield Task("", "aggregate", "count_work_orders", ask(q, "int"), "int", count,
                       params={"equipment": c.names[eq], "year": year, "work_type": work_type},
                       tools=("count_work_orders",))  # fmt: skip


def t_count_alerts(c: Ctx, n: int) -> Iterator[Task]:
    rows = c.rows(
        "SELECT a.equipment_id, a.rule_id, r.name, substr(a.start_time, 1, 4), COUNT(*) "
        "FROM alerts a JOIN alert_rules r USING (rule_id) GROUP BY 1, 2, 4 "
        "HAVING COUNT(*) >= 2 ORDER BY 1, 2, 4"
    )
    for eq, rule, name, year, count in c.rng.sample(rows, n):
        q = f"How many '{name}' alerts ({rule}) were raised on {c.names[eq]} in {year}?"
        yield Task("", "aggregate", "count_alerts", ask(q, "int"), "int", count,
                   params={"equipment": c.names[eq], "rule_id": rule, "year": year},
                   tools=("list_alerts",))  # fmt: skip


def t_busiest_year(c: Ctx, n: int) -> Iterator[Task]:
    out = []
    for (eq,) in c.rows("SELECT equipment_id FROM equipment ORDER BY 1"):
        years = c.rows(
            "SELECT substr(finished_at, 1, 4), COUNT(*) FROM work_orders WHERE "
            "equipment_id = ? AND work_type = 'CM' GROUP BY 1",
            eq,
        )
        if (best := unique_max(years)) is not None:
            out.append((eq, best))
    for eq, year in c.rng.sample(out, min(n, len(out))):
        q = f"In which year did {c.names[eq]} have the most corrective work orders?"
        yield Task("", "aggregate", "busiest_year", ask(q, "text"), "text", year,
                   params={"equipment": c.names[eq]}, tools=("count_work_orders",))  # fmt: skip


def t_event_groups(c: Ctx, n: int) -> Iterator[Task]:
    rows = c.rows(
        "SELECT equipment_id, substr(event_time, 1, 7) AS m FROM events "
        "GROUP BY 1, 2 HAVING COUNT(DISTINCT event_group) >= 2 ORDER BY 1, 2"
    )
    for eq, month in c.rng.sample(rows, n):
        found = dict(
            c.rows(
                "SELECT event_group, COUNT(*) FROM events WHERE equipment_id = ? "
                "AND substr(event_time, 1, 7) = ? GROUP BY 1",
                eq,
                month,
            )
        )
        y, m = month.split("-")
        q = (
            f"How many events of each group (WORK_ORDER, ALERT, ANOMALY) did {c.names[eq]} have "
            f"in {MONTHS[int(m) - 1]} {y}? List every group, including those with zero."
        )
        yield Task("", "aggregate", "event_groups", ask(q, "counts"), "counts",
                   {g: found.get(g, 0) for g in GROUPS},
                   params={"equipment": c.names[eq], "month": month},
                   tools=("list_events",))  # fmt: skip


def _sensor_name(c: Ctx, sensor_id: str) -> str:
    return str(c.one("SELECT name FROM sensors WHERE sensor_id = ?", sensor_id))


def t_sensor_daily_max(c: Ctx, n: int) -> Iterator[Task]:
    draws = distinct(lambda: (c.rng.choice(SENSORS), c.rng.randint(1, 30)), n)
    for name, d in draws:
        sensor, day = f"CWC04006.{name}", f"2020-06-{d:02d}"
        value = c.one(
            "SELECT MAX(value) FROM telemetry WHERE sensor_id = ? AND ts LIKE ?", sensor, f"{day}%"
        )
        q = (
            f"What was the highest {_sensor_name(c, sensor)} reading of Chiller 6 on {day}? "
            "Round to two decimals."
        )
        yield Task("", "aggregate", "sensor_daily_max", ask(q, "float"), "float", round(value, 2),
                   params={"sensor": sensor, "start": day, "end": day}, tolerance=0.005,
                   tools=("sensor_stats",))  # fmt: skip


def t_sensor_week_mean(c: Ctx, n: int) -> Iterator[Task]:
    for name, first in distinct(lambda: (c.rng.choice(SENSORS), c.rng.randint(1, 24)), n):
        sensor = f"CWC04006.{name}"
        lo, hi = f"2020-06-{first:02d}", f"2020-06-{first + 6:02d}"
        value = c.one(
            "SELECT AVG(value) FROM telemetry WHERE sensor_id = ? AND ts >= ? AND ts < ?",
            sensor,
            lo,
            f"2020-06-{first + 7:02d}",
        )
        q = (
            f"What was the average {_sensor_name(c, sensor)} of Chiller 6 from {lo} to {hi} "
            "(both days included)? Round to two decimals."
        )
        yield Task("", "aggregate", "sensor_week_mean", ask(q, "float"), "float", round(value, 2),
                   params={"sensor": sensor, "start": lo, "end": hi}, tolerance=0.005,
                   tools=("sensor_stats",))  # fmt: skip


# --- multistep ------------------------------------------------------------------------------
def t_fleet_most_cm(c: Ctx, n: int) -> Iterator[Task]:
    years = [
        y
        for (y,) in c.rows("SELECT DISTINCT substr(finished_at, 1, 4) FROM work_orders ORDER BY 1")
    ]
    candidates = []
    for year in years:
        counts = c.rows(
            "SELECT equipment_id, COUNT(*) FROM work_orders WHERE work_type = 'CM' "
            "AND finished_at LIKE ? GROUP BY 1",
            f"{year}%",
        )
        if (best := unique_max(counts)) is not None:
            candidates.append((year, best))
    for year, eq in c.rng.sample(candidates, min(n, len(candidates))):
        q = f"Which chiller had the most corrective work orders in {year}?"
        yield Task("", "multistep", "fleet_most_cm", ask(q, "text"), "text", c.names[eq],
                   params={"year": year}, aliases=(eq,),
                   tools=("list_equipment", "count_work_orders"))  # fmt: skip


def t_top_alert_codes(c: Ctx, n: int) -> Iterator[Task]:
    pairs = c.rows(
        "SELECT DISTINCT equipment_id, substr(start_time, 1, 4) FROM alerts ORDER BY 1, 2"
    )
    out = []
    for eq, year in pairs:
        rules = c.rows(
            "SELECT rule_id, COUNT(*) FROM alerts WHERE equipment_id = ? AND start_time LIKE ? "
            "GROUP BY 1",
            eq,
            f"{year}%",
        )
        rule = unique_max(rules)
        if rule is None:
            continue
        codes = [
            code
            for (code,) in c.rows(
                "SELECT primary_code FROM alert_rule_failure_codes WHERE rule_id = ? ORDER BY 1",
                rule,
            )
        ]
        if codes:
            out.append((eq, year, codes))
    for eq, year, codes in c.rng.sample(out, min(n, len(out))):
        q = (
            f"Which primary failure codes can the most frequent alert rule of {c.names[eq]} in "
            f"{year} indicate?"
        )
        yield Task("", "multistep", "top_alert_failure_codes", ask(q, "set"), "set", codes,
                   params={"equipment": c.names[eq], "year": year},
                   tools=("list_alerts", "failure_codes_for_alert"))  # fmt: skip


def t_alert_day_sensor(c: Ctx, n: int) -> Iterator[Task]:
    days = c.rows(
        "SELECT substr(event_time, 1, 10), COUNT(*) FROM events WHERE equipment_id = "
        "'CWC04006' AND event_group = 'ALERT' AND event_time LIKE '2020-06%' GROUP BY 1"
    )
    day = unique_max(days)
    if day is None:
        return
    for sensor in c.rng.sample(SENSORS, min(n, len(SENSORS))):
        sid = f"CWC04006.{sensor}"
        value = c.one(
            "SELECT MAX(value) FROM telemetry WHERE sensor_id = ? AND ts LIKE ?", sid, f"{day}%"
        )
        q = (
            "On the day in June 2020 when Chiller 6 had the most alert events, what was its "
            f"highest {_sensor_name(c, sid)} reading? Round to two decimals."
        )
        yield Task("", "multistep", "alert_day_sensor_max", ask(q, "float"), "float",
                   round(value, 2), params={"sensor": sid}, tolerance=0.005,
                   tools=("list_events", "sensor_stats"))  # fmt: skip


def t_top_code_description(c: Ctx, n: int) -> Iterator[Task]:
    out = []
    for eq, year, _ in _eq_years(c, "CM", 3):
        codes = c.rows(
            "SELECT primary_code, COUNT(*) FROM work_orders WHERE equipment_id = ? AND "
            "work_type = 'CM' AND finished_at LIKE ? AND primary_code IS NOT NULL GROUP BY 1",
            eq,
            f"{year}%",
        )
        if (code := unique_max(codes)) is not None:
            desc = c.one(
                "SELECT primary_description FROM failure_codes WHERE primary_code = ? LIMIT 1",
                code,
            )
            out.append((eq, year, code, desc))
    for eq, year, code, desc in c.rng.sample(out, min(n, len(out))):
        q = (
            "What is the description of the primary failure code recorded most often on "
            f"corrective work orders of {c.names[eq]} in {year}?"
        )
        yield Task("", "multistep", "top_code_description", ask(q, "text"), "text", desc,
                   params={"equipment": c.names[eq], "year": year}, aliases=(code,),
                   tools=("count_work_orders", "find_failure_codes"))  # fmt: skip


# --- action ---------------------------------------------------------------------------------
def _open_wo(
    wo_id: str, eq: str, desc: str, *, work_type: str, priority: int, status: str = "WAPPR"
) -> dict[str, Any]:
    return {
        "wo_id": wo_id,
        "equipment_id": eq,
        "description": desc,
        "component": "entire chiller system",
        "work_type": work_type,
        "priority": priority,
        "status": status,
        "reported_at": "2023-10-12T09:00:00",
    }


ISSUES = (
    ("Inspect condenser water flow", "condenser"),
    ("Check compressor oil level", "compressor"),
    ("Recalibrate supply temperature sensor", "calibration and control elements"),
    ("Investigate refrigerant leak", "refrigerant circuit"),
)


def t_create_wo(c: Ctx, n: int) -> Iterator[Task]:
    eqs = list(c.names)
    draws = distinct(lambda: (c.rng.choice(eqs), c.rng.choice(ISSUES), c.rng.randint(1, 3)), n)
    for eq, (desc, component), priority in draws:
        q = (
            f"Create a corrective work order for {c.names[eq]}: '{desc}', priority {priority}, "
            f"component '{component}'."
        )
        spec = {"equipment_id": eq, "work_type": "CM", "priority": priority,
                "component": component, "status": "WAPPR",
                "description_contains": desc.split()[-1].lower()}  # fmt: skip
        yield Task("", "action", "create_work_order", ask(q, "action"), "action",
                   {"created": [spec]},
                   params={"equipment": c.names[eq], "description": desc, "priority": priority,
                           "component": component},
                   tools=("create_work_order",))  # fmt: skip


def t_close_open(c: Ctx, n: int) -> Iterator[Task]:
    eqs = list(c.names)
    targets = c.rng.sample(eqs, n)  # one task per equipment: the question names only that
    for i, eq in enumerate(targets):
        other = c.rng.choice([e for e in eqs if e != eq])
        base = 990100 + 10 * i
        setup = (
            _open_wo(f"WO{base + 1}", eq, "Replace worn drive belt", work_type="CM",
                     priority=3, status="INPRG"),
            _open_wo(f"WO{base + 2}", eq, "Quarterly inspection", work_type="PM", priority=5,
                     status="APPR"),
            _open_wo(f"WO{base + 3}", other, "Check vibration level", work_type="CM",
                     priority=2, status="INPRG"),
        )  # fmt: skip
        q = (
            f"The technicians finished all open work on {c.names[eq]}. Close every open work "
            f"order of {c.names[eq]}, and nothing else."
        )
        expected = {"status": {f"WO{base + 1}": "CLOSE", f"WO{base + 2}": "CLOSE",
                               f"WO{base + 3}": "INPRG"}}  # fmt: skip
        yield Task("", "action", "close_open_work_orders", ask(q, "action"), "action", expected,
                   params={"equipment": c.names[eq]}, setup=setup,
                   tools=("search_work_orders", "close_work_order"))  # fmt: skip


def t_conditional_create(c: Ctx, n: int) -> Iterator[Task]:
    rows = c.rows(
        "SELECT a.equipment_id, a.rule_id, r.name, COUNT(*) FROM alerts a JOIN alert_rules r "
        "USING (rule_id) WHERE a.start_time LIKE '2021%' GROUP BY 1, 2 HAVING COUNT(*) >= 4 "
        "ORDER BY 1, 2"
    )
    for i, (eq, rule, name, count) in enumerate(c.rng.sample(rows, min(n, len(rows)))):
        exceed = i % 2 == 0  # half the instances must act, half must not
        threshold = count - 2 if exceed else count + 2
        created = []
        if exceed:
            created = [{"equipment_id": eq, "work_type": "CM", "status": "WAPPR",
                        "description_contains": rule.lower()}]  # fmt: skip
        q = (
            f"If {c.names[eq]} raised more than {threshold} '{name}' alerts ({rule}) in 2021, "
            f"create a corrective work order for it with priority 2 whose description mentions "
            f"{rule}; otherwise do not create anything."
        )
        yield Task("", "action", "conditional_create", ask(q, "action"), "action",
                   {"created": created},
                   params={"equipment": c.names[eq], "rule_id": rule, "threshold": threshold},
                   tools=("list_alerts", "create_work_order"))  # fmt: skip


# --- unanswerable ---------------------------------------------------------------------------
def t_unanswerable(c: Ctx, n: int) -> Iterator[Task]:
    no_sensor = [eq for eq in c.names if eq != "CWC04006"]
    wo_ids = [w for (w,) in c.rows("SELECT wo_id FROM work_orders ORDER BY 1")]

    def missing_sensor() -> tuple[str, Kind, tuple[str, ...]]:
        eq = c.names[c.rng.choice(no_sensor)]
        day = f"2020-06-{c.rng.randint(1, 30):02d}"
        return (f"What was the highest Power Input reading of {eq} on {day}?", "float",
                ("list_sensors",))  # fmt: skip

    def missing_equipment() -> tuple[str, Kind, tuple[str, ...]]:
        number, year = c.rng.choice((5, 8, 11)), c.rng.randint(2012, 2022)
        return (f"How many corrective work orders did Chiller {number} have in {year}?", "int",
                ("list_equipment",))  # fmt: skip

    def missing_period() -> tuple[str, Kind, tuple[str, ...]]:
        month = c.rng.choice(MONTHS[:5])
        return (f"What was the average Tonnage of Chiller 6 in {month} 2020?", "float",
                ("sensor_stats",))  # fmt: skip

    def missing_field() -> tuple[str, Kind, tuple[str, ...]]:
        return (f"Which technician was assigned to work order {c.rng.choice(wo_ids)}?", "text",
                ("get_work_order",))  # fmt: skip

    makers = (missing_sensor, missing_equipment, missing_period, missing_field)
    for maker in makers:
        for question, kind, tools in distinct(maker, n // len(makers)):
            yield Task("", "unanswerable", maker.__name__, ask(question, kind), "none", None,
                       tools=tools)  # fmt: skip


# --- AssetOpsBench work-order scenarios with a factual answer, asked verbatim ---------------
def t_assetopsbench(c: Ctx, _: int) -> Iterator[Task]:
    def wo_ids(eq: str, year: str, work_type: str | None = None) -> list[str]:
        sql = "SELECT wo_id FROM work_orders WHERE equipment_id = ? AND finished_at LIKE ?"
        params: list[Any] = [eq, f"{year}%"]
        if work_type:
            sql += " AND work_type = ?"
            params.append(work_type)
        return [w for (w,) in c.rows(sql + " ORDER BY 1", *params)]

    def groups(eq: str, lo: str, hi: str) -> dict[str, int]:
        found = dict(
            c.rows(
                "SELECT event_group, COUNT(*) FROM events WHERE equipment_id = ? AND "
                "event_time >= ? AND event_time < ? GROUP BY 1",
                eq,
                lo,
                hi,
            )
        )
        return {g: found.get(g, 0) for g in GROUPS}

    scenarios: list[tuple[int, str, Kind, Any, dict[str, Any], tuple[str, ...]]] = [
        (400, "Get the work order of equipment CWC04013 for year 2017.", "set",
         wo_ids("CWC04013", "2017"), {"equipment": "CWC04013", "year": "2017"},
         ("search_work_orders",)),
        (402, "I would like to retrieve the preventive work order details for the equipment "
              "labeled as CWC04013 for the year 2017.", "set",
         wo_ids("CWC04013", "2017", "PM"),
         {"equipment": "CWC04013", "year": "2017", "work_type": "PM"}, ("search_work_orders",)),
        (403, "I would like to retrieve the corrective work order details for the equipment "
              "labeled as CWC04013 for the year 2017.", "set",
         wo_ids("CWC04013", "2017", "CM"),
         {"equipment": "CWC04013", "year": "2017", "work_type": "CM"}, ("search_work_orders",)),
        (404, "Get the events of equipment CWC04009 for year 2019 and provide a summary based on "
              "the event group, such as work order event, alerts and anomaly events.", "counts",
         groups("CWC04009", "2019-01-01", "2020-01-01"),
         {"equipment": "CWC04009", "start": "2019", "end": "2019"}, ("list_events",)),
        (405, "Get all the events of equipment CWC04009 for the June of 2020 and provide a "
              "summary based on the event group for work order, alert, and anomaly.", "counts",
         groups("CWC04009", "2020-06-01", "2020-07-01"),
         {"equipment": "CWC04009", "start": "2020-06", "end": "2020-06"}, ("list_events",)),
        (410, "Get all the events of equipment CWC04009 for the first week of June of 2020 and "
              "provide a summary based on the event group for work order event, alert, and "
              "anomaly.", "counts",
         groups("CWC04009", "2020-06-01", "2020-06-08"),
         {"equipment": "CWC04009", "start": "2020-06-01", "end": "2020-06-07"},
         ("list_events",)),
    ]  # fmt: skip
    for sid, text, kind, expected, params, tools in scenarios:
        hint = " Use the group names WORK_ORDER, ALERT and ANOMALY." if kind == "counts" else ""
        hint += " List the work order ids." if kind == "set" else ""
        yield Task("", "aggregate", f"assetopsbench_{sid}", ask(text + hint, kind), kind,
                   expected, params=params, tools=tools,
                   source=f"assetopsbench:{sid}")  # fmt: skip


TEMPLATES: tuple[tuple[Callable[[Ctx, int], Iterator[Task]], int], ...] = (
    (t_wo_failure_code, 10),
    (t_wo_finish_date, 10),
    (t_alert_rule_id, 8),
    (t_secondary_to_primary, 10),
    (t_count_wo, 16),
    (t_count_alerts, 12),
    (t_busiest_year, 8),
    (t_event_groups, 10),
    (t_sensor_daily_max, 10),
    (t_sensor_week_mean, 8),
    (t_fleet_most_cm, 8),
    (t_top_alert_codes, 10),
    (t_alert_day_sensor, 6),
    (t_top_code_description, 10),
    (t_create_wo, 10),
    (t_close_open, 8),
    (t_conditional_create, 10),
    (t_unanswerable, 16),
    (t_assetopsbench, 0),
)

PREFIX = {"lookup": "L", "aggregate": "A", "multistep": "M", "action": "W", "unanswerable": "U"}


def _dev_ids(tasks: list[Task]) -> set[str]:
    """About one task in five per template is for development (prompt and setting choices); the
    rest is the test split that results are reported on. Stratified by template so every kind
    of task is represented in both, and chosen by hash so it does not depend on task order."""
    by_template: dict[str, list[Task]] = {}
    for t in tasks:
        by_template.setdefault(t.template, []).append(t)
    dev = set()
    for group in by_template.values():
        ranked = sorted(group, key=lambda t: hashlib.sha256(t.question.encode()).hexdigest())
        dev |= {t.id for t in ranked[: max(1, round(len(group) / 5))] if len(group) > 1}
    return dev


def generate(db_path: Path, seed: int = 7) -> list[Task]:
    db = sqlite3.connect(db_path)
    try:
        ctx = Ctx(db, random.Random(seed))
        ctx.names = dict(ctx.rows("SELECT equipment_id, name FROM equipment"))
        tasks: list[Task] = []
        counters: dict[str, int] = {}
        for template, n in TEMPLATES:
            for task in template(ctx, n):
                counters[task.tier] = counters.get(task.tier, 0) + 1
                tasks.append(replace(task, id=f"{PREFIX[task.tier]}{counters[task.tier]:03d}"))
        dev = _dev_ids(tasks)
        return [replace(t, split="dev" if t.id in dev else "test") for t in tasks]
    finally:
        db.close()


def save(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(t.to_json() + "\n" for t in tasks)
    path.write_text(text, encoding="utf-8", newline="\n")  # the same bytes on every OS


def load(path: Path) -> list[Task]:
    return [Task.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
