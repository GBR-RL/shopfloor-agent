"""Reliability MCP server: failure codes, failure modes, and what alerts and anomalies point to."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from shopfloor_agent.servers.maintenance import READ
from shopfloor_agent.servers.store import PlantStore, ToolError


def create_server(db_path: Path) -> MCPServer:
    store = PlantStore(db_path)
    server = MCPServer(
        "reliability",
        instructions=(
            "Failure-code hierarchy (category > primary code > secondary code), known failure "
            "modes per asset class, and the failure codes linked to alert rules and KPI anomalies."
        ),
        log_level="WARNING",
    )

    @server.tool(annotations=READ)
    def find_failure_codes(
        query: str | None = None, primary_code: str | None = None, category: str | None = None
    ) -> dict[str, Any]:
        """Failure codes matching a text query (in the descriptions), a primary code such as
        'M006', or a category."""
        clauses, params = [], []
        if query:
            clauses.append(
                "(lower(primary_description) LIKE ? OR lower(secondary_description) LIKE ?)"
            )
            params += [f"%{query.strip().lower()}%"] * 2
        if primary_code:
            clauses.append("primary_code = ?")
            params.append(primary_code.strip().upper())
        if category:
            clauses.append("lower(category) LIKE ?")
            params.append(f"%{category.strip().lower()}%")
        if not clauses:
            cats = store.query(
                "SELECT category, COUNT(DISTINCT primary_code) AS primary_codes "
                "FROM failure_codes GROUP BY category ORDER BY category"
            )
            return {"hint": "give a query, primary_code or category", "categories": cats}
        rows = store.query(
            "SELECT category, primary_code, primary_description, secondary_code, "
            f"secondary_description FROM failure_codes WHERE {' AND '.join(clauses)} "
            "ORDER BY secondary_code LIMIT 40",
            params,
        )
        return {"count": len(rows), "failure_codes": rows}

    @server.tool(annotations=READ)
    def list_failure_modes(asset_class: str) -> dict[str, Any]:
        """Known failure modes of an asset class ('chiller' or 'ahu')."""
        rows = store.query(
            "SELECT description FROM failure_modes WHERE asset_class = ?",
            [asset_class.strip().lower()],
        )
        if not rows:
            classes = [
                r["asset_class"]
                for r in store.query("SELECT DISTINCT asset_class FROM failure_modes ORDER BY 1")
            ]
            raise ToolError(f"no failure modes for '{asset_class}'. Known: {', '.join(classes)}")
        return {"asset_class": asset_class, "failure_modes": [r["description"] for r in rows]}

    @server.tool(annotations=READ)
    def list_alert_rules() -> dict[str, Any]:
        """All alert rules (id and name)."""
        return {"alert_rules": store.query("SELECT rule_id, name FROM alert_rules ORDER BY 1")}

    @server.tool(annotations=READ)
    def failure_codes_for_alert(rule_id: str) -> dict[str, Any]:
        """The primary failure codes an alert rule can indicate."""
        rule = store.query("SELECT * FROM alert_rules WHERE rule_id = ?", [rule_id.strip().upper()])
        if not rule:
            raise ToolError(f"unknown alert rule '{rule_id}'; call list_alert_rules")
        rows = store.query(
            "SELECT DISTINCT m.primary_code, f.primary_description FROM alert_rule_failure_codes m "
            "JOIN failure_codes f USING (primary_code) WHERE m.rule_id = ? ORDER BY 1",
            [rule[0]["rule_id"]],
        )
        return {**rule[0], "failure_codes": rows}

    @server.tool(annotations=READ)
    def failure_codes_for_anomaly(kpi: str, anomaly_type: Literal["High", "Low"]) -> dict[str, Any]:
        """The failure codes linked to a KPI anomaly, e.g. kpi='Flow Efficiency', anomaly_type
        'High' or 'Low'."""
        rows = store.query(
            "SELECT kpi, anomaly_type, category, primary_code, secondary_code "
            "FROM anomaly_failure_codes WHERE lower(kpi) = ? AND anomaly_type = ? ORDER BY 4, 5",
            [kpi.strip().lower(), anomaly_type],
        )
        if not rows:
            kpis = [r["kpi"] for r in store.query("SELECT DISTINCT kpi FROM anomaly_failure_codes")]
            raise ToolError(f"no mapping for '{kpi}' / {anomaly_type}. KPIs: {', '.join(kpis)}")
        return {"count": len(rows), "failure_codes": rows}

    return server
