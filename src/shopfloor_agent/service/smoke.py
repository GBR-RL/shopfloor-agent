"""End-to-end check of a running deployment: ask a question through the HTTP stream, require a
tool call and an answer, read the metrics, and (optionally) find the traces in Jaeger.

    python -m shopfloor_agent.service.smoke http://localhost:8000 --jaeger http://localhost:16686
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

QUESTION = "How many corrective work orders did Chiller 9 have in 2017?"


def ask(base: str, question: str, timeout: float = 900) -> list[dict[str, object]]:
    events = []
    with httpx.stream("POST", f"{base}/ask", json={"question": question, "writes": "off"},
                      timeout=timeout) as response:  # fmt: skip
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                print(line[6:][:200], flush=True)
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", nargs="?", default="http://localhost:8000")
    parser.add_argument("--jaeger", default=None)
    args = parser.parse_args()

    health = httpx.get(f"{args.base}/health", timeout=10).json()
    assert health["llm_reachable"], f"model server not reachable: {health}"
    events = ask(args.base, QUESTION)
    kinds = [e["type"] for e in events]
    assert "tool_start" in kinds, f"the agent called no tool: {kinds}"
    assert kinds[-1] == "answer", f"no answer: {events[-1]}"
    metrics = httpx.get(f"{args.base}/metrics", timeout=10).text
    assert "shopfloor_tool_calls_total" in metrics
    assert "shopfloor_llm_tokens_total" in metrics
    if args.jaeger:
        # Jaeger 2.x query API v3; spans are exported in batches, so allow a few seconds
        services: list[str] = []
        for _ in range(30):
            services = httpx.get(f"{args.jaeger}/api/v3/services", timeout=10).json()["services"]
            if "shopfloor-agent" in services:
                break
            time.sleep(2)
        else:
            raise AssertionError(f"no traces in Jaeger (services: {services})")
        ops = httpx.get(
            f"{args.jaeger}/api/v3/operations", params={"service": "shopfloor-agent"}, timeout=10
        ).json()["operations"]
        names = {o["name"] for o in ops}
        assert "invoke_agent" in names, names
        assert any(n.startswith("execute_tool ") for n in names), names
        assert any(n.startswith("chat ") for n in names), names
        print(f"jaeger: operations {sorted(names)}")
    print("smoke test passed:", events[-1].get("answer"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
