"""Command-line entry point: `shopfloor data | ...`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from shopfloor_agent.config import get_settings

app = typer.Typer(add_completion=False, help="On-prem maintenance agent over MCP.")


@app.callback()
def _setup() -> None:
    """On-prem maintenance agent over MCP."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


DataDirOpt = Annotated[Path | None, typer.Option(help="Defaults to SHOPFLOOR_DATA_DIR")]


@app.command("data")
def data(data_dir: DataDirOpt = None) -> None:
    """Downloads the pinned plant data (~3.5 MB) and builds data/plant.db."""
    from shopfloor_agent.data.build import build
    from shopfloor_agent.data.sources import fetch_all

    settings = get_settings()
    base = data_dir or settings.data_dir
    counts = build(fetch_all(base / "raw"), base / "plant.db")
    typer.echo(json.dumps(counts, indent=2))


SUITE = Path("tasks/suite.jsonl")


@app.command("tasks")
def tasks_cmd(out: Annotated[Path, typer.Option()] = SUITE, seed: int = 7) -> None:
    """Generates the task suite from the plant database (answers computed by SQL)."""
    from collections import Counter

    from shopfloor_agent.eval.tasks import generate, save

    tasks = generate(get_settings().plant_db, seed=seed)
    save(tasks, out)
    typer.echo(f"{len(tasks)} tasks -> {out}")
    typer.echo(
        json.dumps(
            {"tier": Counter(t.tier for t in tasks), "split": Counter(t.split for t in tasks)},
            indent=2,
        )
    )


@app.command("eval")
def eval_cmd(
    name: Annotated[str, typer.Option(help="Results go to results/<name>.jsonl")],
    agent: Annotated[str, typer.Option(help="react | plan_execute | react_verify | oracle")] = (
        "react"
    ),
    split: Annotated[str, typer.Option(help="dev | test | all")] = "dev",
    tier: Annotated[str | None, typer.Option(help="Only this tier")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N tasks")] = None,
    shard: Annotated[str | None, typer.Option(help="INDEX/COUNT of the tasks")] = None,
    max_steps: Annotated[int, typer.Option(help="Model calls per task")] = 10,
    read_only: Annotated[bool, typer.Option(help="Leave the write tools out")] = False,
    suite: Annotated[Path, typer.Option()] = SUITE,
) -> None:
    """Runs the agent on the task suite as isolated episodes; resumes an interrupted run."""
    import anyio

    from shopfloor_agent.eval.oracle import oracle_for
    from shopfloor_agent.eval.runner import make_agent, run_suite
    from shopfloor_agent.eval.tasks import load

    settings = get_settings()
    tasks = [t for t in load(suite) if split in ("all", t.split)]
    if tier:
        tasks = [t for t in tasks if t.tier == tier]
    if shard:
        index, count = (int(v) for v in shard.split("/"))
        tasks = tasks[index::count]
        name = f"{name}.shard-{index}-of-{count}"
    tasks = tasks[:limit] if limit else tasks
    out = settings.results_dir / f"{name}.jsonl"
    meta = {
        "agent": agent,
        "model": settings.llm_model,
        "max_steps": max_steps,
        "read_only": read_only,
    }

    def report(row: dict[str, object]) -> None:
        mark = "PASS" if row["passed"] else "FAIL"
        typer.echo(
            f"{row['task']} {mark} {row['seconds']:>6.0f}s {row['tool_calls']} calls  "
            f"{row['reason'] or ''}"
        )

    typer.echo(f"{len(tasks)} tasks -> {out}")
    rows = anyio.run(
        lambda: run_suite(
            tasks,
            settings.plant_db,
            oracle_for(tasks)
            if agent == "oracle"
            else make_agent(settings, agent, max_steps=max_steps),
            out,
            read_only=read_only,
            meta=meta,
            on_result=report,
        )
    )
    passed = sum(r["passed"] for r in rows if r["task"] in {t.id for t in tasks})
    typer.echo(f"passed {passed}/{len(tasks)}")


@app.command("model-url")
def model_url(name: str) -> None:
    """Prints the download URL of a benchmark model (used by the CI workflows)."""
    from shopfloor_agent.models import MODELS

    if name not in MODELS:
        raise typer.BadParameter(f"unknown model '{name}' ({', '.join(MODELS)})")
    typer.echo(MODELS[name].url)


@app.command("report")
def report_cmd(
    run: Annotated[list[str], typer.Option(help="Run names (results/<name>[.shard-*].jsonl)")],
    split: Annotated[str, typer.Option(help="dev | test | all")] = "test",
    out: Annotated[Path, typer.Option(help="Writes <out>.json and <out>.md")] = Path(
        "results/report"
    ),
) -> None:
    """Pass rates per tier with 95 % intervals, tool use and cost, one row per run."""
    from shopfloor_agent.eval.report import load_run, markdown, summarize

    settings = get_settings()
    summaries = {}
    for name in run:
        files = [*settings.results_dir.glob(f"{name}.jsonl"),
                 *settings.results_dir.glob(f"{name}.shard-*.jsonl")]  # fmt: skip
        rows = load_run(files)
        rows = [r for r in rows if split in ("all", r["split"])]
        if not rows:
            raise typer.BadParameter(f"no {split} results for run '{name}'")
        summaries[name] = summarize(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    table = markdown(summaries)
    out.with_suffix(".md").write_text(table, encoding="utf-8")
    typer.echo(table)


@app.command("ask")
def ask(
    question: str,
    max_steps: Annotated[int, typer.Option(help="Model calls before giving up")] = 10,
    read_only: Annotated[bool, typer.Option(help="Leave the write tools out")] = False,
    agent: Annotated[str, typer.Option(help="react | plan_execute | react_verify")] = "react",
) -> None:
    """Answers one question and prints the tool calls the agent made."""
    import anyio

    from shopfloor_agent.agent.graphs import DESIGNS, make_llm
    from shopfloor_agent.agent.toolkit import Toolkit, plant_servers

    settings = get_settings()

    async def main() -> None:
        async with Toolkit(plant_servers(settings.plant_db, read_only=read_only)) as kit:
            run = await DESIGNS[agent](question, kit, make_llm(settings), max_steps=max_steps)
            if run.notes:
                typer.echo(json.dumps(run.notes, indent=1))
            for c in kit.calls:
                outcome = f"ERROR {c.error}" if c.error else str(c.result)[:160]
                typer.echo(f"  {c.name}({json.dumps(c.arguments)}) -> {outcome}")
            typer.echo(
                f"[{run.stopped}, {run.steps} steps, {run.seconds:.0f} s, "
                f"{run.tokens_in} in / {run.tokens_out} out tokens]"
            )
            typer.echo(run.answer)

    anyio.run(main)


if __name__ == "__main__":
    app()
