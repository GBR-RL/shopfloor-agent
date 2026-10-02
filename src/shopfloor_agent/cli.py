"""Command-line entry point: `shopfloor data | ...`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

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


@app.command("injection-tasks")
def injection_tasks(
    suite: Annotated[Path, typer.Option()] = SUITE,
    strength: Annotated[str, typer.Option(help="basic | strong")] = "basic",
    out: Annotated[
        Path | None, typer.Option(help="Default: tasks/injection[_strong].jsonl")
    ] = None,
) -> None:
    """Builds a prompt-injection suite: per work-order lookup a clean variant and the attacks
    (basic: three goals; strong: three goals in two styles, control tokens and task framing)."""
    from shopfloor_agent.eval.injection import build
    from shopfloor_agent.eval.tasks import load, save

    out = out or Path(
        "tasks/injection.jsonl" if strength == "basic" else "tasks/injection_strong.jsonl"
    )
    tasks = build(load(suite), strength)
    save(tasks, out)
    typer.echo(f"{len(tasks)} tasks -> {out}")


@app.command("eval")
def eval_cmd(
    name: Annotated[str, typer.Option(help="Results go to results/<name>.jsonl")],
    agent: Annotated[
        str, typer.Option(help="react | plan_execute | react_verify | routed | oracle")
    ] = ("react"),
    split: Annotated[str, typer.Option(help="dev | test | all")] = "dev",
    tier: Annotated[str | None, typer.Option(help="Only this tier")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N tasks")] = None,
    shard: Annotated[str | None, typer.Option(help="INDEX/COUNT of the tasks")] = None,
    max_steps: Annotated[int, typer.Option(help="Model calls per task")] = 16,
    read_only: Annotated[bool, typer.Option(help="Leave the write tools out")] = False,
    defense: Annotated[
        str, typer.Option(help="none | spotlight | sanitize | read_only | approval")
    ] = "none",
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
        "defense": defense,
    }

    def report(row: dict[str, object]) -> None:
        mark = "PASS" if row["passed"] else "FAIL"
        typer.echo(
            f"{row['task']} {mark} {row['seconds']:>6.0f}s {row['tool_calls']} calls  "
            f"{row['reason'] or ''}"
        )

    typer.echo(f"{len(tasks)} tasks -> {out}")
    agent_fn = (
        oracle_for(tasks)
        if agent == "oracle"
        else make_agent(settings, agent, max_steps=max_steps, defense=defense)
    )
    rows = anyio.run(
        lambda: run_suite(
            tasks,
            settings.plant_db,
            agent_fn,
            out,
            read_only=read_only,
            defense=defense,
            meta=meta,
            on_result=report,
        )
    )
    passed = sum(r["passed"] for r in rows if r["task"] in {t.id for t in tasks})
    typer.echo(f"passed {passed}/{len(tasks)}")


@app.command("judge")
def judge_cmd(
    run: Annotated[str, typer.Option(help="Benchmark run to grade (results/<run>[.shard-*])")],
    mode: Annotated[str, typer.Option(help="reference | rubric")] = "rubric",
    split: Annotated[str, typer.Option(help="dev | test | all")] = "test",
    limit: Annotated[int | None, typer.Option(help="Only the first N episodes")] = None,
    shard: Annotated[str | None, typer.Option(help="INDEX/COUNT of the episodes")] = None,
) -> None:
    """Grades a run's episodes with the served model as judge and compares the verdicts with
    the deterministic scores. The judge model is whatever SHOPFLOOR_LLM_* points at."""
    import anyio

    from shopfloor_agent.agent.graphs import make_llm
    from shopfloor_agent.eval.judge import agreement, judge, load_questions
    from shopfloor_agent.eval.report import load_run
    from shopfloor_agent.eval.runner import load_results

    if mode not in ("reference", "rubric"):
        raise typer.BadParameter("mode must be reference or rubric")
    settings = get_settings()
    files = [*settings.results_dir.glob(f"{run}.jsonl"),
             *settings.results_dir.glob(f"{run}.shard-*.jsonl")]  # fmt: skip
    rows = [r for r in load_run(files) if split in ("all", r["split"])][:limit]
    name = f"judge-{settings.llm_model}-{mode}-{run}"
    if shard:
        index, count = (int(v) for v in shard.split("/"))
        rows = rows[index::count]
        name = f"{name}.shard-{index}-of-{count}"
    questions = load_questions(
        [SUITE, Path("tasks/injection.jsonl"), Path("tasks/injection_strong.jsonl")]
    )
    out = settings.results_dir / f"{name}.jsonl"
    done = {r["task"]: r for r in load_results(out)}
    llm = make_llm(settings, max_tokens=700)

    async def main() -> list[dict[str, Any]]:
        graded = []
        for row in rows:
            if row["task"] in done:
                graded.append(done[row["task"]])
                continue
            verdict, usage = await judge(llm, row, questions[row["task"]], mode)  # type: ignore[arg-type]
            entry = {"task": row["task"], "tier": row["tier"], "passed": row["passed"],
                     "judge_passed": verdict.passed if verdict else None,
                     "judge_reason": verdict.reason if verdict else None, "mode": mode,
                     "judge_model": settings.llm_model, "agent_model": row.get("model"),
                     **usage}  # fmt: skip
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            graded.append(entry)
            typer.echo(f"{row['task']} truth={row['passed']} judge={entry['judge_passed']}")
        return graded

    graded = anyio.run(main)
    summary = {"overall": agreement(graded)}
    for tier in sorted({str(g["tier"]) for g in graded}):
        summary[tier] = agreement([g for g in graded if g["tier"] == tier])
    Path(f"{out}".removesuffix(".jsonl") + ".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    typer.echo(json.dumps(summary["overall"], indent=2))


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
    from shopfloor_agent.eval.report import load_run, markdown, security_markdown, summarize

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
    # names like "granite-4.2-3b" contain dots, so append extensions instead of with_suffix
    Path(f"{out}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    table = markdown(summaries) + "\n" + security_markdown(summaries)
    Path(f"{out}.md").write_text(table, encoding="utf-8")
    typer.echo(table)


@app.command("rescore")
def rescore_cmd(
    run: Annotated[list[str], typer.Option(help="Run names (results/<name>[.shard-*].jsonl)")],
    suite: Annotated[Path, typer.Option()] = SUITE,
) -> None:
    """Scores saved runs again against the current suite (after a fix to the scoring), in place,
    and prints how many verdicts changed."""
    from shopfloor_agent.eval.check import rescore
    from shopfloor_agent.eval.runner import load_results
    from shopfloor_agent.eval.tasks import load

    settings = get_settings()
    tasks = {t.id: t for t in load(suite)}
    for name in run:
        files = [*settings.results_dir.glob(f"{name}.jsonl"),
                 *settings.results_dir.glob(f"{name}.shard-*.jsonl")]  # fmt: skip
        flipped = []
        for path in files:
            rows = load_results(path)
            new = [rescore(tasks[r["task"]], r) for r in rows]
            flipped += [n["task"] for o, n in zip(rows, new, strict=True)
                        if o["passed"] != n["passed"]]  # fmt: skip
            text = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in new)
            path.write_text(text, encoding="utf-8", newline="\n")
        typer.echo(f"{name}: {len(set(flipped))} verdicts changed {sorted(set(flipped))}")


@app.command("chart")
def chart_cmd(
    kind: Annotated[str, typer.Argument(help="tiers | cost")],
    run: Annotated[list[str], typer.Option(help="NAME=Label of a run; repeatable, in order")],
    out: Annotated[Path, typer.Option(help="Writes <out>-light.png and <out>-dark.png")],
    title: str = "",
    subtitle: str = "",
    split: str = "test",
) -> None:
    """Charts for the README from run results (light and dark versions)."""
    from shopfloor_agent.eval.plots import cost_chart, tier_chart
    from shopfloor_agent.eval.report import load_run, summarize

    settings = get_settings()
    summaries = {}
    for spec in run:
        name, _, label = spec.partition("=")
        files = [*settings.results_dir.glob(f"{name}.jsonl"),
                 *settings.results_dir.glob(f"{name}.shard-*.jsonl")]  # fmt: skip
        rows = [r for r in load_run(files) if split in ("all", r["split"])]
        if not rows:
            raise typer.BadParameter(f"no {split} results for run '{name}'")
        summaries[label or name] = summarize(rows)
    if kind == "tiers":
        paths = tier_chart(summaries, out, title=title, subtitle=subtitle)
    elif kind == "cost":
        points = {k: (v["overall"]["median_seconds"], v["overall"]["pass_rate"])
                  for k, v in summaries.items()}  # fmt: skip
        paths = cost_chart(points, out, title=title, subtitle=subtitle)
    else:
        raise typer.BadParameter("kind must be tiers or cost")
    typer.echo("\n".join(str(p) for p in paths))


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Runs the HTTP service (event stream, write approvals, /metrics, demo page at /)."""
    import uvicorn

    from shopfloor_agent.service.app import create_app

    uvicorn.run(create_app(), host=host, port=port)


@app.command("ask")
def ask(
    question: str,
    max_steps: Annotated[int, typer.Option(help="Model calls before giving up")] = 16,
    read_only: Annotated[bool, typer.Option(help="Leave the write tools out")] = False,
    agent: Annotated[
        str, typer.Option(help="react | plan_execute | react_verify | routed")
    ] = "react",
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
