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


@app.command("ask")
def ask(
    question: str,
    max_steps: Annotated[int, typer.Option(help="Model calls before giving up")] = 10,
    read_only: Annotated[bool, typer.Option(help="Leave the write tools out")] = False,
) -> None:
    """Answers one question with the ReAct agent and prints the tool calls it made."""
    import anyio

    from shopfloor_agent.agent.react import make_llm, run_react
    from shopfloor_agent.agent.toolkit import Toolkit, plant_servers

    settings = get_settings()

    async def main() -> None:
        async with Toolkit(plant_servers(settings.plant_db, read_only=read_only)) as kit:
            run = await run_react(question, kit, make_llm(settings), max_steps=max_steps)
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
