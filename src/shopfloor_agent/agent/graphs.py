"""Agent designs as LangGraph graphs, all over the same toolkit and model.

- react         the model alternates between tool calls and reading results until it answers
- plan_execute  a planning call writes a short plan first (structured JSON output); the ReAct
                loop then carries it out with the plan in its context
- react_verify  ReAct, then a reviewer call checks whether the answer follows from the tool
                results; if not, the agent gets the critique and one more round
- routed        a router call classifies the request first: dependent multi-step work goes to
                plan_execute, everything else to react

Every design gets the same step budget (model calls), so they are compared at equal cost.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from shopfloor_agent.agent.toolkit import Toolkit
from shopfloor_agent.config import Settings

SYSTEM_PROMPT = """You are a maintenance assistant for a plant's chillers.
Answer only from tool results; never invent ids, counts or dates. Today is 2023-10-13.
Equipment is named like "Chiller 6" (id CWC04006). Work order types: PM = preventive,
CM = corrective. Keep the final answer short and give the exact numbers or ids asked for.
If the data cannot answer the question, say so and finish with 'ANSWER: none'."""


def make_llm(settings: Settings, **overrides: Any) -> ChatOpenAI:
    params: dict[str, Any] = {
        "base_url": settings.llm_base_url,
        "api_key": "local",
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "timeout": settings.llm_timeout_s,
        "max_retries": 1,
        # Reasoning traces cost minutes per step on a CPU; tool use works without them.
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    params.update(overrides)
    return ChatOpenAI(**params)


@dataclass
class Run:
    answer: str
    steps: int  # model calls, planner and reviewer included
    seconds: float
    tokens_in: int = 0
    tokens_out: int = 0
    stopped: str = "answered"  # answered | step_limit
    messages: list[AnyMessage] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)  # plan, review verdicts


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


def _usage(messages: Sequence[BaseMessage]) -> tuple[int, int]:
    usage = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
    return sum(u["input_tokens"] for u in usage), sum(u["output_tokens"] for u in usage)


def _final(messages: list[AnyMessage]) -> str:
    last = messages[-1]
    return last.text if isinstance(last, AIMessage) and not last.tool_calls else ""


def with_server_notes(system: str, kit: Toolkit) -> str:
    """The system prompt plus each MCP server's instructions (sent at session start)."""
    if not kit.instructions:
        return system
    notes = "\n".join(f"- {name}: {text}" for name, text in kit.instructions.items())
    return f"{system}\n\nTool servers:\n{notes}"


def react_graph(kit: Toolkit, llm: ChatOpenAI, max_steps: int) -> Any:
    model = llm.bind_tools(kit.tools)
    tools = {t.name: t for t in kit.tools}

    async def agent(state: State) -> dict[str, Any]:
        reply = await model.ainvoke(state["messages"])
        return {"messages": [reply], "steps": state["steps"] + 1}

    async def act(state: State) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        out = []
        for call in last.tool_calls:
            tool = tools.get(call["name"])
            content = (
                await tool.ainvoke(call["args"])
                if tool
                else f"ERROR: unknown tool '{call['name']}'"
            )
            out.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": out}

    def route(state: State) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state["steps"] < max_steps:
            return "act"
        return END

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("act", act)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"act": "act", END: END})
    graph.add_edge("act", "agent")
    return graph.compile()


async def _loop(app: Any, messages: list[AnyMessage], steps: int, max_steps: int) -> State:
    state: State = await app.ainvoke(
        {"messages": messages, "steps": steps}, {"recursion_limit": 2 * max_steps + 4}
    )
    return state


async def run_react(
    question: str,
    kit: Toolkit,
    llm: ChatOpenAI,
    *,
    max_steps: int = 16,
    system: str = SYSTEM_PROMPT,
) -> Run:
    start = time.perf_counter()
    app = react_graph(kit, llm, max_steps)
    state = await _loop(
        app, [SystemMessage(with_server_notes(system, kit)), HumanMessage(question)], 0, max_steps
    )
    answer = _final(state["messages"])
    tin, tout = _usage(state["messages"])
    return Run(answer, state["steps"], time.perf_counter() - start, tin, tout,
               "answered" if answer else "step_limit", state["messages"])  # fmt: skip


# --- plan and execute -------------------------------------------------------------------------
class Plan(BaseModel):
    steps: list[str] = Field(description="1 to 5 short steps, each naming the tool to call")


PLANNER_PROMPT = """Plan how to answer a maintenance question with these tools. Do not answer it.
Write 1 to 5 short steps; each step names one tool and what to pass to it.

Tools:
{tools}"""


async def run_plan_execute(
    question: str,
    kit: Toolkit,
    llm: ChatOpenAI,
    *,
    max_steps: int = 16,
    system: str = SYSTEM_PROMPT,
) -> Run:
    start = time.perf_counter()
    catalog = "\n".join(f"- {t.name}: {t.description}" for t in kit.tools)
    planner = llm.with_structured_output(Plan, method="json_schema", include_raw=True)
    out = await planner.ainvoke(
        [SystemMessage(PLANNER_PROMPT.format(tools=catalog)), HumanMessage(question)]
    )
    plan: Plan | None = out["parsed"]
    steps = plan.steps[:5] if plan else []
    planned = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) or "(no plan)"
    system = f"{system}\n\nFollow this plan, adapting it if a tool result requires:\n{planned}"
    app = react_graph(kit, llm, max_steps)
    state = await _loop(
        app, [SystemMessage(with_server_notes(system, kit)), HumanMessage(question)], 1, max_steps
    )
    answer = _final(state["messages"])
    tin, tout = _usage([out["raw"], *state["messages"]])
    return Run(answer, state["steps"], time.perf_counter() - start, tin, tout,
               "answered" if answer else "step_limit", state["messages"],
               {"plan": steps})  # fmt: skip


# --- ReAct with a reviewer --------------------------------------------------------------------
class Review(BaseModel):
    supported: bool = Field(description="true if the tool results support the final answer")
    problem: str = Field(description="what is missing or wrong; empty when supported")


REVIEW_PROMPT = """You review a maintenance assistant's work. Given the question, the tool calls
it made with their results, and its final answer: is the answer fully supported by the tool
results and does it answer exactly what was asked (right equipment, period, type, format)?
Reply supported=false with a one-sentence problem only if something is wrong or missing."""


def transcript(messages: list[AnyMessage], limit: int = 600) -> str:
    lines = []
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                lines.append(f"CALL {call['name']}({json.dumps(call['args'])})")
        elif isinstance(m, ToolMessage):
            text = str(m.content)
            lines.append(f"RESULT {text[:limit]}{' ...' if len(text) > limit else ''}")
    return "\n".join(lines) or "(no tool calls)"


async def run_react_verify(
    question: str,
    kit: Toolkit,
    llm: ChatOpenAI,
    *,
    max_steps: int = 16,
    system: str = SYSTEM_PROMPT,
) -> Run:
    start = time.perf_counter()
    app = react_graph(kit, llm, max_steps)
    state = await _loop(
        app, [SystemMessage(with_server_notes(system, kit)), HumanMessage(question)], 0, max_steps
    )
    reviewer = llm.with_structured_output(Review, method="json_schema", include_raw=True)
    extra: list[BaseMessage] = []
    verdicts = []
    answer = _final(state["messages"])
    if answer and state["steps"] < max_steps - 1:
        out = await reviewer.ainvoke([
            SystemMessage(REVIEW_PROMPT),
            HumanMessage(f"Question: {question}\n\nTool calls:\n{transcript(state['messages'])}"
                         f"\n\nFinal answer:\n{answer}"),
        ])  # fmt: skip
        extra.append(out["raw"])
        review: Review | None = out["parsed"]
        verdicts.append(review.model_dump() if review else None)
        if review is not None and not review.supported:
            feedback = HumanMessage(
                f"A reviewer found a problem: {review.problem} Check it with the tools, then give "
                "your final answer again, ending with the ANSWER line."
            )
            state = await _loop(app, [*state["messages"], feedback], state["steps"] + 1, max_steps)
            answer = _final(state["messages"])
        else:
            state["steps"] += 1
    tin, tout = _usage([*extra, *state["messages"]])
    return Run(answer, state["steps"], time.perf_counter() - start, tin, tout,
               "answered" if answer else "step_limit", state["messages"],
               {"reviews": verdicts})  # fmt: skip


# --- routed: plan or react, decided per request -------------------------------------------------
class Route(BaseModel):
    reason: str = Field(description="one short sentence", max_length=300)
    route: Literal["plan", "react"]


ROUTER_PROMPT = """You decide how a maintenance assistant should work on a request, before it
starts. It can call tools that look up equipment, sensor readings, work orders, events, alerts
and failure codes, and tools that change work orders. One call can filter by equipment and
date and count or group records (by year, month, type, component or failure code).
- plan: the answer needs the result of one call before the next call can be made, or the same
  lookup repeated for every piece of equipment. For example: which equipment has the most of
  something, or a detail about the most frequent alert or the busiest day.
- react: one call, or a few independent ones, return what is needed: a single record, a count
  or grouped counts, a statistic over a period, a list, or a change to records. That includes
  a maximum over the periods of one piece of equipment, which one grouped count answers.
Pick the route; when in doubt, pick react."""


async def choose_route(llm: ChatOpenAI, question: str) -> tuple[str, str, BaseMessage | None]:
    """(route, reason, raw reply). A reply that cannot be parsed falls back to react."""
    router = llm.with_structured_output(Route, method="json_schema", include_raw=True)
    try:
        out = await router.ainvoke([SystemMessage(ROUTER_PROMPT), HumanMessage(question)])
    except Exception as exc:  # a router failure must not cost the request
        return "react", f"router failed ({type(exc).__name__})", None
    route: Route | None = out["parsed"]
    if route is None:
        return "react", "router reply not parsed", out["raw"]
    return route.route, route.reason, out["raw"]


async def run_routed(
    question: str,
    kit: Toolkit,
    llm: ChatOpenAI,
    *,
    max_steps: int = 16,
    system: str = SYSTEM_PROMPT,
) -> Run:
    """One router call picks plan_execute or react for this request; the router call counts
    against the same budget of model calls."""
    start = time.perf_counter()
    route, reason, raw = await choose_route(llm, question)
    design = run_plan_execute if route == "plan" else run_react
    run = await design(question, kit, llm, max_steps=max_steps - 1, system=system)
    tin, tout = _usage([raw] if raw is not None else [])
    return Run(run.answer, run.steps + 1, time.perf_counter() - start, run.tokens_in + tin,
               run.tokens_out + tout, run.stopped, run.messages,
               {**run.notes, "route": route, "route_reason": reason})  # fmt: skip


AgentDesign = Callable[..., Awaitable[Run]]
DESIGNS: dict[str, AgentDesign] = {
    "react": run_react,
    "plan_execute": run_plan_execute,
    "react_verify": run_react_verify,
    "routed": run_routed,
}
