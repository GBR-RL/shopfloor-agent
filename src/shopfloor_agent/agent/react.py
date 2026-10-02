"""ReAct agent (LangGraph): the model alternates between calling tools and reading their results
until it answers or runs out of steps."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

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


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


@dataclass
class Run:
    answer: str
    steps: int
    seconds: float
    tokens_in: int = 0
    tokens_out: int = 0
    stopped: str = "answered"  # answered | step_limit
    messages: list[AnyMessage] = field(default_factory=list)


async def run_react(question: str, kit: Toolkit, llm: ChatOpenAI, *, max_steps: int = 10) -> Run:
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
    app = graph.compile()

    start = time.perf_counter()
    state = await app.ainvoke(
        {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)], "steps": 0},
        {"recursion_limit": 2 * max_steps + 2},
    )
    messages = state["messages"]
    last = messages[-1]
    usage = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
    return Run(
        answer=last.text if isinstance(last, AIMessage) and not last.tool_calls else "",
        steps=state["steps"],
        seconds=time.perf_counter() - start,
        tokens_in=sum(u["input_tokens"] for u in usage),
        tokens_out=sum(u["output_tokens"] for u in usage),
        stopped="answered" if isinstance(last, AIMessage) and not last.tool_calls else "step_limit",
        messages=messages,
    )
