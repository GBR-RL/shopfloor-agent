"""Observability: Prometheus metrics and OpenTelemetry traces.

Traces follow the OpenTelemetry GenAI semantic conventions (gen_ai.* attributes): one span per
request, per model call and per tool call. They are exported over OTLP when
OTEL_EXPORTER_OTLP_ENDPOINT is set (Jaeger in the Compose stack) and are no-ops otherwise.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from opentelemetry import trace
from prometheus_client import CollectorRegistry, Counter, Histogram

tracer = trace.get_tracer("shopfloor_agent")


def setup_tracing(service_name: str = "shopfloor-agent") -> bool:
    """Installs an OTLP exporter if an endpoint is configured; returns whether it did."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return True


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.requests = Counter("shopfloor_requests_total", "Questions answered",
                                ["agent", "outcome"], registry=self.registry)  # fmt: skip
        self.request_seconds = Histogram(
            "shopfloor_request_seconds", "End-to-end time per question", ["agent"],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200), registry=self.registry,
        )  # fmt: skip
        self.tool_calls = Counter("shopfloor_tool_calls_total", "Tool calls",
                                  ["tool", "outcome"], registry=self.registry)  # fmt: skip
        self.tool_seconds = Histogram("shopfloor_tool_seconds", "Tool call latency", ["tool"],
                                      buckets=(0.005, 0.01, 0.05, 0.1, 0.5, 1, 5),
                                      registry=self.registry)  # fmt: skip
        self.tokens = Counter("shopfloor_llm_tokens_total", "Model tokens", ["direction"],
                              registry=self.registry)  # fmt: skip
        self.approvals = Counter("shopfloor_approvals_total", "Write approvals by decision",
                                 ["decision"], registry=self.registry)  # fmt: skip


class ModelSpans(AsyncCallbackHandler):
    """A span per model call with token usage, and token counters."""

    def __init__(self, metrics: Metrics, model: str) -> None:
        self.metrics, self.model = metrics, model
        self._spans: dict[UUID, Any] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span = tracer.start_span(f"chat {self.model}")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", self.model)
        self._spans[run_id] = span

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.pop(run_id, None)
        usage: dict[str, int] = {}
        for gen in response.generations:
            for g in gen:
                meta = getattr(getattr(g, "message", None), "usage_metadata", None)
                if meta:
                    usage = {"in": meta["input_tokens"], "out": meta["output_tokens"]}
        if usage:
            self.metrics.tokens.labels("in").inc(usage["in"])
            self.metrics.tokens.labels("out").inc(usage["out"])
        if span is not None:
            if usage:
                span.set_attribute("gen_ai.usage.input_tokens", usage["in"])
                span.set_attribute("gen_ai.usage.output_tokens", usage["out"])
            span.end()

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.pop(run_id, None)
        if span is not None:
            span.record_exception(error)
            span.end()
