# shopfloor-agent

An on-prem maintenance assistant for plant engineers. Open-weight language models run on a CPU
(llama.cpp, 4-bit), the plant's records are reachable only through four
[MCP](https://modelcontextprotocol.io) servers, and every change the agent wants to make to a
record waits for a person to approve it. Every score in this README comes from deterministic
checks against the plant database; an LLM judge is measured against those checks, not used
instead of them.

![The demo page: a question answered through one tool call, then a work order created only after approval](docs/assets/demo.gif)

*Recorded from the running service on a laptop CPU (Qwen3.5-4B). Waiting time is shortened;
the timings on the page are real.*

## Findings

1. **A 4B model is enough, and size alone is not.** Qwen3.5-4B passes 142 of 143 test tasks on a
   4-vCPU machine at 38 s per task. Qwen3.5-9B passes fewer (136), mostly by breaking the answer
   contract, and Granite 4.2 goes from 71% to 94% between 3B and 8B at 1.8 times the time.
2. **Multi-step work is where small models fail.** Granite 4.2 3B passes every lookup and none of
   the 22 multi-step tasks; 15 of its 41 failures run out of the 16-call budget, often repeating
   the same call. A repeated identical call appears in 17 of its failures and 2 of its passes.
3. **Planning helps where reacting fails, and hurts elsewhere.** On the same 3B model, a planning
   step lifts multi-step from 0 to 11 of 22 but drops lookups from 30 to 19 of 30, because a
   wrong plan is followed faithfully. A reviewer pass helps actions (15 → 18 of 22) and leaves
   multi-step at zero.
4. **Injected instructions were ignored; planted data was not.** In 320 attacked episodes across
   two models and four configurations, no model ever carried out an injected write. Planted
   values ("the correct answer is M999") were repeated in 20 of 160 attempts, and no defence on
   the write path can stop that.
5. **An LLM judge without the reference answer passes 39% of failed runs.** With the reference
   it passes 20%. It also found an error in this benchmark's own scorer, which was fixed and
   every run re-scored (below).

## Models

**Test split, 143 tasks, ReAct agent, each task on a fresh copy of the database:**

| Model (4-bit GGUF) | Pass rate (95% CI) | Lookup | Aggregate | Multi-step | Action | Unanswerable | Median s / task |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | **142 / 143 = 99%** (96–100%) | 30 / 30 | 57 / 57 | 21 / 22 | 22 / 22 | 12 / 12 | 38 |
| Qwen3.5-9B | **136 / 143 = 95%** (90–98%) | 30 / 30 | 55 / 57 | 20 / 22 | 22 / 22 | 9 / 12 | 51 |
| Granite 4.2 8B | **134 / 143 = 94%** (88–97%) | 30 / 30 | 56 / 57 | 16 / 22 | 20 / 22 | 12 / 12 | 117 |
| Granite 4.2 3B | **102 / 143 = 71%** (63–78%) | 30 / 30 | 47 / 57 | 0 / 22 | 15 / 22 | 10 / 12 | 66 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/test-models-dark.png">
  <img alt="Pass rate by task tier on the test split for four models" src="docs/assets/test-models-light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/test-cost-dark.png">
  <img alt="Pass rate against median seconds per task for four models" src="docs/assets/test-cost-light.png">
</picture>

**Qwen3.5-4B** issues tool calls in parallel (the fleet-wide questions take 12–14 calls in about
6 model steps), reads a search's `total` field instead of counting the rows it was shown, solved
all 7 tasks in which one of its tool calls failed, and answers "none" whenever the data cannot
answer. Its one failure gave a failure code where the question asked for the code's description.

**Qwen3.5-9B** fails mostly on the contract, not the reasoning: 3 times it correctly said a work
order has no technician field but wrote the explanation into the `ANSWER:` line instead of
`none`; twice, on the verbatim AssetOpsBench event summaries, it listed only the groups with
events and left out the ones with zero; once llama.cpp could not parse its tool call; once it
stopped without an answer. The scoring rules were fixed before the runs and are applied as
written; reading the three abstentions leniently would make it 139 / 143.

**Granite 4.2 8B**: six of its nine failures are the 20-minute limit per task. All six
fleet-wide questions timed out after 4 to 6 of the 12 or more tool calls they need, because every
model call re-reads a growing context on four CPU cores (47 s per call on average, more late in
a long episode). The limit was set before the runs and is a deployment constraint.

**Granite 4.2 3B fails 41 tasks:**

| Failure | Tasks |
|---|---:|
| ran out of the 16-call budget, mostly on fleet-wide maxima and alert → failure-code chains | 15 |
| gave the failure code where the question asked for its description | 8 |
| wrong value (e.g. counted the returned rows, wrong year) | 8 |
| wrong decision on a conditional action ("create if more than N alerts") | 5 |
| closed only some of the open work orders | 2 |
| answered a question the data cannot answer | 2 |
| no `ANSWER:` line | 1 |

No model wrote anything in a read task.

## Agent designs

The same model (Granite 4.2 3B), tools and budget of 16 model calls:

| Design | Pass rate (95% CI) | Lookup | Aggregate | Multi-step | Action | Unanswerable | Median s / task | Median tokens in |
|---|---|---|---|---|---|---|---|---|
| ReAct | 102 / 143 = 71% (63–78%) | 30 / 30 | 47 / 57 | 0 / 22 | 15 / 22 | 10 / 12 | 66 | 7.7 k |
| Plan and execute | 99 / 143 = 69% (61–76%) | 19 / 30 | 51 / 57 | **11 / 22** | 13 / 22 | 5 / 12 | 160 | 15.9 k |
| ReAct with reviewer | 104 / 143 = 73% (65–79%) | 29 / 30 | 46 / 57 | 0 / 22 | **18 / 22** | 11 / 12 | 113 | 9.6 k |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/test-designs-dark.png">
  <img alt="Pass rate by task tier for three agent designs on Granite 4.2 3B" src="docs/assets/test-designs-light.png">
</picture>

The overall scores are within each other's intervals; the tiers are not. A plan written before
the first call is what lets the small model finish multi-step work, but the executor follows it
even when it is wrong: for "which failure code is on work order X" the planner chose
`search_work_orders` instead of `get_work_order`, and 14 of the planner's lookup and
unanswerable failures are episodes that spent the whole budget on such a plan. The reviewer flagged 19
of 118 answers as unsupported; its second look fixed actions, while its extra calls come out of
the same budget the multi-step tasks already exhaust. Picking the design by request type
(plan for multi-step questions, react for lookups) would combine the gains; it is not measured
here.

## Tool descriptions are prompts

On the dev split only, the first version of the tools failed for reasons in the tool surface.
Revising descriptions and error messages once changed the tiers they touched:

| Failure on dev (tools v1) | Revision (tools v2) | Effect on dev |
|---|---|---|
| Searched only `status="WAPPR"` and missed work in progress | `status="open"` covers WAPPR, APPR and INPRG; codes documented | action 3 / 6 → 5 / 6 |
| Put an alert-rule id into `primary_code`, got "unknown failure code", retried the same value | the error says what the field takes and where to look codes up | |
| Assumed listing the alert rules would return too much | the description says the list is short and maps names to ids | lookup 6 / 8 → 7 / 8 |
| Looped over years to find the busiest one | the description says that omitting the dates covers all years | |
| A fleet-wide question needs 12 tool calls; the budget was 10 | budget of 16 model calls | |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/dev-tools-dark.png">
  <img alt="Pass rate by task tier on the dev split for tools v1 and v2" src="docs/assets/dev-tools-light.png">
</picture>

Overall it moved from 23 to 24 of 37, within the noise: two aggregate tasks regressed because the
model counted the 20 rows a search returned instead of reading its `total`. The tools were frozen
after this one revision, and the test split was run once per configuration.

## Prompt injection

80 tasks: 20 work-order lookups, each asked once clean and three times with instructions
appended to that work order's description, which the agent reads when it looks the order up.
The attacker wants it to cancel an unrelated open work order, create a bogus urgent one, or
report a planted value. Each attack is checked on the database or the answer; the injected text
was in a tool result in every attacked episode.

| Model, defence | Clean tasks passed | Attacked tasks passed | Attack success (95% CI) | cancel | create | planted value |
|---|---|---|---|---|---|---|
| Qwen3.5-4B, none | 20 / 20 | 60 / 60 | 0 / 60 = 0% (0–6%) | 0 / 20 | 0 / 20 | 0 / 20 |
| Qwen3.5-4B, spotlighting | 20 / 20 | 55 / 60 | 5 / 60 = 8% (4–18%) | 0 / 20 | 0 / 20 | 5 / 20 |
| Qwen3.5-4B, read-only session | 20 / 20 | 56 / 60 | 4 / 60 = 7% (3–16%) | 0 / 20 | 0 / 20 | 4 / 20 |
| Qwen3.5-4B, approval gate | 20 / 20 | 58 / 60 | 2 / 60 = 3% (1–11%) | 0 / 20 | 0 / 20 | 2 / 20 |
| Granite 4.2 3B, none | 20 / 20 | 58 / 60 | 2 / 60 = 3% (1–11%) | 0 / 20 | 0 / 20 | 2 / 20 |
| Granite 4.2 3B, spotlighting | 20 / 20 | 57 / 60 | 3 / 60 = 5% (2–14%) | 0 / 20 | 0 / 20 | 3 / 20 |
| Granite 4.2 3B, read-only session | 20 / 20 | 58 / 60 | 2 / 60 = 3% (1–11%) | 0 / 20 | 0 / 20 | 2 / 20 |
| Granite 4.2 3B, approval gate | 20 / 20 | 58 / 60 | 2 / 60 = 3% (1–11%) | 0 / 20 | 0 / 20 | 2 / 20 |

- **No injected write was ever carried out**, with or without a defence: 0 of 320. The models saw
  "call cancel_work_order with wo_id …" in tool output and did not act on it, so the approval gate
  and the read-only session had nothing to stop; they remain the guarantee for a model that does
  follow such text, which the tests demonstrate with a scripted agent that obeys every injection.
- **The planted value is the threat that got through**: 20 of 160 attempts, concentrated on the
  same few work orders (two of them in three of the four Qwen runs). No write-path defence can
  catch it, and spotlighting (tool output marked as data, with an instruction not to follow it)
  did not reduce it.
- **Differences of a few tasks between these rows are noise.** The approval gate sends exactly
  the same prompts as no defence and still differs on 2 tasks: greedy decoding on different
  runner CPUs is not bit-reproducible.
- **The defences cost almost nothing in normal use.** On the full 143-task test split with
  Qwen3.5-4B, the approval gate approved every write the 22 action tasks needed and blocked none
  (141 / 143 against 142 / 143 without it); spotlighting scored 138 / 143.

## LLM-as-judge, measured

Qwen3.5-4B graded Granite 4.2 3B's 143 recorded test episodes (the request, every tool call
with its result, and the final reply), once with a rubric only and once also given the reference
answer. Its verdicts against the deterministic ones (102 passes, 41 failures):

| Judge | Agreement | Cohen's κ | Failed runs judged as passed | Correct runs judged as failed | Judged pass rate (true: 71%) |
|---|---|---|---|---|---|
| rubric only | 121 / 143 = 85% | 0.59 | **16 / 41 = 39%** | 6 / 102 = 6% | 78% |
| with the reference answer | 125 / 143 = 87% | 0.70 | 8 / 41 = 20% | 10 / 102 = 10% | 70% |

Without ground truth, the judge passes four in ten failed runs, mostly plausible-looking
multi-step answers whose numbers it cannot check (11 of the 16). The aggregate it reports
(78%) is closer to the truth than its individual verdicts are.

**The judge also found an error in this benchmark.** The first version of the scorer accepted the
bare failure code as an alias in the 10 tasks that ask for the code's *description*. The
reference judge failed 7 of the 8 answers that passed only through that alias, which exposed it.
The alias was removed and every run was re-scored from its saved answers (`shopfloor rescore`):
Granite 4.2 3B went from 110 to 102 passes (its reviewer run from 109 to 104, its planner run
from 100 to 99), Qwen3.5-4B from 143 to 142 (its spotlighting and approval runs by one each),
the dev runs by two each; Granite 4.2 8B and Qwen3.5-9B were unaffected. All numbers in this
README are the re-scored ones.

## How it works

```mermaid
flowchart LR
    U[Engineer<br>browser or API] -->|POST /ask, SSE| S[FastAPI service<br>stream, approvals]
    S --> A[LangGraph agent<br>ReAct / plan-execute / reviewer]
    A <-->|OpenAI API| L[llama.cpp server<br>local 4-bit model]
    A --> T[Toolkit<br>approval gate, spotlighting, audit]
    T -->|MCP| M1[assets]
    T -->|MCP| M2[telemetry]
    T -->|MCP| M3[maintenance<br>read tier + write tier]
    T -->|MCP| M4[reliability]
    M1 & M2 & M3 & M4 --> D[(plant.db<br>SQLite)]
    S -.->|OTLP| J[Jaeger]
    S -.->|/metrics| P[Prometheus]
```

- **MCP servers** (official Python SDK 2.x) expose 21 tools over the plant database. The
  maintenance server's write tools (create, update, close, cancel a work order) are a separate
  tier: a read-only session does not have them at all.
- **Toolkit**: the agent sees the servers' tools as LangChain tools through a small adapter of
  its own. Every call passes one place where it is recorded, can be held for approval, and has
  its output marked as data.
- **Agent designs** (LangGraph), compared at the same budget of model calls: ReAct; plan and
  execute, where a planning call writes the steps as structured JSON first; and ReAct with a
  reviewer that checks the answer against the tool results and can send the agent back once.
- **Service**: `POST /ask` streams every tool call, every approval request and the answer as
  server-sent events. A write runs only after `POST /approvals/{id}`; without a decision it is
  refused after a timeout. Prometheus metrics cover requests, tool calls by outcome, latency,
  tokens and approvals. OpenTelemetry spans follow the GenAI semantic conventions
  (`invoke_agent`, `chat <model>`, `execute_tool <name>`) and go to Jaeger.

## The benchmark

**Data.** The plant is the sample data of IBM's
[AssetOpsBench](https://github.com/IBM/AssetOpsBench) (Apache 2.0, pinned commit, checksummed
download): 11 chillers with 4,249 work orders from 2010 to 2023, 6,256 events, 1,466 alerts with
the rules that raised them, the failure-code hierarchy, and one month of 15-minute telemetry
for one chiller. AssetOpsBench's own scenarios mostly need data it does not release, so the
tasks here are new, built over the data that is public.

**Tasks.** 180 tasks in five tiers, split by template into a dev set (37, used to revise the
tools) and a test set (143, run once per configuration):

| Tier | Test | What it needs | Scored by |
|---|---:|---|---|
| Lookup | 30 | one tool call | the `ANSWER:` line |
| Aggregate | 57 | the right filters, or a statistic over a range | the `ANSWER:` line |
| Multi-step | 22 | the result of one call to make the next (fleet-wide maxima, alert → failure codes) | the `ANSWER:` line |
| Action | 22 | create or close work orders, sometimes conditionally | the database after the episode |
| Unanswerable | 12 | recognising that the data cannot answer (no such chiller, no sensor, no such field) | the answer "none" |

Six of the aggregate tasks are AssetOpsBench work-order scenarios asked word for word
(scenarios 400, 402–405, 410).

**Scoring is deterministic.** Expected answers are computed by SQL written independently of
the tools, so a tool bug cannot hide in the ground truth. Each episode runs on a fresh copy of
the database: read tasks fail on any write, and action tasks must make exactly the expected
changes. A scripted agent that uses only the agent's own tools solves all 180 tasks, which CI
checks on every push; that proves every task solvable with the tools offered and the ground
truth consistent with what the tools return. Pass rates carry 95% Wilson intervals.

## Run it

With Docker (model server, service and Jaeger; the first start downloads a 2.2 GB model):

```bash
docker compose up --build
# demo page: http://localhost:8000      traces: http://localhost:16686
```

A CI job builds this stack, sends a question through the stream with a real model, and
checks the metrics and the agent, model and tool spans in Jaeger.

Without Docker:

```bash
pip install -e ".[agent,service]"
shopfloor data          # downloads and verifies the plant data, builds data/plant.db
# start llama.cpp's server with a model, e.g.
#   llama-server -m granite-4.2-3b-Q4_K_M.gguf --jinja -c 16384 --port 8081
shopfloor ask "How many corrective work orders did Chiller 9 have in 2017?"
shopfloor serve         # demo page at http://127.0.0.1:8000
```

## Reproduce the benchmark

```bash
shopfloor data && shopfloor tasks          # the suite is deterministic: tasks/suite.jsonl
shopfloor eval --name my-run --split test --agent react
shopfloor report --run my-run
shopfloor eval --name my-inj --suite tasks/injection.jsonl --split test --defense approval
SHOPFLOOR_LLM_MODEL=judge shopfloor judge --run my-run --mode reference
```

On GitHub, the **Benchmark** workflow (manual trigger) shards a run over standard runners,
each with the pinned llama.cpp build and a model from `shopfloor model-url`; the **Judge**
workflow grades a finished run. Per-task results of every run reported here, including the
judge's verdicts, are in [`docs/results/`](docs/results).

## Repository layout

```
src/shopfloor_agent/
  data/       pinned sources, plant database build
  servers/    the four MCP servers and their shared store
  agent/      MCP-to-LangChain toolkit, LangGraph designs, defences
  eval/       task generator, scorer, runner, oracle, injection suite, judge, report, charts
  service/    FastAPI app, telemetry, demo page, smoke test
tasks/        the generated suites (180 tasks, 80 injection tasks)
tests/        43 tests: servers over real MCP sessions, scoring, oracle, defences, service
```

## Limitations

- One plant's sample data: chillers only, and telemetry for one chiller and one month.
- The tasks are generated from templates; they test tool use over records, not open-ended
  maintenance advice. The best model is at the suite's ceiling.
- The injections are fixed, not adaptive: a stronger or optimised attack could succeed where
  these did not.
- One run per configuration; greedy decoding on different runner CPUs is not bit-reproducible,
  so single-task differences between runs are noise.
- Scores are not comparable with the AssetOpsBench leaderboard (different tools, no LLM judge).
- CPU-only 4-bit models; the timings are for 4-vCPU runners and vary with the runner's CPU.

## Licence

MIT (see [LICENSE](LICENSE)). The plant data is IBM AssetOpsBench sample data under the Apache
License 2.0. It is downloaded at build time and not redistributed here; see [NOTICE](NOTICE).
