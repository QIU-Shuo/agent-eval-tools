---
name: evaluator-creator
description: Generate an LLM-as-a-judge evaluator from a task context the user provides. The evaluator should be implemented in the language and style that fit the user's repo, call a judge LLM, and return structured output containing a `final_verdict` field. Use this skill whenever the user wants to build, write, or improve an evaluator, judge, grader, scorer, rubric, or automated eval for LLM outputs — e.g., "write me a judge for X", "how do I grade Y", "build an evaluator that scores Z", "I need to grade these completions", or when they paste a task prompt and test cases and ask how to score the outputs. Trigger even if the user doesn't say the word "judge" — any request to automatically score, grade, or verdict LLM outputs qualifies.
---

# Evaluator Creator

Help the user produce a production-ready evaluator in the language and framework that fit the repo: takes the inputs being judged, calls an LLM judge with a tailored prompt, parses the response, and returns structured output. The output always includes a `final_verdict` field so downstream code can rely on a single key.

An LLM-as-a-judge evaluator is a prompt that reliably grades other model outputs. The core difficulty: a vague judge is noisy (same output gets different verdicts across runs), and an over-specified judge overfits to the examples that inspired it and rejects acceptable variations. This skill walks the user through the design decisions so the final prompt sits in the middle.

Default policy: **do not provide a reference answer to the judge, even if the repo contains one.** Gold answers can still be used for validation in step 5, metrics, and test-case selection, but the evaluator prompt itself should judge from the task context, candidate output, and any other allowed non-gold evidence.

## Workflow

Six steps, three layers:

- **Gather** (step 1): collect the task context and test cases.
- **Decide the verdict** (step 2): pick the success axis and the score shape.
- **Design the reasoning** (step 3): tell the judge how to think before it decides.
- **Implement and validate** (steps 4-6): encode the prompt in code, test it on real cases, and place it in the repo.

### 1. Clarify task context and collect test cases

Before drafting, you need:

- **Task context**: what the model being evaluated is doing — the task prompt, or a description of it.
- **Test cases**: 1–5 input/output pairs, ideally with the verdict the user would assign. Used in step 5 for validation, and also let you triangulate what counts as "correct" in this task without asking the user separately.

**Infer before asking.** Scan the current conversation first — if the user already dropped a task prompt or examples, use them. Then check the repo: read the agent/prompt being evaluated if it's there, sample any test datasets (common paths: `data/`, `evals/`, `tests/`, `datasets/`). Only ask when something is genuinely missing.

If test cases are still missing after that, invent 1–2 plausible borderline cases yourself for step 5 — don't block on them.

### 2. Identify the success axis and propose a score format

Two decisions happen together here, because the shape proposal depends on the axis.

**Identify success criteria.** Tasks differ in how directly success can be verified. Some have clear, checkable correctness criteria; others require more holistic judgment; some combine both. The evaluator should match its success criteria to the task's degree of verifiability and ambiguity.

When you name the success criteria, fill three slots concretely — don't leave them abstract, and don't let the judge discover them at inference time:

- **Positive anchor** — what drives the *top* end of the verdict? (Binary: what counts as PASS. Likert: what earns a 7. Rubric: what maxes each dimension.) For tasks with an observable outcome, name the outcome signal directly rather than describing a process.
- **Override conditions** — what forces the *bottom* end regardless of other signals? Name them as a short list. These are the hard constraints the judge may never trade off against the positive anchor.
- **Ignorables** — what deviations must the judge *not* let move the score in either direction? Name them: style, ordering, verbosity, hedging, minor procedural deviations that don't violate an override condition. Without this slot filled, judges tend to slide into strict-auditor mode and penalize these by default.

Every verdict shape has a high end, a low end, and a middle — these three slots tell the judge what pushes the verdict to each end and what shouldn't move it at all.

**Score shape** — binary, Likert, multi-dimensional rubric, numeric. This is a design preference, so it isn't sitting in a file, but you can propose a default from what's visible:

- If the criteria are inherently multi-dimensional (e.g. "tone + correctness + completeness"), propose a rubric with one score per dimension.

### 3. Design the prompting strategy

The axis and shape are settled — now decide *how the judge should reason before it answers*. Guide how the judge thinks, not just what label it emits. Build a reasoning protocol from the task in front of you rather than picking from a fixed menu of prompt types.

You should first **identify evidence and verification are allowed or required** Should the judge rely only on the candidate output, or also on structured context and task-derived checks? Is there a cheaper decisive validation path — e.g. inspect intermediate steps, validate intermediate results, test the final answer against the question constraints, or write a lightweight checker for the load-bearing claim?

From those answers, guide how the judge thinks.

The judge's job is to evaluate, not to re-do the task. Judging is generally easier than answering: verifying a claim, spotting a flaw, or checking against a criterion usually requires less capability than producing a correct answer from scratch. The scaffolds below should exploit this asymmetry. Avoid prompts that ask the judge to fully solve the task and only then compare; that throws the asymmetry away and can produce a judge whose own answer is worse than the candidate's.

- **Reasoning-first form filling** — ask the judge to fill `reasoning` before `final_verdict` in one JSON response. This is the default because it gives a clear audit trail and usually improves stability.
- **Extract before judging** — when correctness depends on what answer the candidate actually committed to, tell the judge to first identify the claimed answer or key claim, then verdict against that extracted target.
- **Derive explicit checks first** — when the task is complex, have the judge first turn the criterion into 3–5 short evaluation checks, then judge against those checks. This is the default upgrade path when a one-shot judge is too loose or too hand-wavy.
- **Critique before verdict** — when the main failure mode is invalid reasoning or unsupported claims, have the judge explicitly look for the first decisive flaw before deciding. This is often better than asking it to fully solve the task from scratch.
- **Prefer concrete validation paths** — if the task can be judged more reliably by checking intermediate steps, validating intermediate results, or using a lightweight deterministic checker, do that before falling back to holistic judgment.

Treat these as composable pieces, not exclusive modes. A strong evaluator may extract the candidate's final answer, derive a few explicit checks, verify one critical claim independently, and only then issue a verdict.

Always-on guidance:

- **Be minimal.** Keep only instructions that change the verdict. Cut persona fluff, repeated criteria, and generic warnings.
- **Generalize from examples.** Treat provided examples as evidence about the underlying rule, not templates to imitate. Express criteria so they hold on unseen cases, not just the cases used during drafting.
- **Use structured output.** Ask for a fixed JSON shape so downstream code can parse the result reliably and the judge has an explicit place to put reasoning and verdict.


### 4. Draft the evaluator implementation

Implement the evaluator in the language and style already used by the repo when there is a clear local pattern. Match the project's existing eval harness, client wrappers, typing style, and file layout. Only default to Python when the repo gives no stronger signal.

**Implementation shape** — match the project's conventions if there are any (sync vs async, client passing, model name, module style). If the repo has a config module like `config/models.py` with pre-built clients, use those. If the repo is in another language, follow that ecosystem instead. Otherwise default to Python:

```python
import json
import anthropic

JUDGE_SYSTEM = """<2–5 sentence role + task>"""
JUDGE_USER_TEMPLATE = """<user prompt with {placeholders}, describing what each schema field should contain>"""

# Constrain the output format — adjust the schema to match the score shape from step 2.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "final_verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
    },
    "required": ["reasoning", "final_verdict"],
    "additionalProperties": False,
}

async def grade(*, client: anthropic.AsyncAnthropic, model: str = "claude-opus-4-7", **inputs) -> dict:
    """Grade <task>. Returns dict containing `final_verdict` and `reasoning`."""
    response = await client.messages.create(
        model=model,
        # Use a very high completion cap. Tight caps starve judge reasoning and
        # can turn an otherwise good evaluator into a noisy one.
        max_tokens=64000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": JUDGE_USER_TEMPLATE.format(**inputs)}],
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return json.loads(text)
```

The snippet above uses the Anthropic SDK as a concrete example. The same principle applies regardless of provider — OpenAI, LiteLLM, and others all expose structured-output / JSON-schema enforcement, only the parameter name differs:

- **Always enforce the output format via the SDK's structured-output mode.** This is the contract downstream code relies on — no parse-failure fallback, no enum typos like `"CORRRECT"`. Anthropic: `output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}`. OpenAI (differ from ChatCompletion API and Responses API): `response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}`. LiteLLM passes either form through. If a typed-model helper is available (Anthropic's `client.messages.parse(output_format=...)`, OpenAI's `client.beta.chat.completions.parse(...)`), prefer it for typed access.

**Output JSON contract** — `final_verdict` is always required in the schema. Adjust its shape to match the score format from step 2:

- Binary: `{"type": "string", "enum": ["PASS", "FAIL"]}` (default; swap labels for task-appropriate ones like `["CORRECT", "INCORRECT"]` when they fit better)
- Likert: `{"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7]}` — 7 points is the default (more granularity than 5, still anchorable; `enum` instead of `minimum/maximum` since numeric range constraints aren't supported)
- Rubric: top-level `final_verdict` as overall PASS/FAIL plus a sibling `"scores"` object whose properties are one per dimension

Always include `reasoning` as a required string field, listed *before* `final_verdict` in the schema's `properties` order — that's the model's cue to think through the case before deciding.

**Prompt design** — the system + user prompts live as constants in the function. System: short, role + task. User: criteria + ambiguity policy + content placeholders + a brief description of what each schema field should contain.

**Don't starve the completion budget.** Avoid setting `max_tokens` / `max_completion_tokens` caps for judge calls. Use the largest practical completion limit the provider allows, or at least a deliberately high ceiling. A low cap blocks reasoning, increases truncation risk, and can make the evaluator look worse than the prompt actually is.

**Rendering structured inputs.** If what's being judged is structured (agent trajectory, list of tool calls, JSON payload, retrieved-document set), don't dump it raw. Write a small task-specific `render_*()` helper that flattens the structure into compact text, and call it before substituting into the template.

**Render only what's load-bearing for the verdict.** Walk through the structure and ask, *"would the judge's verdict change if this field were missing?"* If no, drop it. Common offenders to strip: request/trace IDs, timestamps, model versions, internal flags, token-usage stats, retry counts, schema metadata. Keeping them in the prompt burns tokens, distracts the model, and gives it spurious correlations to anchor on. The shorter the rendered input, the more reliable the verdict.


```python
def render_trajectory(messages: list[dict]) -> str:
    """Flatten role/content into a transcript. Drops timestamps, message IDs, usage stats."""
    return "\n\n".join(
        f"## {m['role'].upper()}\n{m['content']}" for m in messages
    )

# In grade():
rendered = render_trajectory(trajectory)
messages=[{"role": "user", "content": JUDGE_USER_TEMPLATE.format(trajectory=rendered)}]
```

Don't bundle a one-size-fits-all renderer — what counts as load-bearing depends on the data shape and the criteria, so write a fresh helper per evaluator.

**Robustness** — `output_config` enforces the JSON shape, so `json.loads()` won't fail under normal use. The remaining failure modes are rare: `stop_reason == "refusal"` (safety refusal — output may not match schema) and `stop_reason == "max_tokens"` (truncated mid-JSON). If a production batch matters, check both before parsing and surface the offending case for review.

### 5. Validate against 5–10 test cases

Assemble **5 to 10 cases** from any available source — user-provided, inferred from the repo (existing test sets, agent traces, known-hard examples), or invented. If gathering gets you fewer than 5, invent the rest; stop at 10 — beyond that, extra cases rarely surface new failure modes and instead push the prompt toward overfitting. Aim for coverage over count: include clear passes, clear fails, and at least 2 borderline cases where the verdict is non-obvious — borderline territory is where judges break.

**Prefer running the function** if API keys and a client are available — write a small driver that calls `grade()` on each case and prints a table of `case → final_verdict → expected → reasoning`. Real verdicts beat simulated ones. If running isn't feasible (no keys, no time), simulate by reasoning through what the LLM would plausibly output.
**Improve the prompt from failures.** Where the verdict is wrong, inconsistent, or ambiguous, fix the prompt inside the function — don't just flag the issue. Treat each case as evidence of a **GENERAL** failure mode and rewrite the criterion so it applies to unseen cases too; don't overfit to the specific example. If two or more cases expose the same gap, restructure rather than patch.
**Generalize over perfection.** You don't need every test case to pass. A concise, generalizable criterion beats a prompt that nails every example. If fixing one idiosyncratic case would require bloating the prompt or narrowing the criterion to that case, leave the mismatch and move on.
**Cap the loop at 3 rounds.** Revise the prompt, re-run the cases, revise again — at most 3 iterations. Past round 3, further edits tend to overfit to the test set rather than generalize. If cases are still failing at round 3, ship what you have (per *generalize over perfection*) and note the open gaps in the usage comment.

### 6. Present

Write the evaluator directly into the appropriate place in the repo — don't dump a code block for the user to copy. Pick the location from what's visible: existing `evals/`, `evaluators/`, `judges/`, or `tests/` directories; a sibling file next to the agent/prompt being evaluated; or the project's eval harness module if there is one. If nothing obvious exists, create a single new file at a sensible path (e.g. `evals/grade_<task>.py`) and mention it in your reply.

Embed a usage comment in the file itself — at the top of the module or as a docstring on `grade()` — covering how to call the function, what inputs it expects, and what the returned dict looks like. This replaces the separate rationale: the comment travels with the code and is there when the user revisits the evaluator months later.
