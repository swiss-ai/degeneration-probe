"""Phase 4 of the dataset generation pipeline: LLM-judge calibration labels.

The heuristics in ``label.py`` (``repetition_score``, ``lrs_score`` /
``lrs_period_repeat_count``) are cheap proxies for "this rollout is
degenerating." This module asks an LLM to independently judge a small,
targeted sample of rollouts -- blind to the heuristic scores, seeing only the
prompt and completion text -- so the heuristics' thresholds can be checked
against something closer to ground truth.

Two pieces:

1. ``select_calibration_sample`` -- builds the rollout sample to judge, from
   two strata (a rollout can land in only one, "truncated" takes priority):
     - "truncated": hit ``max_new_tokens`` without reaching EOS. The
       population most likely to contain real degeneration loops.
     - "flagged_natural_eos": heuristically flagged (``max_repetition_score >
       DEGENERATION_THRESHOLD`` or ``lrs_period_repeat_count >=
       MIN_PERIOD_REPEAT_COUNT``) but the model stopped on its own -- the
       interesting population for checking false positives (or degeneration
       that self-terminates).
   Written once to ``paths.llm_judge_sample_path`` so the sample is fixed and
   reproducible across judge backends/reruns.

2. A backend-agnostic judging loop (``run_judging``) built on a small
   ``JudgeBackend`` protocol with three implementations -- ``AnthropicBackend``
   (native structured outputs, guaranteed-valid JSON, pay-per-token),
   ``ClaudeAgentSDKBackend`` (routes through the real `claude` CLI subprocess
   with a ``claude setup-token`` OAuth token, drawing on Pro/Max subscription
   usage instead of metered billing), and ``OpenRouterBackend``
   (OpenAI-compatible endpoint; not every open model reliably honors a JSON
   schema response_format, so it falls back to prompting for JSON and
   retrying on parse failure). Swapping backends is a CLI flag, not a code
   change. Results are a resumable manifest (``paths.llm_judge_results_path``,
   one file per backend) using the same atomic-write / skip-already-done
   pattern as ``cache_activations.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import pandas as pd
from pydantic import BaseModel, Field

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig
from degeneration_probe.dataset_gen.label import write_shard_atomic

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"

# Matches label.py's DEFAULT_DEGENERATION_THRESHOLD / DEFAULT window's
# canonical cutoff -- kept as a separate constant here (rather than importing
# label.py's) because this one specifically defines the *judge sample*, not
# the labeling pipeline itself.
DEGENERATION_THRESHOLD = 0.8
MIN_PERIOD_REPEAT_COUNT = 3

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_PARSE_RETRIES = 2


# --- judge schema --------------------------------------------------------------

class JudgeVerdict(BaseModel):
    is_degenerating: bool = Field(
        description="Whether the completion exhibits degenerate repetition -- "
        "looping, echoing, or otherwise unproductive repeated text -- as "
        "opposed to repetition that is a normal/expected part of the task "
        "(e.g. restating a step in a math derivation, formatting boilerplate)."
    )
    repeated_pattern: Optional[str] = Field(
        default=None,
        description="The exact repeated span (verbatim substring from the "
        "completion), if is_degenerating is true. null otherwise.",
    )
    confidence: str = Field(description="One of: high, medium, low.")
    comment: str = Field(
        description="A short free-text note, e.g. 'high repetition but not "
        "degenerating -- this is a step-by-step long division' or 'clear "
        "infinite loop on the word occurs starting mid-response'."
    )


JUDGE_JSON_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "is_degenerating": {"type": "boolean"},
        "repeated_pattern": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "comment": {"type": "string"},
    },
    "required": ["is_degenerating", "repeated_pattern", "confidence", "comment"],
    "additionalProperties": False,
}

JUDGE_SYSTEM_PROMPT = """You are judging whether a language model's completion \
degenerated into repetitive, low-information text (looping on a phrase, \
sentence, or pattern without making progress), as opposed to repetition that \
is a normal, expected part of the task (e.g. restating steps in a derivation, \
enumerating similar cases, formatting/boilerplate). You will see only the \
prompt and the completion -- judge on the text alone."""

JUDGE_USER_TEMPLATE = """PROMPT:
{prompt_text}

COMPLETION:
{completion_text}

Judge whether this completion is degenerating."""


# --- backend protocol ------------------------------------------------------------

class JudgeBackend(Protocol):
    name: str

    def judge(self, prompt_text: str, completion_text: str) -> JudgeVerdict:
        ...


class AnthropicBackend:
    """Structured outputs guarantee valid JSON -- no parse-retry loop needed."""

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL) -> None:
        import anthropic

        self.model = model
        # Zero-arg client resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN /
        # an `ant auth login` OAuth profile, in that order -- works with a
        # Pro/Max subscription login as well as a plain API key.
        self.client = anthropic.Anthropic()

    def judge(self, prompt_text: str, completion_text: str) -> JudgeVerdict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(
                        prompt_text=prompt_text, completion_text=completion_text
                    ),
                }
            ],
            output_config={"format": {"type": "json_schema", "schema": JUDGE_JSON_SCHEMA}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        return JudgeVerdict.model_validate_json(text)


# The one observed text for the "usage actually exhausted" crash (see
# ClaudeAgentSDKBackend's docstring) -- the CLI's last result looked like a
# normal success, so the SDK's ProcessError-replacement falls back to
# printing the bare subtype, "success", instead of any real error detail.
_USAGE_EXHAUSTED_CRASH_TEXT = "Claude Code returned an error result: success"


class UsageExhausted(Exception):
    """Raised when the CLI reports the OAuth token's subscription usage is
    rate-limited (``RateLimitEvent`` status ``"rejected"``, or a 429
    surfacing on the result itself). Distinct from a generic judge failure:
    every other remaining row would fail the same way until the window
    resets, so ``run_judging`` stops the batch on this instead of recording
    hundreds of instant, uninformative failures."""

    def __init__(self, rate_limit_type: Optional[str], resets_at: Optional[int]) -> None:
        self.rate_limit_type = rate_limit_type
        self.resets_at = resets_at
        when = (
            time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(resets_at))
            if resets_at is not None
            else "unknown"
        )
        super().__init__(
            f"subscription usage exhausted (rate_limit_type={rate_limit_type}, "
            f"resets at {when})"
        )


class ClaudeAgentSDKBackend:
    """Routes judge calls through the real Claude Code CLI subprocess (via
    ``claude_agent_sdk.query``) using a ``claude setup-token`` OAuth token, so
    calls draw on Pro/Max subscription usage rather than pay-per-token API
    billing. Not a raw Messages API call -- ``query()`` spawns/talks to the
    actual `claude` binary, so the OAuth token is used the way it's meant to
    be (this is why it's allowed to work at all; a raw API call spoofing a
    "Claude Code" identity to accept the same token would not be legitimate
    and was deliberately not done here).

    ``tools=[]`` disables all built-in tools (no file/bash access -- this is
    a one-shot text-judgment call, not an agentic session) and
    ``setting_sources=[]`` isolates the call from any local CLAUDE.md/hooks/
    settings. ``output_format`` mirrors the Messages API's
    ``output_config.format`` shape, so, like AnthropicBackend, this gets
    schema-validated JSON back directly (via ``ResultMessage.structured_output``)
    with no parse-retry loop needed.

    The CLI streams a ``RateLimitEvent`` whenever subscription usage status
    changes; a ``status == "rejected"`` here (or a 429 on the result) raises
    ``UsageExhausted`` rather than a generic error, so callers (``run_judging``)
    can stop the whole batch instead of burning through every remaining row
    as an instant failure.

    Observed in practice: once usage is actually exhausted, the CLI
    subprocess doesn't reliably deliver either of those signals -- instead it
    emits a normal-looking ``ResultMessage`` (``is_error=False``,
    ``subtype="success"``) and then crashes, which the SDK surfaces as a bare
    ``Exception("Claude Code returned an error result: success")`` raised out
    of the ``async for`` loop itself (see ``claude_agent_sdk._internal.query``
    -- it replaces a post-result ``ProcessError`` with the CLI's last result
    text, which falls back to the literal ``subtype`` when ``errors`` is
    empty). That exact "...: success" text is the only content ever observed
    for this crash -- any other error text from that same code path (e.g. a
    real ``error_during_execution``/``error_max_turns`` subtype, or the CLI's
    own "failed to provide valid structured output" message) reflects a
    genuine one-off judge failure, not usage exhaustion, and is left as a
    normal per-row failure.
    """

    name = "claude_agent_sdk"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, oauth_token: Optional[str] = None) -> None:
        self.model = model
        self.oauth_token = oauth_token or _read_key_file(
            Path.home() / "keys" / ".anthropic_oauth_token"
        )

    def judge(self, prompt_text: str, completion_text: str) -> JudgeVerdict:
        import asyncio

        return asyncio.run(self._judge_async(prompt_text, completion_text))

    async def _judge_async(self, prompt_text: str, completion_text: str) -> JudgeVerdict:
        from claude_agent_sdk import ClaudeAgentOptions, RateLimitEvent, ResultMessage, query

        options = ClaudeAgentOptions(
            tools=[],
            system_prompt=JUDGE_SYSTEM_PROMPT,
            model=self.model,
            output_format={"type": "json_schema", "schema": JUDGE_JSON_SCHEMA},
            setting_sources=[],
            env={"CLAUDE_CODE_OAUTH_TOKEN": self.oauth_token},
        )
        user_content = JUDGE_USER_TEMPLATE.format(
            prompt_text=prompt_text, completion_text=completion_text
        )
        result: Optional[ResultMessage] = None
        rate_limit_rejected: Optional[RateLimitEvent] = None
        try:
            async for message in query(prompt=user_content, options=options):
                if isinstance(message, RateLimitEvent) and message.rate_limit_info.status == "rejected":
                    rate_limit_rejected = message
                elif isinstance(message, ResultMessage):
                    result = message
        except Exception as exc:
            if rate_limit_rejected is not None:
                info = rate_limit_rejected.rate_limit_info
                raise UsageExhausted(rate_limit_type=info.rate_limit_type, resets_at=info.resets_at) from exc
            # A ResultMessage already arrived before this exception fired (the SDK
            # raises a second, generic exception while reaping the process *after*
            # yielding the real result -- see class docstring). If that result
            # says is_error, we already know the real cause (e.g. a content-policy
            # refusal, api_refusal_category="bio" observed for a highly repetitive
            # "kill the X" rollout in practice) -- surface that instead of falling
            # through to the ambiguous usage-exhausted crash-text guess below,
            # which would otherwise misclassify a normal per-row refusal as the
            # whole account's usage being gone and wrongly halt/rotate the batch.
            if result is not None and result.is_error:
                if result.api_error_status == 429:
                    raise UsageExhausted(rate_limit_type=None, resets_at=None) from exc
                raise ValueError(
                    f"claude_agent_sdk judge failed (subtype={result.subtype}, "
                    f"stop_reason={getattr(result, 'stop_reason', None)}, "
                    f"errors={result.errors}): result={result.result!r}"
                ) from exc
            if str(exc) == _USAGE_EXHAUSTED_CRASH_TEXT:
                raise UsageExhausted(rate_limit_type=None, resets_at=None) from exc
            raise

        if rate_limit_rejected is not None:
            info = rate_limit_rejected.rate_limit_info
            raise UsageExhausted(rate_limit_type=info.rate_limit_type, resets_at=info.resets_at)
        if result is None:
            raise ValueError("claude_agent_sdk query produced no ResultMessage")
        if result.is_error:
            if result.api_error_status == 429:
                raise UsageExhausted(rate_limit_type=None, resets_at=None)
            raise ValueError(
                f"claude_agent_sdk judge failed (subtype={result.subtype}, "
                f"errors={result.errors}): result={result.result!r}"
            )
        if result.structured_output is None:
            raise ValueError(
                f"claude_agent_sdk judge returned no structured_output "
                f"(subtype={result.subtype}): result={result.result!r}"
            )
        return JudgeVerdict.model_validate(result.structured_output)


MAX_RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_DELAY_SECONDS = 5.0


def _call_with_backoff(fn, *args, **kwargs):
    """Retry on 429 / 5xx with exponential backoff. Free-tier OpenRouter
    models get temporarily rate-limited upstream fairly often (observed
    directly while building this) -- this is a distinct failure mode from
    "the model didn't return valid JSON" and needs its own retry, not the
    parse-retry loop below."""
    import openai

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (openai.RateLimitError, openai.InternalServerError) as exc:
            last_error = exc
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            delay = RATE_LIMIT_BASE_DELAY_SECONDS * (2**attempt)
            print(f"[openrouter] rate/server error ({exc}), retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES})")
            time.sleep(delay)
    raise last_error  # unreachable, but keeps type-checkers happy


class OpenRouterBackend:
    """OpenAI-compatible endpoint. Many open models don't reliably honor a
    strict JSON-schema response_format, so this tries it first and falls back
    to a plain prompt-for-JSON + parse-and-retry loop on failure. Rate limits
    (common on free-tier models) are retried with backoff independently of
    that fallback."""

    name = "openrouter"

    def __init__(self, model: str = DEFAULT_OPENROUTER_MODEL, api_key: Optional[str] = None) -> None:
        import openai

        self.model = model
        key = api_key or _read_key_file(Path.home() / "keys" / ".openrouter_key")
        self.client = openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    def judge(self, prompt_text: str, completion_text: str) -> JudgeVerdict:
        import openai

        user_content = JUDGE_USER_TEMPLATE.format(
            prompt_text=prompt_text, completion_text=completion_text
        )
        try:
            response = _call_with_backoff(
                self.client.chat.completions.create,
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge_verdict",
                        "schema": JUDGE_JSON_SCHEMA,
                        "strict": True,
                    },
                },
            )
            return JudgeVerdict.model_validate_json(response.choices[0].message.content)
        except openai.BadRequestError:
            pass  # model doesn't support strict json_schema -- fall back below
        except ValueError:
            # response_format was accepted but the content still wasn't valid JSON
            # matching the schema (observed: a model rambling/repeating until its
            # output was cut off mid-object) -- the retry loop below handles this.
            pass

        schema_hint = json.dumps(JUDGE_JSON_SCHEMA, indent=2)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{user_content}\n\nRespond with ONLY a JSON object matching "
                f"this schema, no other text:\n{schema_hint}",
            },
        ]
        last_error: Optional[Exception] = None
        for _ in range(MAX_PARSE_RETRIES + 1):
            response = _call_with_backoff(
                self.client.chat.completions.create,
                model=self.model,
                max_tokens=1024,
                messages=messages,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw[raw.find("{") :]
            try:
                return JudgeVerdict.model_validate_json(raw)
            except Exception as exc:  # noqa: BLE001 -- retry on any parse failure
                last_error = exc
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": f"That was not valid JSON matching the schema "
                        f"({exc}). Reply with ONLY the corrected JSON object.",
                    }
                )
        raise ValueError(f"OpenRouter judge failed to produce valid JSON after retries: {last_error}")


def _read_key_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"No API key found at {path}. Pass api_key explicitly or write the key there."
        )
    return path.read_text().strip()


def build_backend(
    backend_name: str, model: Optional[str] = None, oauth_key_file: Optional[str] = None
) -> JudgeBackend:
    if backend_name == "anthropic":
        return AnthropicBackend(model=model or DEFAULT_ANTHROPIC_MODEL)
    if backend_name == "openrouter":
        return OpenRouterBackend(model=model or DEFAULT_OPENROUTER_MODEL)
    if backend_name == "claude_agent_sdk":
        # oauth_key_file lets a caller point at a specific account's token
        # (e.g. one of several `claude setup-token` logins rotated across
        # sbatch waves) instead of the default `~/keys/.anthropic_oauth_token`.
        oauth_token = _read_key_file(Path(oauth_key_file)) if oauth_key_file else None
        return ClaudeAgentSDKBackend(model=model or DEFAULT_ANTHROPIC_MODEL, oauth_token=oauth_token)
    raise ValueError(
        f"Unknown backend {backend_name!r}, expected 'anthropic', 'openrouter', "
        "or 'claude_agent_sdk'"
    )


# --- calibration sample selection -----------------------------------------------

def select_calibration_sample(config: DatasetGenConfig) -> pd.DataFrame:
    """Build the judge sample: every truncated rollout, plus every rollout
    heuristically flagged as degenerating that nonetheless reached EOS on its
    own. Returns one row per rollout with prompt_id, rollout_idx, domain,
    stratum, prompt_text, completion_text.
    """
    domains = sorted(paths.configured_domain_names(config))

    prompts_df = pd.read_parquet(paths.prompts_path(config), columns=["prompt_id", "domain", "prompt_text"])

    gen_frames, lab_frames = [], []
    for domain in domains:
        gen_frames.append(
            pd.read_parquet(
                paths.generations_shard_path(config, domain, 0),
                columns=["prompt_id", "rollout_idx", "generated_text", "stop_reason"],
            )
        )
        lab_frames.append(
            pd.read_parquet(
                paths.labels_shard_path(config, domain, 0),
                columns=["prompt_id", "rollout_idx", "repetition_score", "lrs_period_repeat_count"],
            )
        )
    gen = pd.concat(gen_frames, ignore_index=True)
    lab = pd.concat(lab_frames, ignore_index=True)

    lab = lab.copy()
    lab["max_repetition_score"] = lab["repetition_score"].apply(_safe_nanmax)
    lab["lrs_period_repeat_count"] = lab["lrs_period_repeat_count"].fillna(0)

    merged = gen.merge(
        lab[["prompt_id", "rollout_idx", "max_repetition_score", "lrs_period_repeat_count"]],
        on=["prompt_id", "rollout_idx"],
        how="left",
    )
    if len(merged) != len(gen):
        raise ValueError("merge changed row count -- duplicate (prompt_id, rollout_idx) in labels")

    truncated = merged["stop_reason"] != "eos"
    flagged = (merged["max_repetition_score"] > DEGENERATION_THRESHOLD) | (
        merged["lrs_period_repeat_count"] >= MIN_PERIOD_REPEAT_COUNT
    )

    merged["stratum"] = None
    merged.loc[truncated, "stratum"] = "truncated"
    merged.loc[(~truncated) & flagged, "stratum"] = "flagged_natural_eos"
    sample = merged[merged["stratum"].notna()].copy()

    sample = sample.merge(prompts_df, on="prompt_id", how="left")
    sample = sample.rename(columns={"generated_text": "completion_text"})
    return sample[
        ["prompt_id", "rollout_idx", "domain", "stratum", "prompt_text", "completion_text"]
    ].reset_index(drop=True)


def _safe_nanmax(arr) -> float:
    import numpy as np

    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or bool(np.all(np.isnan(arr))):
        return float("nan")
    return float(np.nanmax(arr))


# --- resumable judging loop ------------------------------------------------------

def _load_existing_results(results_path: Path) -> pd.DataFrame:
    if results_path.exists():
        return pd.read_parquet(results_path)
    return pd.DataFrame(
        columns=[
            "prompt_id",
            "rollout_idx",
            "status",
            "is_degenerating",
            "repeated_pattern",
            "confidence",
            "comment",
            "error",
        ]
    )


def _done_keys(results_df: pd.DataFrame) -> set:
    ok = results_df[results_df["status"] == "ok"]
    return set(zip(ok["prompt_id"], ok["rollout_idx"]))


def run_judging(
    sample: pd.DataFrame,
    backend: JudgeBackend,
    results_path: Path,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Judge every row in `sample` not already recorded as "ok" in
    `results_path`, writing (atomically) after every rollout so a crash,
    rate-limit, or manual interrupt loses at most one in-flight call.

    If a backend raises ``UsageExhausted`` (subscription usage rate-limited),
    the batch stops immediately rather than recording every remaining row as
    an instant, uninformative failure -- rerunning the same command later
    resumes from where it stopped, once usage resets.
    """
    results_df = _load_existing_results(results_path)
    done = _done_keys(results_df)

    todo = sample[~sample.apply(lambda r: (r.prompt_id, r.rollout_idx) in done, axis=1)]
    if limit is not None:
        todo = todo.head(limit)

    print(f"[{backend.name}] {len(done)} already judged, {len(todo)} remaining")

    for _, row in todo.iterrows():
        record = {
            "prompt_id": row.prompt_id,
            "rollout_idx": int(row.rollout_idx),
            "status": "ok",
            "is_degenerating": None,
            "repeated_pattern": None,
            "confidence": None,
            "comment": None,
            "error": None,
        }
        stop_after_this_row = False
        try:
            verdict = backend.judge(row.prompt_text, row.completion_text)
            record.update(
                is_degenerating=verdict.is_degenerating,
                repeated_pattern=verdict.repeated_pattern,
                confidence=verdict.confidence,
                comment=verdict.comment,
            )
        except UsageExhausted as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"[{backend.name}] STOPPING: {exc}. Rerun the same command later to resume.")
            stop_after_this_row = True
        except Exception as exc:  # noqa: BLE001 -- never let one bad rollout kill the run
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"[{backend.name}] FAILED {row.prompt_id}/{row.rollout_idx}: {exc}")

        results_df = pd.concat(
            [results_df[~((results_df.prompt_id == row.prompt_id) & (results_df.rollout_idx == row.rollout_idx))],
             pd.DataFrame([record])],
            ignore_index=True,
        )
        write_shard_atomic(results_df, results_path)

        if stop_after_this_row:
            break

    return results_df


# --- CLI -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--backend", choices=["anthropic", "openrouter", "claude_agent_sdk"], required=True
    )
    parser.add_argument("--model", type=str, default=None, help="Overrides the backend's default model.")
    parser.add_argument("--limit", type=int, default=None, help="Judge at most this many (new) rollouts.")
    parser.add_argument(
        "--oauth-key-file",
        type=str,
        default=None,
        help="claude_agent_sdk only: path to an OAuth token file (defaults to "
        "~/keys/.anthropic_oauth_token). Lets separate runs rotate across "
        "several accounts' subscription usage.",
    )
    args = parser.parse_args()

    config = DatasetGenConfig.from_yaml(args.config)

    sample_path = paths.llm_judge_sample_path(config)
    if sample_path.exists():
        sample = pd.read_parquet(sample_path)
    else:
        sample = select_calibration_sample(config)
        write_shard_atomic(sample, sample_path)
    print(f"Calibration sample: {len(sample)} rollouts ({sample['stratum'].value_counts().to_dict()})")

    backend = build_backend(args.backend, model=args.model, oauth_key_file=args.oauth_key_file)
    results_path = paths.llm_judge_results_path(config, backend.name)

    start = time.time()
    results_df = run_judging(sample, backend, results_path, limit=args.limit)
    elapsed = time.time() - start
    print(f"[{backend.name}] wrote {len(results_df)} results to {results_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
