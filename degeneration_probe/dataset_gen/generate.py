"""Phase 1 of the dataset generation pipeline: sample rollouts with per-token entropy.

For every prompt in ``paths.prompts_path(config)`` (written by ``build_prompts.py``)
this module generates ``config.n_rollouts_per_prompt`` sampled rollouts, and for
every generated token records both the sampled token id and the entropy of the
model's *actual* next-token distribution at that step.

CRITICAL correctness note -- read before touching the decode loop
-------------------------------------------------------------------
``transformers.generate(output_scores=True, return_dict_in_generate=True)`` would
be the obvious way to get "scores" out of generation, but the returned ``scores``
are the logits *after* temperature scaling and top-p filtering (LogitsProcessor
output -- filtered-out tokens become ``-inf``). Computing entropy from that
measures the uncertainty of the truncated *sampling* distribution, not the
model's actual uncertainty, which would systematically bias entropy down and
defeat the point of the metric (we want high entropy to mean "the model is
genuinely uncertain/diverse", not "diverse after we already threw away the
tail").

So this module never calls ``.generate()``. Instead it runs its own step loop
directly against the model's forward pass (``use_cache=True`` /
``past_key_values``), so a single forward pass per step serves both purposes:

  (a) entropy = -sum(softmax(raw_logits) * log_softmax(raw_logits)) over the
      FULL vocab, at temperature=1, computed from the RAW logits the model
      produced (see ``compute_entropy_from_logits``).
  (b) separately, the SAME raw logits are temperature-scaled and top-p
      filtered (see ``apply_temperature_and_top_p``) to actually sample the
      next token (see ``sample_next_tokens``).

Neither step re-runs the model -- both are pure tensor ops over the one
forward pass's output logits.

Seed scheme
-----------
Every (prompt_id, rollout_idx) needs a distinct, reproducible seed. We use::

    seed = config.seed * PROMPT_HASH_MODULUS * ROLLOUT_IDX_MODULUS
           + stable_hash(prompt_id) * ROLLOUT_IDX_MODULUS
           + rollout_idx

where ``stable_hash`` is the first bytes of ``sha256(prompt_id)`` reduced mod
``PROMPT_HASH_MODULUS`` (1e9). Python's builtin ``hash()`` is randomized
per-process for strings (via ``PYTHONHASHSEED``) so it cannot be used here;
sha256 is stable across processes/machines/python versions.

This deviates from the literal "mod 100_000" suggestion in the original spec:
with only 100_000 buckets and ~370 prompt_ids (an earlier, smaller build's
scale), the birthday bound puts the chance of at least one hash collision
across prompt_ids at roughly 1 - exp(-370^2 / (2*100_000)) =~ 50%, which is
far too likely for comfort (a collision there would silently give two
*different* prompts identical seeds for the same rollout_idx, correlating
supposedly-independent samples). Using a 1e9 modulus instead pushes that same
bound down to ~7e-5 even at that scale, and stays negligible at the current
~3,030-prompt build (and well beyond). ``ROLLOUT_IDX_MODULUS`` (1000) just
reserves headroom so ``+ rollout_idx`` can never carry into the hash term
(``n_rollouts_per_prompt`` is 10).

Sampling determinism
---------------------
Per-row sampling uses one ``torch.Generator`` per (prompt_id, rollout_idx),
seeded once via ``derive_seed`` and advanced across all of that rollout's
decode steps. Generators (and the multinomial draw itself) always run on CPU,
even when the model runs on GPU: CUDA's RNG/algorithm can vary across GPU
architectures and driver versions, so pinning sampling to CPU keeps a given
seed reproducible regardless of which node/GPU the build happens to run on.
The (batch, vocab) softmax probabilities are moved to CPU once per step before
the per-row ``torch.multinomial`` calls -- cheap relative to the forward pass.

Batching
--------
Tasks are grouped into fixed-size batches (left-padded) and each batch is run
until every row in it has stopped (EOS or max_new_tokens), rather than using a
dynamic/continuous-batching scheduler that refills finished slots with new
tasks. This wastes some compute on already-finished rows within a batch, but
keeps the loop simple and easy to verify; see the module docstring's wall-clock
estimate in the CLI usage notes for the throughput this achieves in practice.

Resumability
------------
Before processing a domain, any prompt/rollout already present in that
domain's shard parquet (``paths.generations_shard_path``) OR in a leftover
partial-result file under ``paths.work_root(config) / "generations" / domain``
is skipped. Each completed rollout is first written as its own small JSON file
under the work root (temp-file + ``os.replace`` -- atomic on the same
filesystem), then the shard parquet is rebuilt from
(existing shard rows + all partial files) and swapped into place the same way.
This means a crash can never leave a half-written shard file, and at most the
in-flight batch's results need to be regenerated on the next run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch

from degeneration_probe.dataset_gen import paths
from degeneration_probe.dataset_gen.config import DatasetGenConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset" / "degeneration-dataset-apertus-8b-instruct.yaml"

Task = Tuple[str, int]  # (prompt_id, rollout_idx)

SHARD_COLUMNS = [
    "prompt_id",
    "rollout_idx",
    "seed",
    "generated_token_ids",
    "generated_text",
    "per_token_entropy",
    "num_tokens",
    "stop_reason",
    "wall_time_seconds",
]

PROMPT_HASH_MODULUS = 10**9
ROLLOUT_IDX_MODULUS = 1000


# --- seed derivation ---------------------------------------------------------

def derive_seed(config_seed: int, prompt_id: str, rollout_idx: int) -> int:
    """Deterministic, (practically) distinct seed per (prompt_id, rollout_idx).

    See the module docstring's "Seed scheme" section for the rationale.
    """
    if not (0 <= rollout_idx < ROLLOUT_IDX_MODULUS):
        raise ValueError(
            f"rollout_idx must be in [0, {ROLLOUT_IDX_MODULUS}) for seed distinctness, "
            f"got {rollout_idx}"
        )
    digest = hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()
    stable_hash = int(digest, 16) % PROMPT_HASH_MODULUS
    return (
        config_seed * PROMPT_HASH_MODULUS * ROLLOUT_IDX_MODULUS
        + stable_hash * ROLLOUT_IDX_MODULUS
        + rollout_idx
    )


# --- chat templating (mirrors degeneration_probe/data/dataset.py) ------------

def build_generation_prompt(tokenizer, prompt: str) -> str:
    """Build the chat-templated prefix a rollout's generated tokens continue.

    Exactly mirrors
    ``degeneration_probe.data.dataset.TokenizedProbingDataset._build_generation_prompt``:
    render the single-user-turn conversation with
    ``apply_chat_template(..., add_generation_prompt=True)``, then strip the
    template's own ``bos_token`` back out (many chat templates prepend it, and
    we tokenize with ``add_special_tokens=False`` below to avoid a duplicate).
    """
    conversation = [{"role": "user", "content": prompt}]
    try:
        prompt_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(conversation, tokenize=False)

    if tokenizer.bos_token and tokenizer.bos_token in prompt_text:
        prompt_text = prompt_text.replace(tokenizer.bos_token, "")

    return prompt_text


def load_tokenizer_for_generation(config: DatasetGenConfig):
    """Load the tokenizer (vocab, merges, special tokens, AND chat_template)
    to use for generation/activation-caching.

    ``config.tokenizer_name`` overrides ``config.model_name`` wholesale --
    i.e. whatever is in the tokenizer_name directory (vocab *and*
    chat_template) is used as-is, with no mixing-in of the checkpoint's own
    chat_template.jinja. Confirmed explicitly with the checkpoint owner
    (durech, 2026-07-13, re: the two local Apertus 1.5 checkpoints both
    needing ``tokenizers/apertus_emu3.5_wavtok_text_only`` instead of their
    own bundled tokenizer files): "tokenizer path is canonical for whatever
    is in it" -- i.e. do NOT patch the checkpoint's own default system
    prompt back in, even though apertus_emu3.5_wavtok_text_only's own
    chat_template renders an empty one where each checkpoint's own template
    would inject "You are Apertus 1.5 Omni, ...". An earlier version of this
    function did exactly that override; it was wrong and has been removed.
    """
    from transformers import AutoTokenizer

    tokenizer_name = config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# --- entropy / sampling (pure tensor ops, no model calls) --------------------

def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats), over the last dim, of softmax(logits) at temperature=1.

    ``logits`` must be the RAW model logits (before any temperature scaling or
    top-p filtering) -- see the module docstring. Computed in float32
    regardless of input dtype so bf16 logits don't lose precision in
    log_softmax. Returns a tensor shaped like ``logits`` minus its last dim.
    """
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def apply_temperature_and_top_p(
    raw_logits: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """Temperature-scale then top-p filter RAW logits, for SAMPLING only.

    Never use the result of this function for entropy -- see the module
    docstring. Filtered-out tokens are set to ``-inf`` so a softmax over the
    result naturally excludes them; this mirrors the standard nucleus-sampling
    definition (HF's ``TopPLogitsWarper``): keep the smallest prefix of
    tokens (sorted by probability, descending) whose cumulative probability is
    >= top_p, always keeping at least the top-1 token.
    """
    scaled = raw_logits.float() / temperature
    if top_p >= 1.0:
        return scaled

    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift right so the token that first crosses top_p is kept, and always
    # keep the top-1 token even if its own probability already exceeds top_p.
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return scaled.masked_fill(indices_to_remove, float("-inf"))


def sample_next_tokens(
    filtered_logits: torch.Tensor, generators: Sequence[torch.Generator]
) -> torch.Tensor:
    """Sample one token per row from softmax(filtered_logits), one CPU generator per row.

    Always runs on CPU (see module docstring's "Sampling determinism" section)
    regardless of what device ``filtered_logits`` lives on.
    """
    probs = torch.softmax(filtered_logits, dim=-1).to("cpu")
    batch_size = probs.shape[0]
    tokens = torch.empty(batch_size, dtype=torch.long)
    for i in range(batch_size):
        tokens[i] = torch.multinomial(probs[i], num_samples=1, generator=generators[i])[0]
    return tokens


def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids that account for left padding (padded positions get 1, unused)."""
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def _eos_token_ids(tokenizer, model) -> Set[int]:
    ids: Set[int] = set()
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and gen_cfg.eos_token_id is not None:
        value = gen_cfg.eos_token_id
        if isinstance(value, int):
            ids.add(value)
        else:
            ids.update(int(v) for v in value)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return ids


# --- the manual decode loop ---------------------------------------------------

@dataclass
class RolloutResult:
    prompt_id: str
    rollout_idx: int
    seed: int
    generated_token_ids: List[int]
    generated_text: str
    per_token_entropy: List[float]
    num_tokens: int
    stop_reason: str  # "eos" | "length"
    wall_time_seconds: float


def generate_batch(
    model,
    tokenizer,
    tasks: Sequence[Tuple[str, int, int, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[RolloutResult]:
    """Run one batch of (prompt_id, rollout_idx, seed, prompt_text) tasks to completion.

    Every row runs until it emits an eos token or hits ``max_new_tokens``,
    whichever comes first; the whole batch loop ends once every row is done.
    ``wall_time_seconds`` in the returned results is the elapsed time for the
    WHOLE batch (all rows in a batch are computed in lockstep on the same
    forward passes, so no row finished any faster than the batch as a whole).
    """
    start_time = time.time()

    prompt_texts = [build_generation_prompt(tokenizer, task[3]) for task in tasks]
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    batch_size = input_ids.shape[0]

    generators = [torch.Generator(device="cpu").manual_seed(task[2]) for task in tasks]

    eos_ids = _eos_token_ids(tokenizer, model)

    finished = [False] * batch_size
    stop_reason: List[Optional[str]] = [None] * batch_size
    generated_ids: List[List[int]] = [[] for _ in range(batch_size)]
    generated_entropy: List[List[float]] = [[] for _ in range(batch_size)]

    position_ids = _build_position_ids(attention_mask)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    for step in range(max_new_tokens):
        entropy = compute_entropy_from_logits(logits)
        filtered_logits = apply_temperature_and_top_p(logits, temperature, top_p)
        next_tokens = sample_next_tokens(filtered_logits, generators)

        for i in range(batch_size):
            if finished[i]:
                continue
            token_id = int(next_tokens[i].item())
            generated_ids[i].append(token_id)
            generated_entropy[i].append(float(entropy[i].item()))
            if token_id in eos_ids:
                finished[i] = True
                stop_reason[i] = "eos"

        if all(finished) or step == max_new_tokens - 1:
            break

        next_tokens = next_tokens.to(device)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)],
            dim=-1,
        )
        position_ids = position_ids[:, -1:] + 1
        with torch.no_grad():
            outputs = model(
                input_ids=next_tokens.view(-1, 1),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    for i in range(batch_size):
        if stop_reason[i] is None:
            stop_reason[i] = "length"

    wall_time = time.time() - start_time

    results = []
    for i, (prompt_id, rollout_idx, seed, _prompt_text) in enumerate(tasks):
        token_ids = generated_ids[i]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        results.append(
            RolloutResult(
                prompt_id=prompt_id,
                rollout_idx=rollout_idx,
                seed=seed,
                generated_token_ids=token_ids,
                generated_text=text,
                per_token_entropy=generated_entropy[i],
                num_tokens=len(token_ids),
                stop_reason=stop_reason[i],
                wall_time_seconds=wall_time,
            )
        )
    return results


# --- resumability (pure, no model/GPU needed) --------------------------------

def all_tasks_for_domain(prompts_df: pd.DataFrame, domain: str, n_rollouts: int) -> List[Task]:
    prompt_ids = prompts_df.loc[prompts_df["domain"] == domain, "prompt_id"].tolist()
    return [(pid, r) for pid in prompt_ids for r in range(n_rollouts)]


def compute_remaining_tasks(all_tasks: Sequence[Task], done: Set[Task]) -> List[Task]:
    return [t for t in all_tasks if t not in done]


def load_existing_shard(shard_path: Path) -> pd.DataFrame:
    if shard_path.exists():
        return pd.read_parquet(shard_path)
    return pd.DataFrame(columns=SHARD_COLUMNS)


def done_tasks_from_shard(df: pd.DataFrame) -> Set[Task]:
    if df.empty:
        return set()
    return {(str(pid), int(ridx)) for pid, ridx in zip(df["prompt_id"], df["rollout_idx"])}


def _partial_result_path(work_dir: Path, prompt_id: str, rollout_idx: int) -> Path:
    return work_dir / f"{prompt_id}__{rollout_idx:04d}.json"


def write_partial_result(work_dir: Path, result: RolloutResult) -> Path:
    """Write one completed rollout's result as its own JSON file (temp + atomic rename)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    final_path = _partial_result_path(work_dir, result.prompt_id, result.rollout_idx)
    fd, tmp_path = tempfile.mkstemp(dir=str(work_dir), prefix=".tmp_partial_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return final_path


def load_partial_results(work_dir: Path) -> Dict[Task, dict]:
    """Load all completed-rollout JSON files under ``work_dir``.

    Silently skips files that fail to parse (e.g. a leftover temp file from a
    crash mid-write, which never got renamed and so isn't picked up anyway --
    this also guards against any oddity that slips through) so a corrupted
    leftover never blocks the rest of the resume.
    """
    results: Dict[Task, dict] = {}
    if not work_dir.exists():
        return results
    for path in sorted(work_dir.glob("*.json")):
        if path.name.startswith(".tmp_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        results[(str(record["prompt_id"]), int(record["rollout_idx"]))] = record
    return results


def write_shard_atomic(df: pd.DataFrame, shard_path: Path) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(shard_path.parent), prefix=".tmp_shard_", suffix=".parquet"
    )
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, shard_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def consolidate_shard(shard_path: Path, work_dir: Path) -> pd.DataFrame:
    """Merge (existing shard rows) + (partial-result files not yet in the shard).

    Rewrites the shard only if there's something new to add (existing shard
    rows always win over a same-keyed partial, though in practice the two
    should never disagree since a partial is only ever written once for a
    given (prompt_id, rollout_idx)).
    """
    existing_df = load_existing_shard(shard_path)
    partials = load_partial_results(work_dir)

    existing_keys = done_tasks_from_shard(existing_df)
    new_rows = [record for key, record in partials.items() if key not in existing_keys]

    if not new_rows:
        return existing_df

    new_df = pd.DataFrame.from_records(new_rows, columns=SHARD_COLUMNS)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["prompt_id", "rollout_idx"], keep="last")
    write_shard_atomic(combined, shard_path)
    return combined


def _chunk(seq: Sequence[Task], size: int) -> Iterator[List[Task]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


# --- per-domain driver ---------------------------------------------------------

def run_domain(
    config: DatasetGenConfig,
    model,
    tokenizer,
    prompts_df: pd.DataFrame,
    domain: str,
    batch_size: int,
    device: torch.device,
    shard_idx: int = 0,
) -> Path:
    shard_path = paths.generations_shard_path(config, domain, shard_idx)
    work_dir = paths.work_root(config) / "generations" / domain

    all_tasks = all_tasks_for_domain(prompts_df, domain, config.n_rollouts_per_prompt)
    existing_df = load_existing_shard(shard_path)
    done = done_tasks_from_shard(existing_df) | set(load_partial_results(work_dir).keys())

    remaining = compute_remaining_tasks(all_tasks, done)
    print(
        f"[{domain}] {len(all_tasks)} total tasks, {len(done)} already done, "
        f"{len(remaining)} remaining"
    )

    prompt_text_by_id = dict(zip(prompts_df["prompt_id"], prompts_df["prompt_text"]))

    for batch_tasks in _chunk(remaining, batch_size):
        task_specs = [
            (pid, ridx, derive_seed(config.seed, pid, ridx), prompt_text_by_id[pid])
            for pid, ridx in batch_tasks
        ]
        results = generate_batch(
            model, tokenizer, task_specs, config.max_new_tokens, config.temperature,
            config.top_p, device,
        )
        for result in results:
            write_partial_result(work_dir, result)
        consolidate_shard(shard_path, work_dir)

    # Final consolidation covers the case where `remaining` was empty this run
    # but a previous run left partials that were never folded into the shard.
    consolidate_shard(shard_path, work_dir)
    return shard_path


# --- CLI entry point -----------------------------------------------------------

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="Path to a DatasetGenConfig YAML file (default: configs/dataset/degeneration-dataset-apertus-8b-instruct.yaml).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Rollouts per decode batch.")
    parser.add_argument(
        "--domains", nargs="*", default=None,
        help="Subset of domains to generate (default: all configured domains).",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=None,
        help="Debug: cap number of prompts per domain (for dry runs).",
    )
    parser.add_argument(
        "--max-rollouts", type=int, default=None,
        help="Debug: override n_rollouts_per_prompt (for dry runs).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Debug: override max_new_tokens (for dry runs).",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(_DTYPES))
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    config = DatasetGenConfig.from_yaml(args.config)
    if args.max_rollouts is not None:
        config.n_rollouts_per_prompt = args.max_rollouts
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens

    prompts_df = pd.read_parquet(paths.prompts_path(config))
    domains = args.domains or sorted(paths.configured_domain_names(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _DTYPES[args.dtype]

    tokenizer_name = config.tokenizer_name or config.model_name
    print(
        f"Loading tokenizer {tokenizer_name!r} + model {config.model_name!r} "
        f"(dtype={args.dtype}, device={device})..."
    )
    tokenizer = load_tokenizer_for_generation(config)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=dtype).to(device)
    model.eval()

    for domain in domains:
        domain_prompts = prompts_df
        if args.max_prompts is not None:
            keep_ids = set(
                prompts_df.loc[prompts_df["domain"] == domain, "prompt_id"].tolist()[: args.max_prompts]
            )
            domain_prompts = prompts_df[
                (prompts_df["domain"] != domain) | prompts_df["prompt_id"].isin(keep_ids)
            ]
        run_domain(config, model, tokenizer, domain_prompts, domain, args.batch_size, device)


if __name__ == "__main__":
    main()
