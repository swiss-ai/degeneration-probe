"""Recompute the generator's log-probability of every token it sampled.

Generation stored the entropy of each next-token distribution and the
probability of the most likely token, but not the probability of the token that
was actually drawn. The windowed negative log-probability baseline needs the
latter, so this script runs the generator once more over each rollout, with the
prompt built exactly as at generation time, and reads it off the raw logits
(before temperature and top-p, as for the stored entropy).

The recomputed entropy is compared with the stored one for every rollout, which
checks that the context matches the one the tokens were sampled under.

    python scripts/compute_token_logprobs.py \\
        --build-root <dataset build> --model-name <hf id> \\
        --splits val test_indomain test_heldout_domains --shard 0 --num-shards 4

Writes <build-root>/token_logprobs/shard_<i>_of_<n>.parquet with columns
prompt_id, rollout_idx, token_logprob (one float per generated token) and
entropy_max_abs_diff.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.dataset_gen.generate import build_generation_prompt  # noqa: E402


def load_rollouts(build_root: Path, splits, shard: int, num_shards: int) -> pd.DataFrame:
    onset = pd.read_parquet(build_root / "onset_labels" / "onset_labels.parquet")
    onset = onset[onset["split"].isin(splits)]
    prompts = pd.read_parquet(build_root / "prompts" / "prompts.parquet", columns=["prompt_id", "prompt_text"])
    frames = []
    for domain in sorted(onset["domain"].astype(str).unique()):
        keys = onset.loc[onset["domain"].astype(str) == domain, ["prompt_id", "rollout_idx"]]
        for path in sorted((build_root / "generations" / domain).glob("*.parquet")):
            part = pd.read_parquet(path, columns=["prompt_id", "rollout_idx", "generated_token_ids", "per_token_entropy"])
            frames.append(part.merge(keys, on=["prompt_id", "rollout_idx"], how="inner"))
    rollouts = pd.concat(frames, ignore_index=True).merge(prompts, on="prompt_id", how="left", validate="many_to_one")
    rollouts = rollouts.sort_values(["prompt_id", "rollout_idx"]).reset_index(drop=True)
    return rollouts.iloc[shard::num_shards].reset_index(drop=True)


@torch.no_grad()
def score(model, input_ids: torch.Tensor, prompt_length: int, completion: np.ndarray):
    """Log-probability of each completion token and the entropy before it, from raw logits."""
    logits = model(input_ids=input_ids.unsqueeze(0).to(model.device)).logits[0]
    # The token at completion position t is predicted from input position prompt_length + t - 1.
    predicting = logits[prompt_length - 1 : prompt_length - 1 + len(completion)].float()
    log_probs = torch.log_softmax(predicting, dim=-1)
    target = torch.as_tensor(completion, dtype=torch.long, device=log_probs.device)
    token_logprob = log_probs.gather(1, target[:, None])[:, 0]
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return token_logprob.cpu().numpy().astype(np.float32), entropy.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--splits", nargs="+", default=["val", "test_indomain", "test_heldout_domains"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    rollouts = load_rollouts(args.build_root, args.splits, args.shard, args.num_shards)
    print(f"{len(rollouts)} rollouts in shard {args.shard} of {args.num_shards}", flush=True)
    rows, start = [], time.time()
    for i, row in enumerate(rollouts.itertuples(index=False)):
        prompt_text = build_generation_prompt(tokenizer, row.prompt_text)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion = np.asarray(row.generated_token_ids, dtype=np.int64)
        input_ids = torch.as_tensor(list(prompt_ids) + completion.tolist(), dtype=torch.long)
        token_logprob, entropy = score(model, input_ids, len(prompt_ids), completion)
        stored = np.asarray(row.per_token_entropy, dtype=np.float64)[: len(entropy)]
        diff = float(np.nanmax(np.abs(stored - entropy[: len(stored)]))) if len(stored) else float("nan")
        rows.append({"prompt_id": row.prompt_id, "rollout_idx": int(row.rollout_idx),
                     "token_logprob": token_logprob, "entropy_max_abs_diff": diff})
        if (i + 1) % 200 == 0:
            recent = pd.Series([r["entropy_max_abs_diff"] for r in rows[-200:]])
            print(f"{i + 1}/{len(rollouts)}  {time.time() - start:.0f}s  entropy diff median "
                  f"{recent.median():.4f} max {recent.max():.4f}", flush=True)

    out = args.build_root / "token_logprobs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard_{args.shard}_of_{args.num_shards}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    diffs = pd.Series([r["entropy_max_abs_diff"] for r in rows])
    print(f"wrote {len(rows)} rollouts to {path}; entropy max-abs-diff median {diffs.median():.4f}, "
          f"95th pct {diffs.quantile(0.95):.4f}, max {diffs.max():.4f}")


if __name__ == "__main__":
    main()
