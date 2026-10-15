"""How the probes' flags depend on token position, with intervals over prompts.

Healthy responses are short and degenerate onsets come late, so position alone
carries part of the degeneration label. This script measures how much the
probes rely on it and corrects the comparisons that ignore it. Every number is
the mean over seeds of a pooled share (flagged tokens over tokens, early alarms
over alarms), and every interval is a 95% leave-one-prompt-out jackknife: all
rollouts of a prompt are removed together, since rollouts of one prompt behave
alike and the splits are drawn by prompt.

It writes five tables into --out-dir:

    onset_profile.csv     the share of degenerate tokens flagged by distance to
                          the onset, with two healthy references: healthy tokens
                          at the same absolute positions, and at the same
                          positions within the same source
    healthy_position.csv  the share of healthy tokens flagged, and their mean
                          score in logits, by absolute position, over all
                          healthy responses and split by source, by total
                          response length, and by local repetition of the text
    repetition.csv        the mean local repetition of healthy text by position
    position_only.csv     a scorer that reads nothing but the position,
                          calibrated to the same budget, next to the probes
    per_source.csv        AUROC and onset-aligned metrics per source

    python scripts/position_analysis.py --out-dir <dir> [--budget 0.05]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from degeneration_probe.evaluation.scores import read_scores  # noqa: E402

STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
OUTPUTS = STORE / "outputs"
RECIPE = "frontier_window W=512 centered K=256"
SPLIT = "test_indomain"
DATASETS = {"apertus": "apertus-8b-instruct", "llama3p1": "llama3p1-8b-instruct", "mistral": "mistral-7b-instruct-v0p1"}
# The frozen probe, the adapted probe at the frozen probe's layer, and the adapted probe at its own selected layer.
PROBES = {
    "apertus": [("frozen", 15), ("adapted", 15), ("adapted", 30)],
    "llama3p1": [("frozen", 15), ("adapted", 15), ("adapted", 6)],
    "mistral": [("frozen", 5), ("adapted", 5), ("adapted", 11)],
}
# Local repetition: the bigram repetition of the trailing window, the causal form of the labelling statistic.
REPETITION = "repetition_trailing"

WARNING = 256
BEFORE = [(lo, lo + 32) for lo in range(0, 1024, 32)] + [(1024, None)]
AFTER = [(lo, lo + 32) for lo in range(0, 256, 32)]
POSITION_BIN, POSITION_END = 128, 2048
# Healthy rate used as the reference for degenerate tokens, on a finer grid. A bin
# reached by fewer than MIN_RESPONSES healthy responses takes the pooled rate of
# every position from there on.
REFERENCE_BIN, MIN_RESPONSES = 64, 30
LENGTH_GROUPS = [(0, 512), (512, 1024), (1024, 2048), (2048, None)]
REPETITION_GROUPS = 4


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def jackknife(num: np.ndarray, den: np.ndarray, floor: float | None = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean over seeds of sum(num) / sum(den), with a 95% leave-one-prompt-out interval.

    `num` and `den` are seeds x prompts x bins. The interval is the normal one
    from the jackknife standard error, cut at `floor` (shares cannot go below zero).
    """
    total_num, total_den = num.sum(1), den.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        # Bins that no response reaches stay empty and come out as NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        point = np.nanmean(np.where(total_den > 0, total_num / total_den, np.nan), 0)
        rest_num, rest_den = total_num[:, None] - num, total_den[:, None] - den
        loo = np.nanmean(np.where(rest_den > 0, rest_num / rest_den, np.nan), 0)
        prompts = num.shape[1]
        se = np.sqrt((prompts - 1) / prompts * np.nansum((loo - np.nanmean(loo, 0)) ** 2, 0))
    low = point - 1.96 * se
    return point, low if floor is None else np.maximum(low, floor), point + 1.96 * se


def per_prompt(values: np.ndarray, prompt: np.ndarray, prompts: int) -> np.ndarray:
    """Sum responses x bins into prompts x bins."""
    out = np.zeros((prompts, values.shape[1]))
    np.add.at(out, prompt, values)
    return out


def depth_dirs(table: pd.DataFrame, model: str, regime: str, layer: int) -> list[Path]:
    rows = table[(table["model"] == model) & (table["layer"] == layer)
                 & (table["lora"].eq("none") if regime == "frozen" else table["lora"].ne("none"))]
    rows = rows.drop_duplicates("seed").sort_values("seed")
    return [OUTPUTS / r.run_name / r.attempt / "selected" / f"layer_{layer:02d}" for r in rows.itertuples()]


def threshold(depth: Path, budget: float) -> float:
    thresholds = json.loads((depth / "decision_thresholds.json").read_text())["thresholds"]
    return next(t["tau"] for t in thresholds if abs(t["target_negative_fpr"] - budget) < 1e-9)


def load(path: Path) -> pd.DataFrame:
    frame = read_scores(path, validate=False).sort_values(["prompt_id", "rollout_idx"]).reset_index(drop=True)
    frame["is_positive"] = frame["is_positive"].astype(bool)
    return frame


def bin_counts(flags: np.ndarray, values: np.ndarray, start: int, stop: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flags, value sums and tokens of positions [start, stop) in bins of `width` from 0."""
    bins = (stop - start) // width
    stop = min(stop, flags.size)
    index = np.arange(start, stop) // width - start // width
    return (np.bincount(index, weights=flags[start:stop], minlength=bins),
            np.bincount(index, weights=values[start:stop], minlength=bins),
            np.bincount(index, minlength=bins).astype(float))


class Model:
    """The test responses of one model, the per-prompt index, and the repetition baseline."""

    def __init__(self, model: str, reference: pd.DataFrame):
        self.model = model
        self.frame = reference
        self.prompt_ids, self.prompt = np.unique(reference["prompt_id"].to_numpy(), return_inverse=True)
        self.prompts = len(self.prompt_ids)
        self.healthy = ~reference["is_positive"].to_numpy()
        self.domain = reference["domain"].to_numpy()
        baseline = load(OUTPUTS / "baselines" / DATASETS[model] / REPETITION / "scores" / f"{SPLIT}.parquet")
        lookup = {(p, r): np.asarray(s, np.float64) for p, r, s in zip(baseline["prompt_id"], baseline["rollout_idx"], baseline["scores"])}
        self.repetition = [lookup.get((p, r)) for p, r in zip(reference["prompt_id"], reference["rollout_idx"])]
        pooled = np.concatenate([s for s, h in zip(self.repetition, self.healthy) if h and s is not None])
        self.repetition_edges = np.quantile(pooled, np.linspace(0, 1, REPETITION_GROUPS + 1)[1:-1])

    def check(self, frame: pd.DataFrame) -> None:
        same = frame[["prompt_id", "rollout_idx"]].to_numpy() == self.frame[["prompt_id", "rollout_idx"]].to_numpy()
        if not same.all():
            raise ValueError(f"{self.model}: seeds were scored on different responses")


def healthy_reference(scores: list[np.ndarray], healthy: np.ndarray, domain: np.ndarray, tau: float):
    """Healthy flag rate by position, overall and per source, on the reference grid."""
    bins = 4096 // REFERENCE_BIN + 1

    def rate(mask):
        hits, tokens, reach = np.zeros(bins), np.zeros(bins), np.zeros(bins)
        for s in (s for s, m in zip(scores, mask) if m):
            index = np.arange(s.size) // REFERENCE_BIN
            hits += np.bincount(index, weights=s >= tau, minlength=bins)
            tokens += np.bincount(index, minlength=bins)
            reach += np.bincount(index, minlength=bins) > 0
        reliable = np.nonzero(reach >= MIN_RESPONSES)[0]
        out = np.divide(hits, tokens, out=np.zeros(bins), where=tokens > 0)
        last = reliable.max() if reliable.size else -1
        out[last + 1:] = hits[last + 1:].sum() / max(tokens[last + 1:].sum(), 1)
        return out, last

    overall, _ = rate(healthy)
    by_source = {}
    for d in np.unique(domain):
        source_rate, last = rate(healthy & (domain == d))
        by_source[d] = np.where(np.arange(bins) <= last, source_rate, overall)
    return overall, by_source


def scorer_tables(model: Model, frames: list[pd.DataFrame], taus: list[float], name: str) -> dict[str, pd.DataFrame]:
    """Every table of one scorer, its seeds pooled."""
    seeds = len(frames)
    pos = ~model.healthy
    n_before, n_after = len(BEFORE), len(AFTER)
    n_position = POSITION_END // POSITION_BIN
    scopes = (["all"] + [f"source={d}" for d in np.unique(model.domain)]
              + [f"length={lo}-{hi or ''}" for lo, hi in LENGTH_GROUPS]
              + [f"repetition=q{k + 1}" for k in range(REPETITION_GROUPS)])
    onset = {k: np.zeros((seeds, len(frames[0]), n_before + n_after)) for k in ("hits", "tokens", "ref", "ref_source")}
    healthy = {k: np.zeros((seeds, len(scopes), len(frames[0]), n_position)) for k in ("hits", "logit", "tokens")}
    metrics = {k: np.zeros((seeds, len(frames[0]), 1)) for k in ("post_hits", "post_tokens", "warn_hits", "warn_tokens", "early", "alarmed")}
    leads, peaks = [], []
    lengths = np.array([s.size for s in frames[0]["scores"]])
    group = np.array([next(i for i, (lo, hi) in enumerate(LENGTH_GROUPS) if n >= lo and (hi is None or n < hi)) for n in lengths])
    sources = list(np.unique(model.domain))
    for seed, (frame, tau) in enumerate(zip(frames, taus)):
        scores = [np.asarray(s, np.float64) for s in frame["scores"]]
        overall_rate, source_rate = healthy_reference(scores, model.healthy, model.domain, tau)
        seed_leads, seed_peaks = [], np.array([s.max() if s.size else 0.0 for s in scores])
        for i, s in enumerate(scores):
            flags = (s >= tau).astype(float)
            if model.healthy[i]:
                values = logit(s)
                for scope in (0, 1 + sources.index(model.domain[i]), 1 + len(sources) + group[i]):
                    h, v, t = bin_counts(flags, values, 0, POSITION_END, POSITION_BIN)
                    healthy["hits"][seed, scope, i], healthy["logit"][seed, scope, i], healthy["tokens"][seed, scope, i] = h, v, t
                repetition = model.repetition[i]
                if repetition is not None and repetition.size == s.size:
                    quartile = np.searchsorted(model.repetition_edges, repetition)
                    for k in range(REPETITION_GROUPS):
                        mask = (quartile == k).astype(float)
                        scope = 1 + len(sources) + len(LENGTH_GROUPS) + k
                        h, v, _ = bin_counts(flags * mask, values * mask, 0, POSITION_END, POSITION_BIN)
                        t = bin_counts(mask, mask, 0, POSITION_END, POSITION_BIN)[0]
                        healthy["hits"][seed, scope, i], healthy["logit"][seed, scope, i], healthy["tokens"][seed, scope, i] = h, v, t
                continue
            onset_at = int(frame["onset_position"].iloc[i])
            positions = np.arange(s.size)
            ref = overall_rate[np.minimum(positions // REFERENCE_BIN, overall_rate.size - 1)]
            ref_source = source_rate[model.domain[i]][np.minimum(positions // REFERENCE_BIN, overall_rate.size - 1)]
            for k, (lo, hi) in enumerate(BEFORE):
                start, stop = (max(0, onset_at - hi), max(0, onset_at - lo)) if hi is not None else (0, max(0, onset_at - 1024))
                for key, values in (("hits", flags), ("tokens", np.ones_like(flags)), ("ref", ref), ("ref_source", ref_source)):
                    onset[key][seed, i, k] = values[start:stop].sum()
            for k, (lo, hi) in enumerate(AFTER):
                for key, values in (("hits", flags), ("tokens", np.ones_like(flags)), ("ref", ref), ("ref_source", ref_source)):
                    onset[key][seed, i, n_before + k] = values[onset_at + lo: onset_at + hi].sum()
            metrics["post_hits"][seed, i], metrics["post_tokens"][seed, i] = flags[onset_at:].sum(), s.size - onset_at
            window = flags[max(0, onset_at - WARNING): onset_at]
            metrics["warn_hits"][seed, i], metrics["warn_tokens"][seed, i] = window.sum(), window.size
            fired = np.flatnonzero(flags)
            if fired.size:
                metrics["alarmed"][seed, i] = 1
                metrics["early"][seed, i] = float(fired[0] < onset_at)
                seed_leads.append(fired[0] - onset_at)
        leads.append(np.median(seed_leads))
        peaks.append(seed_peaks)

    collapse = lambda a: np.stack([per_prompt(a[s], model.prompt, model.prompts) for s in range(seeds)])
    tables = {}
    # Onset profile with its two references.
    hits, tokens = collapse(onset["hits"]), collapse(onset["tokens"])
    point, low, high = jackknife(hits, tokens)
    ref, ref_low, ref_high = jackknife(collapse(onset["ref"]), tokens)
    ref_source, _, _ = jackknife(collapse(onset["ref_source"]), tokens)
    reached = (onset["tokens"][0] > 0)
    tables["onset_profile"] = pd.DataFrame({
        "side": ["before"] * n_before + ["after"] * n_after,
        "lo": [lo for lo, _ in BEFORE] + [lo for lo, _ in AFTER],
        "bin": [f"{lo}-{hi}" if hi else f">{lo}" for lo, hi in BEFORE] + [f"{lo}-{hi}" for lo, hi in AFTER],
        "flag_rate": point, "ci_low": low, "ci_high": high,
        "healthy_same_position": ref, "healthy_same_position_ci_low": ref_low, "healthy_same_position_ci_high": ref_high,
        "healthy_same_position_and_source": ref_source,
        "responses": reached.sum(0), "prompts": np.array([np.unique(model.prompt[reached[:, k]]).size for k in range(reached.shape[1])]),
    })
    # Healthy position profiles.
    rows = []
    for j, scope in enumerate(scopes):
        tok = collapse(healthy["tokens"][:, j])
        share = jackknife(collapse(healthy["hits"][:, j]), tok)
        mean_logit = jackknife(collapse(healthy["logit"][:, j]), tok, floor=None)
        reach = healthy["tokens"][0, j] > 0
        for k in range(n_position):
            rows.append({"scope": scope, "lo": k * POSITION_BIN, "flag_rate": share[0][k], "ci_low": share[1][k], "ci_high": share[2][k],
                         "mean_logit": mean_logit[0][k], "mean_logit_ci_low": mean_logit[1][k],
                         "mean_logit_ci_high": mean_logit[2][k], "tokens": int(tok[0, :, k].sum()),
                         "responses": int(reach[:, k].sum()), "prompts": int(np.unique(model.prompt[reach[:, k]]).size)})
    tables["healthy_position"] = pd.DataFrame(rows)
    # Onset-aligned metrics, overall and per source, with AUROC.
    rows = []
    for source in ["all"] + sources:
        keep = np.ones(len(model.domain), bool) if source == "all" else model.domain == source
        sub = lambda a: np.stack([per_prompt(a[s][keep], model.prompt[keep], model.prompts) for s in range(seeds)])
        row = {"source": source, "healthy": int((model.healthy & keep).sum()), "degenerate": int((pos & keep).sum()),
               "degenerate_prompts": int(np.unique(model.prompt[pos & keep]).size)}
        for key, (num, den) in {"post_onset": ("post_hits", "post_tokens"), "warning_256": ("warn_hits", "warn_tokens"),
                                "early": ("early", "alarmed")}.items():
            p, lo, hi = jackknife(sub(metrics[num]), sub(metrics[den]))
            row.update({key: p[0], f"{key}_ci_low": lo[0], f"{key}_ci_high": hi[0]})
        labels = pos[keep]
        aucs = [roc_auc_score(labels, peak[keep]) for peak in peaks]
        loo = []
        for prompt in np.unique(model.prompt[keep]):
            rest = model.prompt[keep] != prompt
            if labels[rest].any() and (~labels[rest]).any():
                loo.append(np.mean([roc_auc_score(labels[rest], peak[keep][rest]) for peak in peaks]))
        loo = np.asarray(loo)
        se = np.sqrt((loo.size - 1) / loo.size * ((loo - loo.mean()) ** 2).sum())
        row.update({"auroc": np.mean(aucs), "auroc_ci_low": np.mean(aucs) - 1.96 * se, "auroc_ci_high": min(1.0, np.mean(aucs) + 1.96 * se)})
        if source == "all":
            row["median_lead"] = float(np.mean(leads))
        rows.append(row)
    tables["per_source"] = pd.DataFrame(rows)
    for table in tables.values():
        table.insert(0, "scorer", name)
    return tables


def repetition_table(model: Model) -> pd.DataFrame:
    """Mean local repetition of healthy text by position, with its jackknife interval."""
    n_position = POSITION_END // POSITION_BIN
    values, tokens = np.zeros((len(model.frame), n_position)), np.zeros((len(model.frame), n_position))
    for i, repetition in enumerate(model.repetition):
        if model.healthy[i] and repetition is not None:
            _, values[i], tokens[i] = bin_counts(np.zeros_like(repetition), repetition, 0, POSITION_END, POSITION_BIN)
    point, low, high = jackknife(per_prompt(values, model.prompt, model.prompts)[None], per_prompt(tokens, model.prompt, model.prompts)[None])
    return pd.DataFrame({"model": model.model, "lo": np.arange(n_position) * POSITION_BIN, "repetition": point, "ci_low": low, "ci_high": high})


def position_only(model: Model, val: pd.DataFrame, budget: float) -> pd.DataFrame:
    """A scorer whose score is the position: alarm once a response passes the length that 1 - budget of healthy validation responses stay under."""
    lengths = np.array([len(s) for s in val.loc[~val["is_positive"], "scores"]])
    cut = int(np.min([c for c in range(4097) if np.mean(lengths > c) <= budget]))
    frame = model.frame
    pos = frame[frame["is_positive"]]
    n = np.array([len(s) for s in pos["scores"]])
    onset = pos["onset_position"].astype(int).to_numpy()
    flags_post = np.clip(n - np.maximum(onset, cut), 0, None)
    window_lo = np.maximum(0, onset - WARNING)
    warn_hits = np.clip(onset - np.maximum(window_lo, cut), 0, None)
    alarmed = n > cut
    prompt = model.prompt[frame["is_positive"].to_numpy()]
    col = lambda a: per_prompt(np.asarray(a, float)[:, None], prompt, model.prompts)[None]
    row = {"scorer": f"{model.model} position only", "cut": cut,
           "healthy_alarm_rate": float(np.mean([len(s) > cut for s in frame.loc[~frame["is_positive"], "scores"]]))}
    for key, (num, den) in {"post_onset": (flags_post, n - onset), "warning_256": (warn_hits, onset - window_lo),
                            "early": (alarmed & (cut < onset), alarmed)}.items():
        p, lo, hi = jackknife(col(num), col(den))
        row.update({key: p[0], f"{key}_ci_low": lo[0], f"{key}_ci_high": hi[0]})
    row["median_lead"] = float(np.median((cut - onset)[alarmed])) if alarmed.any() else np.nan
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.05)
    args = parser.parse_args()
    table = pd.read_csv(OUTPUTS / "analysis" / "selected_metrics.csv")
    table = table[(table["recipe"].isin([RECIPE, f"{RECIPE} +LoRA"])) & (table["split"] == SPLIT)]
    out = {k: [] for k in ("onset_profile", "healthy_position", "per_source", "repetition", "position_only")}
    for model_name, probes in PROBES.items():
        model = None
        for regime, layer in probes:
            dirs = depth_dirs(table, model_name, regime, layer)
            frames = [load(d / "scores" / f"{SPLIT}.parquet") for d in dirs]
            if model is None:
                model = Model(model_name, frames[0])
                out["repetition"].append(repetition_table(model))
                out["position_only"].append(position_only(model, load(dirs[0] / "scores" / "val.parquet"), args.budget))
            for frame in frames:
                model.check(frame)
            name = f"{model_name} {regime} L{layer}"
            print(f"{name}: {len(frames)} seeds", flush=True)
            for key, part in scorer_tables(model, frames, [threshold(d, args.budget) for d in dirs], name).items():
                out[key].append(part)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key, parts in out.items():
        pd.concat(parts, ignore_index=True).to_csv(args.out_dir / f"{key}.csv", index=False)
        print(f"-> {args.out_dir / key}.csv")


if __name__ == "__main__":
    main()
