"""Write the baseline tables with every baseline at its setting chosen on validation.

The comparison table tab:baselines and its versions on the other test splits
(tab:probes-judge, tab:heldout, tab:heldout-human), the judge-labelled in-domain
split at the other two budgets (tab:budget-1pct, tab:budget-10pct), every scorer read from
q3_comparison_sweep.csv with every metric as a column: AUROC, the
detection rate, post-onset coverage, warning coverage, the early-warning rate,
the lead, the realised false-alarm rate and the pre-window coverage. Four sweep tables list the validation post-onset and
warning coverage at the 1% budget of every setting evaluated, the quantities
the selection rule reads.

    python scripts/q3_comparison.py --sweep
    python scripts/select_baseline_sweep.py
    python scripts/baseline_sweep_tables.py --out <dir>

With --human, tab:baselines, tab:heldout-human and the two budget tables are
read from q3_comparison_sweep_human.csv (``q3_comparison.py --sweep --human``),
the human-verified test splits; tab:probes-judge and tab:heldout keep judge
labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.q3_comparison as q3  # noqa: E402
import scripts.rebuttal_tables as rt  # noqa: E402
from scripts.score_baselines import parse_baseline  # noqa: E402
from scripts.select_baseline_sweep import POST_FLOOR, SELECTION_BUDGET  # noqa: E402

ANALYSIS = rt.ANALYSIS
SEQREP = "windowed seq-rep-n"
GROUPS = [(group, [SEQREP if n == "windowed seq-rep-2" else n for n in names]) for group, names in rt.SCORER_GROUPS
          if group != "hidden states"] + [("hidden states", ["frozen probe", "adapted probe, same layer",
                                                            "adapted probe, own layer"])]
NAMES = {**rt.SCORER_NAMES, SEQREP: r"windowed seq-rep-$n$"}


def comparison_rows(budget: float, split: str, human: bool = False) -> dict:
    """Per model, the comparison rows keyed by scorer, with the three reported probes: the frozen probe at its selected
    layer, the adapted probe at that layer, and the adapted probe at its own selected layer."""
    table = pd.read_csv(ANALYSIS / ("q3_comparison_sweep_human.csv" if human else "q3_comparison_sweep.csv"))
    table = table[(table["split"] == split) & np.isclose(table["budget"], budget)]
    rows = {}
    for model, probes in rt.PROBES.items():
        part = table[table["model"] == model]
        frozen_layer, own_layer = probes[0][1], probes[-1][1]
        picked = {}
        for _, row in part.iterrows():
            name = row["scorer"]
            if name.startswith("repeated-n-gram rule"):
                picked["repeated-n-gram rule"] = row
            elif name.startswith("frozen probe"):
                picked["frozen probe"] = row
            elif name == f"adapted probe (L{frozen_layer})":
                picked["adapted probe, same layer"] = row
            elif name == f"adapted probe (L{own_layer})":
                picked["adapted probe, own layer"] = row
            elif not name.startswith("adapted probe"):
                picked[name] = row
        rows[model] = picked
    return rows


def degenerate_counts(split: str, human: bool = False) -> dict:
    """The number of degenerate responses of each model on a split, among those the judge ruled on."""
    counts = {}
    for model, dataset in q3.MODELS.items():
        frame = q3.labels(dataset, human=human)
        counts[model] = int(frame.loc[frame["split"] == split, "is_positive"].sum())
    return counts


def unreviewed_models(split: str) -> set:
    """The models whose capped responses on a split are not all reviewed yet. Their block of a human-verified table
    is left as placeholders, since a partly reviewed split would mix human and judge labels."""
    pending = set()
    for model, dataset in q3.MODELS.items():
        frame = pd.read_parquet(q3.STORE / f"degeneration-dataset-{dataset}" / "onset_labels" / "onset_labels.parquet")
        capped = frame[(frame["split"] == split) & (frame["stop_reason"] != "eos")]
        reviewed = set(pd.read_csv(q3.HUMAN_LABELS / f"labels_{dataset}.csv")["id"])
        if not set(capped["prompt_id"] + "::" + capped["rollout_idx"].astype(str)) <= reviewed:
            pending.add(model)
    return pending


def pending_note(pending: set) -> str:
    if not pending:
        return ""
    names = " and ".join(rt.MODELS[model][1] for model in q3.MODELS if model in pending)
    return rf" \todo{{{names}: human review not complete yet.}}"


def shown_name(model: str, name: str) -> str:
    frozen_layer, own_layer = rt.PROBES[model][0][1], rt.PROBES[model][-1][1]
    return {"frozen probe": f"frozen probe, L{frozen_layer}",
            "adapted probe, same layer": f"adapted probe, L{frozen_layer}",
            "adapted probe, own layer": f"adapted probe, L{own_layer}"}.get(name, NAMES.get(name))


def lead_cell(row) -> str:
    lead, std = row["median_offset"], row.get("median_offset_std")
    if pd.isna(lead):
        return "--"
    if row["seeds"] < 2 or pd.isna(std):
        return f"${lead:+.0f}$"
    return f"${lead:+.0f}\\sd{{{std:.0f}}}$"


def auc_cell(row) -> str:
    mean, std = row["rollout_auc"], row.get("rollout_auc_std")
    if pd.isna(std) or row["seeds"] < 2:
        return f"${mean:.4f}$"
    return f"${mean:.4f}\\sd{{{std:.4f}}}$"


def fa_cell(row) -> str:
    """The realised false-alarm rate on healthy responses, in percent."""
    std = row.get("negative_fpr_std", np.nan)
    return rt.cell({**row, "negative_fpr": row["negative_fpr"] * 100, "negative_fpr_std": std * 100}, "negative_fpr", 1)


# Metrics of the main comparison, in column order: (column, how to print, ranked direction or None).
COLUMNS = [("rollout_auc", auc_cell, "max"),
           ("detection_rate", lambda r: rt.cell(r, "detection_rate", 2), None),
           ("in_pattern_recall", lambda r: rt.cell(r, "in_pattern_recall"), "max"),
           ("warning_recall_256", lambda r: rt.cell(r, "warning_recall_256"), "max"),
           ("fired_before_frontier", lambda r: rt.cell(r, "fired_before_frontier"), "max"),
           ("median_offset", lead_cell, "min"),
           ("negative_fpr", fa_cell, None),
           ("preonset_coverage_256", lambda r: rt.cell(r, "preonset_coverage_256"), None)]
# Ranking compares values at the precision they are printed with, so scores that print alike rank alike.
DIGITS = {"rollout_auc": 4, "detection_rate": 2, "median_offset": 0}
# Early and lead are read over alarmed degenerate responses only.
ALARMED_ONLY = {"fired_before_frontier", "median_offset"}


def comparison_table(budget: float, split: str, label: str, caption: str, placeholder: bool = False,
                     human: bool = False, pending: set = frozenset()) -> str:
    """Every scorer with every metric as a column, one block per model. With ``placeholder`` every value is a red
    question mark, for a split whose labels are not in yet; ``pending`` does the same for the named models only.
    With ``human`` the rows come from the human-verified split."""
    rows = comparison_rows(budget, split, human)
    counts = degenerate_counts(split, human)
    lines = []
    all_placeholder = placeholder
    for model, picked in rows.items():
        placeholder = all_placeholder or model in pending
        best = {}
        for metric, _, direction in COLUMNS:
            if direction is None:
                continue
            values = pd.Series({name: row[metric] for name, row in picked.items()}).dropna()
            values = values.round(DIGITS.get(metric, 3))
            if metric in ALARMED_ONLY:
                values = values[[picked[n]["detection_rate"] >= 0.9 for n in values.index]]
            top = values.idxmax() if direction == "max" else values.idxmin()
            spread = picked[top].get(f"{metric}_std", 0)
            best[metric] = (values[top], 0 if pd.isna(spread) else spread)
        if lines:
            lines.append(r"    \midrule")
        # The model heads its block instead of filling a column of its own, which keeps the table within the text width.
        n = "?" if placeholder else str(counts[model])
        lines.append(rf"    \multicolumn{{9}}{{@{{}}l}}{{{rt.MODELS[model][1]} ($n={n}$)}} \\")
        for g, (_, names) in enumerate(GROUPS):
            if g:
                lines.append(r"    \addlinespace[2pt]")
            for name in names:
                row = picked.get(name)
                if row is None:
                    continue
                cells = [rf"\quad {shown_name(model, name)}"]
                for metric, fmt, direction in COLUMNS:
                    if placeholder:
                        cells.append(rt.PLACEHOLDER)
                        continue
                    if pd.isna(row[metric]):
                        cells.append("--")
                        continue
                    text = fmt(row)
                    alarmed = row["detection_rate"] >= 0.9
                    if direction is not None and (alarmed or metric not in ALARMED_ONLY):
                        value, spread = best[metric]
                        own = row.get(f"{metric}_std", 0)
                        shown = round(row[metric], DIGITS.get(metric, 3))
                        if shown == value:
                            text = rf"$\mathbf{{{text.strip('$')}}}$"
                        elif abs(value - shown) <= max(spread, 0 if pd.isna(own) else own):
                            text = rf"\underline{{{text}}}"
                    if metric in ALARMED_ONLY and not alarmed:
                        text += r"$^\dagger$"
                    cells.append(text)
                lines.append("    " + " & ".join(cells) + r" \\")
    return rf"""\begin{{table}}[!htbp]
  \centering
  \caption{{{caption}}}
  \label{{{label}}}
  \footnotesize
  \setlength{{\tabcolsep}}{{3pt}}
  \resizebox{{\linewidth}}{{!}}{{%
  \begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
    \toprule
    & \shortstack{{Response\\classification}} & \multicolumn{{2}}{{c}}{{Failure detection}} & \multicolumn{{3}}{{c}}{{Early warning}} & & \\
    \cmidrule(lr){{2-2}}\cmidrule(lr){{3-4}}\cmidrule(lr){{5-7}}
    Scorer & AUROC $\uparrow$ & Detection $\uparrow$ & Post $\uparrow$ & $\mathrm{{Warn}}_{{256}}$ $\uparrow$ & Early $\uparrow$ & Lead $\downarrow$ & FA (\%) & Pre \\
    \midrule
{chr(10).join(lines)}
    \bottomrule
  \end{{tabular}}%
  }}
\end{{table}}
"""


def sweep_table(families: list[tuple[str, str]], params: list[tuple[str, str]], label: str, caption: str) -> str:
    """Validation post-onset and warning coverage at the selection budget of every setting of some baseline families,
    the selected setting of each model in bold and the settings under the post-onset floor in grey."""
    frame = pd.read_csv(ANALYSIS / "baseline_sweep_all.csv")
    frame = frame[(frame["split"] == "val") & np.isclose(frame["budget"], SELECTION_BUDGET)]
    chosen = pd.read_csv(ANALYSIS / "baseline_sweep_selected.csv")
    chosen = chosen[chosen["role"] == "selected"].drop_duplicates(["model", "family"])
    selected = {(r.model, r.family): r.setting for r in chosen.itertuples()}
    lines = []
    for f, (family, shown) in enumerate(families):
        part = frame[frame["family"] == family]
        settings = sorted(part["setting"].unique(), key=lambda s: tuple(parse_baseline(s)[1][k] for k, _ in params))
        if f:
            lines.append(r"    \midrule")
        for i, setting in enumerate(settings):
            values = parse_baseline(setting)[1]
            cells = ([shown if i == 0 else ""] if len(families) > 1 else []) + [str(values[k]) for k, _ in params]
            for model in rt.MODELS:
                row = part[(part["model"] == model) & (part["setting"] == setting)].iloc[0]
                for metric in ("in_pattern_recall", "warning_recall_256"):
                    text = f"{row[metric]:.3f}"
                    if selected[(model, family)] == setting:
                        text = rf"\textbf{{{text}}}"
                    if row["in_pattern_recall"] < POST_FLOOR:
                        text = rf"\textcolor{{gray}}{{{text}}}"
                    cells.append(text)
            lines.append("    " + " & ".join(cells) + r" \\")
    keys = ([r"Baseline"] if len(families) > 1 else []) + [name for _, name in params]
    lead = len(keys)
    models = " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, name in rt.MODELS.values())
    rules = "".join(rf"\cmidrule(lr){{{lead + 1 + 2 * i}-{lead + 2 + 2 * i}}}" for i in range(len(rt.MODELS)))
    metrics = " & ".join([r"Post & $\mathrm{Warn}_{256}$"] * len(rt.MODELS))
    return rf"""\begin{{table}}[!htbp]
  \centering
  \caption{{{caption}}}
  \label{{{label}}}
  \scriptsize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{{"l" * (len(families) > 1) + "r" * (len(params) + 2 * len(rt.MODELS))}}}
    \toprule
    {" & " * (lead - 1)} & {models} \\
    {rules}
    {" & ".join(keys)} & {metrics} \\
    \midrule
{chr(10).join(lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


COMPARISON_NOTE = (
    r"Scorers are grouped, from top to bottom, by what they read: nothing, the emitted tokens, the generator's "
    r"confidence in them, or the hidden states. Each baseline is read at the setting chosen on validation "
    r"(\Cref{app:baselines}). Probes are labelled with the layer they read. $n$: number of "
    r"degenerate responses. $^\dagger$: raises an alarm on fewer than $90\%$ of degenerate responses, so its Early "
    r"and Lead cover a minority of them and are left out of the comparison. FA and Pre are not ranked. Best value in each "
    r"other column and model in bold, values within one standard deviation of it underlined. Mean $\pm$ one standard deviation over "
    r"three seeds for probes; the baselines are deterministic, and random alarms are given in closed form.")

SWEEP_NOTE = (
    r"Validation split, $\beta=1\%$. Bold: the setting selected for that model (\Cref{app:baselines}). Grey: below "
    r"the post-onset floor of $0.3$. A coverage near $0$ "
    r"marks a setting on which healthy responses already reach the top of the score range, so the calibrated "
    r"threshold sits at that top and almost no token is flagged.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test_indomain")
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--human", action="store_true",
                        help="read tab:baselines and tab:heldout-human from the human-verified test splits")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pending_in = unreviewed_models(args.split) if args.human else set()
    pending_out = unreviewed_models("test_heldout_domains") if args.human else set()
    outputs = {
        "baselines.tex": comparison_table(
            args.budget, args.split, "tab:baselines",
            rf"Probes against the baselines on the {'human-verified ' if args.human else ''}in-domain test split at "
            rf"$\beta={args.budget * 100:.0f}\%$. " + COMPARISON_NOTE + pending_note(pending_in),
            human=args.human, pending=pending_in),
        "probes_judge.tex": comparison_table(
            args.budget, args.split, "tab:probes-judge", r"As \Cref{tab:baselines}, with judge labels."),
        "budget_1pct.tex": comparison_table(
            0.01, args.split, "tab:budget-1pct",
            (r"As \Cref{tab:baselines}" if args.human else r"As \Cref{tab:probes-judge}") + r", at $\beta=1\%$."
            + pending_note(pending_in), human=args.human, pending=pending_in),
        "budget_10pct.tex": comparison_table(
            0.10, args.split, "tab:budget-10pct",
            (r"As \Cref{tab:baselines}" if args.human else r"As \Cref{tab:probes-judge}") + r", at $\beta=10\%$."
            + pending_note(pending_in), human=args.human, pending=pending_in),
        "heldout.tex": comparison_table(
            args.budget, "test_heldout_domains", "tab:heldout",
            r"As \Cref{tab:baselines}, on the held-out sources (\texttt{codeforces} and \texttt{medical\_o1}), "
            r"with judge labels."),
        "heldout_human.tex": comparison_table(
            args.budget, "test_heldout_domains", "tab:heldout-human",
            r"As \Cref{tab:heldout}, with human-verified labels."
            + (pending_note(pending_out) if args.human else r" \todo{fill in once the held-out reviews are merged.}"),
            placeholder=not args.human, human=args.human, pending=pending_out),
        "sweep_confidence.tex": sweep_table(
            [("entropy", "entropy"), ("neglogp", r"$-\log p$")], [("w", "$S$")], "tab:sweep-confidence",
            r"Window $S$ of windowed entropy and windowed negative log-probability. On \textsc{Llama} no window "
            r"reaches the floor, and the window with the highest $\mathrm{Warn}_{256}$ is selected. " + SWEEP_NOTE),
        "sweep_ngram.tex": sweep_table(
            [("ngram", "")], [("n", "$n$")], "tab:sweep-ngram",
            r"$N$-gram size $n$ of the repeated-$n$-gram rule. " + SWEEP_NOTE),
        "sweep_seqrep.tex": sweep_table(
            [("repetition_trailing", "")], [("n", "$n$"), ("w", "$S$")], "tab:sweep-seqrep",
            r"$N$-gram order $n$ and window $S$ of windowed seq-rep-$n$. " + SWEEP_NOTE),
        "sweep_repl.tex": sweep_table(
            [("rep_l", "")], [("l", r"$\ell$"), ("w", "$S$")], "tab:sweep-repl",
            r"Look-back $\ell$ and window $S$ of rep$/\ell$. " + SWEEP_NOTE),
    }
    for name, text in outputs.items():
        (args.out / name).write_text(text)
        print(f"-> {args.out / name}")


if __name__ == "__main__":
    main()
