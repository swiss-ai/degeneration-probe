import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
    import io
    import json
    import re
    import shutil
    import subprocess
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy.stats import kendalltau

    REPO = Path(__file__).resolve().parents[1]
    STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
    OUTPUTS = STORE / "outputs"
    ANALYSIS = OUTPUTS / "analysis"
    PAPER_DIR = REPO / "paper"
    TABLE_DIR = Path(__file__).resolve().parent / "tables" / "paper_tables"
    BUILD_DIR = TABLE_DIR / "build"
    TECTONIC = shutil.which("tectonic") or str(Path.home() / ".local" / "bin" / "tectonic")

    MODEL_LABEL = {"apertus": r"\textsc{Apertus}", "llama3p1": r"\textsc{Llama}", "mistral": r"\textsc{Mistral}"}
    MODEL_DATASET = {
        "apertus": "apertus-8b-instruct",
        "llama3p1": "llama3p1-8b-instruct",
        "mistral": "mistral-7b-instruct-v0p1",
    }
    # The generators the paper reports on; add "mistral" to put it in every
    # multi-model table.
    TABLE_MODELS = ("apertus", "llama3p1")
    LAYER = 15
    BUDGETS = (0.01, 0.05, 0.10)
    S4 = "frontier_window W=512 centered K=256"
    S4_LORA = S4 + " +LoRA"
    DEPTH_RECIPES = {
        "all_tokens K=1024": r"all\_tokens\_h1024",
        "rollout_balanced W=128 K=0": r"rollout\_balanced",
    }
    DEPTH_LAYERS = (4, 8, 12, 15, 20, 30)
    # Paper names of the positional-control scorers, in table order.
    CONTROL_SCORERS = {
        "best probe by AUC (random_window W=64)": "best probe by AUC",
        "selected recipe, adapted L15": "best probe, adapted",
        "token cap": "token cap",
        "worst probe by AUC (frontier W=1024 trailing K=128)": "worst probe by AUC",
    }
    # Paper columns of the baseline table and the scorer each one reads.
    BASELINES = {
        "lrs": "LRS",
        "rep_l": r"rep/$\ell$",
        "repetition_trailing": r"\shortstack[r]{windowed\\rep.}",
        "entropy": r"\shortstack[r]{windowed\\entropy}",
    }

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return (
        ANALYSIS,
        BASELINES,
        BUDGETS,
        BUILD_DIR,
        CONTROL_SCORERS,
        DEPTH_LAYERS,
        DEPTH_RECIPES,
        LAYER,
        MODEL_DATASET,
        MODEL_LABEL,
        OUTPUTS,
        PAPER_DIR,
        S4,
        S4_LORA,
        STORE,
        TABLE_DIR,
        TABLE_MODELS,
        TECTONIC,
        hashlib,
        io,
        json,
        kendalltau,
        mo,
        np,
        pd,
        re,
        shutil,
        subprocess,
    )


@app.cell
def _(np):
    def missing(x):
        return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))

    def fx(x, d):
        return "--" if missing(x) else f"{x:.{d}f}"

    def tiny(x, d):
        """`d` decimals, or one significant digit for a nonzero value that would print as zero."""
        if missing(x):
            return "--"
        if x != 0 and abs(x) < 0.5 * 10 ** (-d):
            return f"{x:.{int(-np.floor(np.log10(abs(x))))}f}"
        return f"{x:.{d}f}"

    def signed(x, d=0):
        return "--" if missing(x) else f"{x:+.{d}f}"

    def pm(mean, std, d, *, fmt=fx, style="pm", bold=False, std_fmt=None):
        """One mean with its seed spread, in the paper's `\\pm` or `\\sd` style."""
        if missing(mean):
            return "--"
        core = fmt(mean, d)
        if bold:
            core = rf"\textbf{{{core}}}"
        if missing(std):
            return f"${core}$"
        spread = (std_fmt or fx)(std, d)
        return rf"${core} \pm {spread}$" if style == "pm" else rf"${core}\sd{{{spread}}}$"

    def seeds(frame, columns):
        """Mean, standard deviation and count over the seeds in `frame`."""
        out = {}
        for column in columns:
            values = frame[column].astype(float).dropna()
            out[column] = values.mean() if len(values) else np.nan
            out[column + "_std"] = values.std() if len(values) > 1 else np.nan
        out["n"] = len(frame)
        return out

    def median_range(values):
        values = sorted(int(v) for v in values)
        if not values:
            return "--"
        med = float(np.median(values))
        core = f"${med:.0f}$" if med.is_integer() else f"${med:.1f}$"
        return core if values[0] == values[-1] else f"{core} ({values[0]}--{values[-1]})"

    return fx, median_range, pm, seeds, signed, tiny


@app.cell
def _(BUILD_DIR, PAPER_DIR, TABLE_DIR, TECTONIC, hashlib, re, subprocess):
    PREVIEW_PREAMBLE = r"""\documentclass{article}
    \usepackage[T1]{fontenc}
    \usepackage{times}
    \usepackage{amsmath,amsfonts,booktabs,multirow,graphicx,xcolor,float,nicefrac,xspace}
    \setlength{\textwidth}{5.5in}
    \usepackage[active,tightpage]{preview}
    \PreviewEnvironment{table}
    % Leave room on the right so a table wider than the text column shows it.
    \def\PreviewBbAdjust{-6pt -6pt 0.6in 6pt}
    \providecommand{\sd}[1]{{\scriptstyle\pm#1}}
    \newcommand{\method}{\textsc{Onward}\xspace}
    \pagestyle{empty}
    \begin{document}
    """.replace("\n    ", "\n")

    def compile_tex(source, stem, extra=(), directory=BUILD_DIR):
        """Compile `source` once per distinct content; return (pdf path or None, log)."""
        digest = hashlib.sha1(source.encode()).hexdigest()[:12]
        tex = directory / f"{stem}_{digest}.tex"
        pdf, log = tex.with_suffix(".pdf"), tex.with_suffix(".log.txt")
        if not pdf.is_file():
            tex.write_text(source)
            done = subprocess.run(
                [TECTONIC, "-X", "compile", *extra, tex.name],
                cwd=directory, capture_output=True, text=True, timeout=600,
            )
            log.write_text(done.stdout + done.stderr)
        return (pdf if pdf.is_file() else None), (log.read_text() if log.is_file() else "")

    def preview(table_latex, stem):
        return compile_tex(PREVIEW_PREAMBLE + table_latex + "\n\\end{document}\n", stem)

    def overfull(log):
        widths = [float(w) for w in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)]
        return max(widths) if widths else 0.0

    def paper_source():
        return (PAPER_DIR / "main.tex").read_text()

    def table_blocks(text):
        """Every table float of a LaTeX source, keyed by its label."""
        blocks = {}
        for match in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", text, flags=re.S):
            label = re.search(r"\\label\{([^}]*)\}", match.group(0))
            if label:
                blocks[label.group(1)] = match
        return blocks

    def register_table(label, latex, notes=()):
        """Write a regenerated table, compile it on its own, and keep what the combined PDF needs."""
        stem = label.replace(":", "_")
        (TABLE_DIR / f"{stem}.tex").write_text(latex + "\n")
        pdf, log = preview(latex, stem + "_new")
        notes = list(notes)
        wide = overfull(log)
        if wide > 1:
            notes.append(f"The table is {wide:.0f}pt wider than the 5.5in text column.")
        if pdf is not None:
            (TABLE_DIR / "pdf").mkdir(exist_ok=True)
            (TABLE_DIR / "pdf" / f"{stem}.pdf").write_bytes(pdf.read_bytes())
        else:
            notes.append("The table did not compile on its own; see its log in the build folder.")
        return {"label": label, "latex": latex, "notes": notes}

    return compile_tex, paper_source, register_table, table_blocks


@app.cell
def _(
    ANALYSIS,
    BASELINES,
    MODEL_DATASET,
    OUTPUTS,
    STORE,
    TABLE_MODELS,
    json,
    mo,
    np,
    pd,
):
    # Per (run, depth, split, budget) metrics of every scored probe.
    metrics = pd.read_csv(ANALYSIS / "selected_metrics.csv")
    metrics["seed"] = metrics["seed"].astype(int)
    metrics["regime"] = np.where(metrics["lora"].eq("none"), "frozen", "adapted")
    metrics["scorer_dir"] = [
        f"{OUTPUTS}/{run}/{attempt}/selected/layer_{int(layer):02d}"
        for run, attempt, layer in zip(metrics["run_name"], metrics["attempt"], metrics["layer"])
    ]

    # Best accuracy is not part of the aggregated evaluations; it comes from the
    # response-level pool.
    pool_path = ANALYSIS / "response_metrics_pool.csv"
    pool = pd.read_csv(pool_path) if pool_path.is_file() else pd.DataFrame(columns=["scorer_dir", "split", "best_accuracy", "auc", "rollouts", "positives"])
    metrics = metrics.merge(pool[["scorer_dir", "split", "best_accuracy"]], on=["scorer_dir", "split"], how="left")

    control = {
        split: pd.read_csv(ANALYSIS / f"positional_control_{split}.csv")
        for split in ("val", "test_indomain")
        if (ANALYSIS / f"positional_control_{split}.csv").is_file()
    }

    labels = {}
    for model in dict.fromkeys(("apertus",) + tuple(TABLE_MODELS)):
        path = STORE / f"degeneration-dataset-{MODEL_DATASET[model]}" / "onset_labels" / "onset_labels.parquet"
        if path.is_file():
            labels[model] = pd.read_parquet(path)

    def baseline_views(name, split, dataset="apertus-8b-instruct"):
        """The evaluation views of one heuristic baseline, or None before it is scored."""
        folder = OUTPUTS / "baselines" / dataset / name / "evaluation"
        if not (folder / split / "view_b_coverage.csv").is_file():
            return None
        b = pd.read_csv(folder / split / "view_b_coverage.csv")
        c = pd.read_csv(folder / split / "view_c_lead_time.csv")
        summary = json.loads((folder / split / "summary.json").read_text())
        tuned = json.loads((folder / "val" / "summary.json").read_text())
        saturated = {t["target_negative_fpr"]: bool(t.get("saturated")) for t in tuned["thresholds"]}
        frame = b.merge(c, on="target_negative_fpr", suffixes=("", "_c"))
        frame["saturated"] = frame["target_negative_fpr"].map(saturated)
        pool_row = pool[(pool["scorer_dir"] == str(OUTPUTS / "baselines" / dataset / name)) & (pool["split"] == split)]
        return {
            "views": frame,
            "auc": summary.get("rank_metrics", {}).get("rollout_auc"),
            "best_accuracy": float(pool_row["best_accuracy"].iloc[0]) if len(pool_row) else np.nan,
        }

    _missing = []
    if pool.empty:
        _missing.append("`response_metrics_pool.csv` (best accuracy): `sbatch cluster/response_metrics_pool.sbatch`")
    if baseline_views(next(iter(BASELINES)), "test_indomain") is None:
        _missing.append("heuristic baselines: `sbatch cluster/baselines.sbatch`, then the pool job again")
    for _split in ("val", "test_indomain"):
        if _split not in control:
            _missing.append(f"positional control on `{_split}`: `scripts/positional_control.py`")
    _models = metrics.groupby(["model", "regime"])["run_name"].nunique()
    _status = mo.md(
        f"Data: {metrics['run_name'].nunique()} scored runs in `selected_metrics.csv` "
        f"(written {pd.Timestamp((ANALYSIS / 'selected_metrics.csv').stat().st_mtime, unit='s'):%Y-%m-%d %H:%M}); "
        + ", ".join(f"{m} {r}: {n}" for (m, r), n in _models.items())
    )
    mo.vstack([_status] + [mo.callout(mo.md("Missing input: " + m), kind="warn") for m in _missing])
    return baseline_views, control, labels, metrics


@app.cell
def _(metrics):
    def probe_rows(model, recipe, layer, split, budget=0.01):
        """The seeds of one recipe read at one depth, on one split and budget."""
        return metrics[
            (metrics["model"] == model) & (metrics["recipe"] == recipe) & (metrics["layer"] == layer)
            & (metrics["split"] == split) & (metrics["target_negative_fpr"] == budget)
        ].sort_values("seed")

    return (probe_rows,)


@app.cell
def _(TABLE_MODELS, labels, register_table):
    def corpus_table():
        splits = [("train", "Train"), ("val", "Validation"), ("test_indomain", "Test, in-domain"), ("test_heldout_domains", "Test, held-out")]
        models = [m for m in TABLE_MODELS if m in labels]
        base = labels[models[0]]
        rows, data = [], []
        for split, name in splits:
            part = base[base["split"] == split]
            cells = [name, f"{part['prompt_id'].nunique():,}".replace(",", "{,}"), f"{len(part):,}".replace(",", "{,}")]
            record = {"split": split, "prompts": part["prompt_id"].nunique(), "responses": len(part)}
            for model in models:
                degenerate = int(labels[model].query("split == @split")["is_positive"].sum())
                cells.append(f"{degenerate:,}".replace(",", "{,}"))
                record[model] = degenerate
            rows.append("        " + " & ".join(cells) + r" \\")
            data.append(record)
        notes = []
        for model in models[1:]:
            if labels[model].groupby("split")["prompt_id"].nunique().to_dict() != base.groupby("split")["prompt_id"].nunique().to_dict():
                notes.append(f"Prompt counts of `{model}` differ from `{models[0]}`; the table prints the latter.")
        heads = " & ".join({"apertus": r"\textsc{Apertus}", "llama3p1": r"\textsc{Llama}", "mistral": r"\textsc{Mistral}"}[m] for m in models)
        n = len(models)
        latex = rf"""\begin{{table}}[H]
        \centering
        \caption{{Corpus composition and number of responses labelled as degenerate with a localisable onset. Splits are constructed at the prompt level. \texttt{{codeforces}} and \texttt{{medical\_o1}} appear only in the held-out
        test split.}}
        \label{{tab:corpus-splits}}
        \begin{{tabular}}{{lrr|{'r' * n}}}
            \toprule
            & & & \multicolumn{{{n}}}{{c}}{{Degenerate responses}} \\
            \cmidrule(lr){{4-{3 + n}}}
            Split & Prompts & Responses & {heads} \\
            \midrule
    {chr(10).join(rows)}
            \bottomrule
        \end{{tabular}}
    \end{{table}}"""
        return register_table("tab:corpus-splits", latex, notes)

    table_corpus = corpus_table() if labels else None
    return (table_corpus,)


@app.cell
def _(metrics, pd, register_table):
    def shortlist_table():
        # Name, selection, W, anchor, target, and the recipe that implements it.
        shortlist = [
            (r"all\_tokens", r"all\_tokens", "128", "--", r"hard, $K{=}0$", "all_tokens K=0"),
            (r"rollout\_balanced", r"rollout\_balanced", "128", "--", r"hard, $K{=}0$", "rollout_balanced W=128 K=0"),
            (r"random\_window", r"random\_window", "64", "--", r"hard, $K{=}0$", "random_window W=64 K=0"),
            (r"frontier\_window", r"frontier\_window", "256", "centered", r"hard, $K{=}0$", "frontier_window W=256 centered K=0"),
            (r"frontier\_window+hn", r"frontier\_window\_hard\_negative", "256", "centered", r"hard, $K{=}0$", None),
            (r"all\_tokens\_h1024", r"all\_tokens", "128", "--", r"hard, $K{=}1024$", "all_tokens K=1024"),
            (r"frontier512\_h256", r"frontier\_window", "512", "centered", r"hard, $K{=}256$", "frontier_window W=512 centered K=256"),
            (r"frontier512\_trailing\_h512", r"frontier\_window", "512", "trailing", r"hard, $K{=}512$", "frontier_window W=512 trailing K=512"),
            (r"soft\_exp256", r"frontier\_window", "512", "centered", r"soft exp., $\lambda{=}256$", "frontier_window W=512 centered soft exponential256"),
            (r"soft\_lin256", r"frontier\_window", "512", "centered", r"soft linear, $\lambda{=}256$", "frontier_window W=512 centered soft linear256"),
            (r"repetition\_mse", r"frontier\_window", "128", "centered", "repetition, MSE", None),
        ]
        present = set(metrics["recipe"])
        kept = [row for row in shortlist if row[5] in present]
        dropped = [row[0].replace("\\", "") for row in shortlist if row[5] not in present]
        body = "\n\n".join(
            rf"""\texttt{{{name}}}
    & \texttt{{{selection}}}
    & {w} & {anchor} & {target} \\""" for name, selection, w, anchor, target, _ in kept
        )
        latex = rf"""\begin{{table}}[H]
    \centering
    \caption{{
    Training recipes used throughout the paper. $W$ denotes the selection-window
    width or token budget, depending on the selection rule. For onset-aligned
    selection rules, \emph{{anchor}} specifies whether the window is centered on the
    degeneration onset or ends immediately before it. Unless otherwise specified,
    probes are trained with binary cross-entropy.
    }}
    \label{{tab:strategy-shortlist}}
    \begin{{tabular}}{{lllll}}
    \toprule
    Name & Selection & $W$ & Anchor & Target \\
    \midrule
    {body}
    \bottomrule
    \end{{tabular}}
    \end{{table}}"""
        notes = [f"No run of the new sweep implements {', '.join(f'`{d}`' for d in dropped)}; those rows are left out."] if dropped else []
        data = pd.DataFrame([{"name": r[0].replace("\\", ""), "recipe": r[5], "in runs": r[5] in present} for r in shortlist])
        return register_table("tab:strategy-shortlist", latex, notes)

    table_shortlist = shortlist_table()
    return (table_shortlist,)


@app.cell
def _(
    CONTROL_SCORERS,
    control,
    fx,
    labels,
    pd,
    pm,
    register_table,
    seeds,
    signed,
):
    def four_scorers_table():
        frame = control["test_indomain"]
        frame = frame[(frame["transform"] == "as scored") & (frame["persistence"] == 1)]
        columns = ["rollout_auc", "best_accuracy", "in_pattern_recall", "warning_recall_256", "fired_before_frontier", "median_offset", "never_fired_positives"]
        stats = pd.DataFrame([{"scorer": name, **seeds(frame[frame["scorer"] == key], columns)} for key, name in CONTROL_SCORERS.items()])
        higher = {"rollout_auc": True, "best_accuracy": True, "in_pattern_recall": True, "warning_recall_256": True, "fired_before_frontier": True, "median_offset": False}
        for column, up in higher.items():
            stats[column + "_rank"] = stats[column].rank(ascending=not up, method="min").astype("Int64")
        digits = {"rollout_auc": 4, "best_accuracy": 4, "in_pattern_recall": 2, "warning_recall_256": 3, "fired_before_frontier": 2}
        rows = []
        for _, r in stats.iterrows():
            cells = [f"{r['scorer']:<19}"]
            for column, d in digits.items():
                cells.append(f"{pm(r[column], r[column + '_std'], d, style='sd')} ({r[column + '_rank']})")
            lead = pm(r["median_offset"], r["median_offset_std"], 0, fmt=signed, style="sd", std_fmt=fx)
            cells.append(f"{lead} ({r['median_offset_rank']})")
            rows.append("    " + " & ".join(cells) + r" \\")
        degenerate = int(labels["apertus"].query("split == 'test_indomain'")["is_positive"].sum())
        misses = [(r["scorer"], r["never_fired_positives"]) for _, r in stats.iterrows() if r["never_fired_positives"] > 0]
        if misses:
            misses.sort(key=lambda x: -x[1])
            count = [f"${n:.0f}$" if float(n).is_integer() else f"${n:.1f}$" for _, n in misses]
            who = [name.replace("best probe, adapted", "adapted probe") for name, _ in misses]
            listed = f"the {who[0]} never alarms on {count[0]}" + "".join(f" and the {w} on {c}" for w, c in zip(who[1:], count[1:]))
            sentence = (f"Lead is a median over responses that alarmed at all: {listed} of the ${degenerate}$ "
                        "degenerate responses, the others never miss.")
        else:
            sentence = "Every scorer alarms on every degenerate response."
        latex = rf"""\begin{{table}}[H]
      \caption{{
        The four scorers of Figure~\ref{{fig:metrics}}(a,b) on the in-domain test
        split, with the rank each metric assigns in parentheses. Probes are read at
        layer~15 of \textsc{{Apertus}}, mean over three seeds; the token-cap baseline
        is deterministic. \emph{{Post}}: post-onset coverage; \emph{{Early}}: fraction
        of alerted degenerate responses alarming before onset; \emph{{Lead}}: median
        alarm offset from onset in tokens (lower is better). {sentence}
        $\beta=1\%$, $m=1$.
      }}
     \label{{tab:four-scorers}}
      \centering
      \footnotesize
      \setlength{{\tabcolsep}}{{3.5pt}}
      \begin{{tabular}}{{lrrrrrr}}
        \toprule
        Scorer & AUC & Accuracy & Post & Warn$_{{256}}$ & Early & Lead \\
        \midrule
    {chr(10).join(rows)}
        \bottomrule
      \end{{tabular}}
    \end{{table}}"""
        return register_table("tab:four-scorers", latex)

    table_four_scorers = four_scorers_table() if "test_indomain" in control else None
    return (table_four_scorers,)


@app.cell
def _(CONTROL_SCORERS, control, fx, register_table, tiny):
    def positional_table():
        frame = control["val"]
        frame = frame[frame["persistence"] == 1]
        columns = ["rollout_auc", "best_accuracy", "in_pattern_recall", "warning_recall_256", "fired_before_frontier"]
        blocks, data = [], []
        for key, name in CONTROL_SCORERS.items():
            lines = []
            for i, transform in enumerate(("as scored", "shuffled", "argmax only")):
                part = frame[(frame["scorer"] == key) & (frame["transform"] == transform)]
                means = part[columns].mean()
                data.append({"scorer": name, "scores": transform, **means.to_dict()})
                first = rf"\multirow{{3}}{{*}}{{{name}}}" if i == 0 else ""
                cells = [fx(means["rollout_auc"], 4), fx(means["best_accuracy"], 4), tiny(means["in_pattern_recall"], 3),
                         tiny(means["warning_recall_256"], 3), tiny(means["fired_before_frontier"], 3)]
                lines.append((f"    {first}\n      & {transform:<11} & " if i == 0 else f"      & {transform:<11} & ") + " & ".join(cells) + r" \\")
            blocks.append("\n".join(lines))
        body = (chr(10) + "    " + r"\midrule" + chr(10)).join(blocks)
        latex = rf"""\begin{{table}}[H]
      \caption{{
        The positional control. Both transforms preserve every response's maximum,
        so the response score, its ranking, and the calibrated threshold are
        unchanged by construction. Validation split, $\beta=1\%$, $m=1$; three
        seeds averaged except for the deterministic token-cap baseline.
      }}
      \label{{tab:positional-control}}
      \centering
      \small
      \begin{{tabular}}{{llccccc}}
        \toprule
        Scorer & Scores & AUC & Accuracy & $\mathrm{{Post}}$ & $\mathrm{{Warn}}_{{256}}$ & Early \\
        \midrule
    {body}
        \bottomrule
      \end{{tabular}}
    \end{{table}}"""
        return register_table("tab:positional-control", latex)

    table_positional = positional_table() if "val" in control else None
    return (table_positional,)


@app.cell
def _(CONTROL_SCORERS, control, fx, pm, register_table, seeds, tiny):
    def persistence_table():
        frame = control["val"]
        blocks, data, notes = [], [], []
        for key, name in CONTROL_SCORERS.items():
            lines = []
            for i, transform in enumerate(("as scored", "shuffled", "argmax only")):
                part = frame[(frame["scorer"] == key) & (frame["transform"] == transform)]
                aucs, warns = [], []
                for m in (1, 4, 8):
                    s = seeds(part[part["persistence"] == m], ["rollout_auc", "warning_recall_256"])
                    data.append({"scorer": name, "scores": transform, "m": m, **s})
                    # A spread of exactly zero is printed without one, as for a scorer that never fires.
                    auc_std = s["rollout_auc_std"] if s["rollout_auc_std"] else None
                    aucs.append(pm(s["rollout_auc"], auc_std, 4, style="sd"))
                    warns.append(pm(s["warning_recall_256"], s["warning_recall_256_std"], 3 if s["warning_recall_256"] >= 5e-4 or s["warning_recall_256"] == 0 else 5, fmt=tiny, style="sd", std_fmt=tiny))
                    if m > 1 and transform == "argmax only" and fx(s["rollout_auc"], 4) != "0.5000":
                        notes.append(f"{s['rollout_auc']:.4f}")
                first = rf"\multirow{{3}}{{*}}{{{name}}}" if i == 0 else ""
                lead = f"    {first}\n      & {transform:<11} & " if i == 0 else f"      & {transform:<11} & "
                lines.append(lead + " & ".join(aucs + warns) + r" \\")
            blocks.append("\n".join(lines))
        body = (chr(10) + "    " + r"\midrule" + chr(10)).join(blocks)
        latex = rf"""\begin{{table}}[H]
      \caption{{
        The positional control at three persistence settings. A scorer fires only
        where $m$ consecutive tokens clear the threshold; each row is recalibrated
        at each $m$ so that all entries spend $\beta=1\%$. Three seeds, one
        standard deviation, except for the deterministic token-cap baseline. A
        spread printed as $0.0000$ is below $5\times10^{{-5}}$. An AUC of exactly
        $0.5000$ marks a scorer that never fires. Validation split.
      }}
      \label{{tab:persistence-control}}
      \centering
      \scriptsize
      \setlength{{\tabcolsep}}{{3pt}}
      \begin{{tabular}}{{ll ccc ccc}}
        \toprule
        & & \multicolumn{{3}}{{c}}{{AUC}} & \multicolumn{{3}}{{c}}{{$\mathrm{{Warn}}_{{256}}$}} \\
        \cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}
        Scorer & Scores & $m{{=}}1$ & $m{{=}}4$ & $m{{=}}8$ & $m{{=}}1$ & $m{{=}}4$ & $m{{=}}8$ \\
        \midrule
    {body}
        \bottomrule
      \end{{tabular}}
    \end{{table}}"""
        if notes:
            notes = [f"The argmax-only rows at $m>1$ have AUC {', '.join(sorted(set(notes)))}, "
                     "not exactly 0.5000 as the caption says."]
        return register_table("tab:persistence-control", latex, notes)

    table_persistence = persistence_table() if "val" in control else None
    return (table_persistence,)


@app.cell
def _(metrics, register_table):
    def metric_room_table():
        # Every trained Apertus scoring pass on validation, frozen and adapted,
        # selectable or not, at the 1% budget.
        pool = metrics[(metrics["model"] == "apertus") & (metrics["split"] == "val") & (metrics["target_negative_fpr"] == 0.01)]
        n = len(pool)
        rows = [
            ("AUC", "rollout_auc"), ("Best accuracy", "best_accuracy"), (r"Accuracy at $\beta=1\%$", "accuracy"),
            None, (r"$\mathrm{Warn}_{256}$", "warning_recall_256"), (r"$\mathrm{Post}$", "in_pattern_recall"),
        ]
        lines, data = [], []
        for row in rows:
            if row is None:
                lines.append(r"    \midrule")
                continue
            name, column = row
            values = pool[column].dropna()
            if values.empty:
                lines.append(f"    {name:<23} & -- & -- \\\\")
                continue
            distinct = values.round(4).nunique()
            data.append({"metric": column, "min": values.min(), "max": values.max(), "distinct": distinct, "count": len(values)})
            lines.append(f"    {name:<23} & {values.min():.4f} -- {values.max():.4f} & {distinct:>3} / {len(values)} \\\\")
        latex = rf"""\begin{{table}}[H]
      \caption{{
        Resolution of each metric over the ${n}$ trained scoring passes of the
        validation pool.
      }}
      \label{{tab:metric-room}}
      \centering
      \small
      \begin{{tabular}}{{lcc}}
        \toprule
        Metric & Range & Distinct values \\
        \midrule
    {chr(10).join(lines)}
        \bottomrule
      \end{{tabular}}
    \end{{table}}"""
        notes = ["Distinct values are counted at four decimals, the precision the range is printed at."]
        return register_table("tab:metric-room", latex, notes)

    table_metric_room = metric_room_table()
    return (table_metric_room,)


@app.cell
def _(kendalltau, metrics, np, register_table, signed):
    def rank_transfer_table():
        # One candidate per Apertus recipe, read at the depth with the highest
        # seed-mean validation Warn_256 among selectable depths.
        columns = ["rollout_auc", "best_accuracy", "warning_recall_256", "in_pattern_recall"]
        at_one = metrics[(metrics["model"] == "apertus") & (metrics["target_negative_fpr"] == 0.01) & metrics["selectable"]]
        means = at_one.groupby(["recipe", "layer", "split"])[columns].mean().reset_index()
        val = means[means["split"] == "val"]
        chosen = val.sort_values("warning_recall_256", ascending=False).groupby("recipe").head(1)[["recipe", "layer"]]
        both = chosen.merge(means, on=["recipe", "layer"]).pivot_table(index=["recipe", "layer"], columns="split", values=columns)
        both = both.dropna(subset=[("warning_recall_256", "val"), ("warning_recall_256", "test_indomain")])
        n = len(both)

        def tau(a, b):
            if a not in both or b not in both:
                return np.nan
            pair = both[[a, b]].dropna()
            return kendalltau(pair[a], pair[b]).statistic if len(pair) > 2 else np.nan

        rows = [("AUC", "rollout_auc"), ("Best accuracy", "best_accuracy"), None,
                (r"$\mathrm{Warn}_{256}$", "warning_recall_256"), (r"$\mathrm{Post}$", "in_pattern_recall")]
        results = []
        for row in rows:
            if row is None:
                continue
            name, column = row
            results.append({
                "metric": name, "column": column,
                "self": tau((column, "val"), (column, "test_indomain")),
                "warn": tau((column, "val"), ("warning_recall_256", "test_indomain")),
            })
        best = np.nanmax([r["warn"] for r in results])
        lines = []
        for row in rows:
            if row is None:
                lines.append(r"    \midrule")
                continue
            r = next(x for x in results if x["column"] == row[1])
            warn = signed(r["warn"], 2)
            warn = rf"$\mathbf{{{warn}}}$" if r["warn"] == best else f"${warn}$"
            lines.append(f"    {r['metric']:<21} & ${signed(r['self'], 2)}$ & {warn} \\\\")
        latex = rf"""\begin{{table}}[H]
      \caption{{
        Rank transfer from validation to test, Kendall's $\tau$ over $n={n}$
        candidates at $\beta=1\%$ and $m=1$. Left: whether a metric reproduces its
        own ordering across the split. Right: whether its validation ordering
        predicts the test ordering by $\mathrm{{Warn}}_{{256}}$.
      }}
      \label{{tab:rank-transfer}}
      \centering
      \small
      \begin{{tabular}}{{lcc}}
        \toprule
        Ranking on validation & Reproduces itself on test & Predicts test $\mathrm{{Warn}}_{{256}}$ \\
        \midrule
    {chr(10).join(lines)}
        \bottomrule
      \end{{tabular}}
    \end{{table}}"""
        candidates = both.copy()
        candidates.columns = [f"{c}_{s}" for c, s in candidates.columns]
        notes = [f"Candidates: each Apertus recipe at its validation-chosen depth (best seed-mean Warn$_{{256}}$ among selectable depths), {n} in all."]
        return register_table("tab:rank-transfer", latex, notes)

    table_rank_transfer = rank_transfer_table()
    return (table_rank_transfer,)


@app.cell
def _(
    BUDGETS,
    LAYER,
    MODEL_LABEL,
    S4,
    S4_LORA,
    TABLE_MODELS,
    fx,
    pm,
    probe_rows,
    register_table,
    seeds,
    signed,
):
    def budgeted_table():
        columns = ["in_pattern_recall", "warning_recall_256", "median_offset", "fired_before_frontier"]
        blocks, data, notes = [], [], []
        for budget in BUDGETS:
            lines = []
            for i, model in enumerate(TABLE_MODELS):
                for j, (regime, recipe) in enumerate((("Frozen", S4), ("Adapted", S4_LORA))):
                    part = probe_rows(model, recipe, LAYER, "test_indomain", budget)
                    s = seeds(part, columns)
                    data.append({"budget": budget, "model": model, "regime": regime, **s})
                    if s["n"] < 3 and budget == BUDGETS[0]:
                        notes.append(f"{model} {regime.lower()} at layer {LAYER}: {s['n']} seed(s) scored.")
                    head = f"{budget * 100:.0f}\\%" if i == 0 and j == 0 else ""
                    model_cell = MODEL_LABEL[model] if j == 0 else ""
                    lines.append(
                        (f"    {head}\n" if head else "")
                        + f"      & {model_cell} & {regime}\n"
                        f"      & {pm(s['in_pattern_recall'], s['in_pattern_recall_std'], 3)}\n"
                        f"      & {pm(s['warning_recall_256'], s['warning_recall_256_std'], 3)}\n"
                        f"      & {pm(s['median_offset'], s['median_offset_std'], 0, fmt=signed, std_fmt=fx)}\n"
                        f"      & {pm(s['fired_before_frontier'], s['fired_before_frontier_std'], 3)} \\\\"
                    )
            blocks.append("\n".join(lines))
        body = (chr(10) + "    " + r"\midrule" + chr(10)).join(blocks)
        latex = rf"""\begin{{table}}[H]
      \caption{{
        Detection and early-warning performance of the selected layer-{LAYER} probes
        on the in-domain test set across false-alarm budgets $\beta$.
        Mean $\pm$ one standard deviation over three seeds.
      }}
      \label{{tab:budgeted}}
      \centering
      \small
      \begin{{tabular}}{{l|l|l|rrrr}}
        \toprule
        Budget & Model & Regime & Post & Warn$_{{256}}$ & Lead & Early \\
        \midrule
    {body}
        \bottomrule
      \end{{tabular}}
    \end{{table}}"""
        return register_table("tab:budgeted", latex, notes)

    table_budgeted = budgeted_table()
    return (table_budgeted,)


@app.cell
def _(
    DEPTH_LAYERS,
    DEPTH_RECIPES,
    np,
    pd,
    pm,
    probe_rows,
    register_table,
    seeds,
):
    def layers_table():
        columns = [("rollout_auc", 4), ("best_accuracy", 4), ("in_pattern_recall", 3), ("warning_recall_128", 4), ("warning_recall_256", 4)]
        blocks, data, notes = [], [], []
        for recipe, name in DEPTH_RECIPES.items():
            stats = []
            for layer in DEPTH_LAYERS:
                part = probe_rows("apertus", recipe, layer, "val")
                stats.append({"layer": layer, "selectable": int(part["selectable"].sum()), **seeds(part, [c for c, _ in columns])})
            frame = pd.DataFrame(stats)
            data.append(frame.assign(recipe=recipe))
            lines = [rf"    \multicolumn{{6}}{{l}}{{\textbf{{{name}}} (frozen)}} \\"]
            for _, r in frame.iterrows():
                cells = [f"{int(r['layer']):>2}"]
                for column, d in columns:
                    top = frame[column].round(d).max()
                    bold = not np.isnan(r[column]) and round(r[column], d) == top
                    cells.append(pm(r[column], r[column + "_std"], d, bold=bold))
                lines.append("    " + " & ".join(cells) + r" \\")
                if r["selectable"] < r["n"]:
                    notes.append(f"{recipe}, layer {int(r['layer'])}: {int(r['n'] - r['selectable'])} of {int(r['n'])} seeds never clear the post-onset floor (fallback step).")
            blocks.append("\n".join(lines))
        body = (chr(10) + "    " + r"\midrule" + chr(10) + chr(10)).join(blocks)
        latex = rf"""\begin{{table}}[H]
      \caption{{
        Layer-wise validation performance for two representative frozen probe
        training strategies on Apertus, at $\beta=1\%$. Mean $\pm$ one standard
        deviation over three seeds.
      }}
      \label{{tab:layers-by-strategy}}
      \centering
      \small
      \setlength{{\tabcolsep}}{{4pt}}
      \begin{{tabular}}{{rrrrrr}}
        \toprule
        Layer & AUC & Best acc. & Post & Warn$_{{128}}$ & Warn$_{{256}}$ \\
        \midrule
    {body}
        \bottomrule
      \end{{tabular}}
    \end{{table}}"""
        return register_table("tab:layers-by-strategy", latex, notes)

    table_layers = layers_table()
    return (table_layers,)


@app.cell
def _(
    BASELINES,
    BUDGETS,
    LAYER,
    S4,
    S4_LORA,
    baseline_views,
    fx,
    labels,
    pd,
    pm,
    probe_rows,
    register_table,
    seeds,
    signed,
):
    def baselines_table():
        test = labels["apertus"].query("split == 'test_indomain'")
        test = test[~test["onset_resolution"].isin(["judge_failed", "not_found"])]
        floor = 1 - test["is_positive"].mean()
        views = {name: baseline_views(name, "test_indomain") for name in BASELINES}
        probes = {}
        for regime, recipe in (("frozen", S4), ("adapted", S4_LORA)):
            for budget in BUDGETS:
                probes[regime, budget] = seeds(
                    probe_rows("apertus", recipe, LAYER, "test_indomain", budget),
                    ["rollout_auc", "best_accuracy", "in_pattern_recall", "warning_recall_256", "median_offset"],
                )

        def base(name, budget, column, d):
            v = views[name]
            if v is None:
                return "--"
            row = v["views"][v["views"]["target_negative_fpr"] == budget].iloc[0]
            if row["saturated"] or row["detected_positives"] == 0:
                return r"$\dag$"
            return signed(row[column]) if column == "median_offset" else fx(row[column], d)

        def probe(regime, budget, column, d):
            s = probes[regime, budget]
            if column == "median_offset":
                return pm(s[column], s[column + "_std"], 0, fmt=signed, std_fmt=fx)
            return pm(s[column], s[column + "_std"], d)

        names = list(BASELINES)
        auc = ["--"] + [fx(views[n]["auc"], 3) if views[n] else "--" for n in names] + [probe(r, 0.01, "rollout_auc", 4) for r in ("frozen", "adapted")]
        acc = [f"{floor:.4f}"] + [fx(views[n]["best_accuracy"], 4) if views[n] else "--" for n in names] + [probe(r, 0.01, "best_accuracy", 4) for r in ("frozen", "adapted")]
        lines = [
            r"    \multicolumn{2}{@{}l}{AUC}       & " + " & ".join(auc) + r" \\",
            r"    \multicolumn{2}{@{}l}{Best acc.} & " + " & ".join(acc) + r" \\",
            r"    \midrule",
        ]
        for k, budget in enumerate(BUDGETS):
            for i, (label, column, d) in enumerate((("Post", "in_pattern_recall", 3), ("Warn", "warning_recall_256", 3), ("Lead", "median_offset", 0))):
                never = "--" if column == "median_offset" else "0"
                cells = [never] + [base(n, budget, column, d) for n in names] + [probe(r, budget, column, d) for r in ("frozen", "adapted")]
                head = rf"\multirow{{3}}{{*}}{{$\beta={budget * 100:.0f}\%$}}" if i == 0 else ""
                lines.append((f"    {head}\n      & {label} & " if i == 0 else f"      & {label} & ") + " & ".join(cells) + r" \\")
            if k < len(BUDGETS) - 1:
                lines.append(r"    \addlinespace")
        heads = " & ".join(BASELINES.values())
        latex = rf"""\begin{{table}}[H]
      \caption{{Baseline scorers versus the \textsc{{Apertus}} probe on the in-domain
        test split. AUC and best accuracy are threshold-free; the remaining rows
        use thresholds calibrated on the validation set and then held fixed.
        Probe results are at layer~{LAYER}, mean $\pm$ one standard deviation over
        three seeds. \emph{{Warn}} denotes $\mathrm{{Warn}}_{{256}}$; \emph{{Lead}} is the
        median alarm offset from onset in tokens (negative: before onset).
        $\dag$: no threshold satisfies the false-alarm budget while producing
        alarms.}}
      \label{{tab:baselines-vs-probes}}
      \centering
      \footnotesize
      \setlength{{\tabcolsep}}{{3.5pt}}
      \begin{{tabular}}{{@{{}}llrrrrrrr@{{}}}}
        \toprule
        & & \multicolumn{{5}}{{c}}{{Baseline scorers}} & \multicolumn{{2}}{{c}}{{Probe (Apertus)}} \\
        \cmidrule(lr){{3-7}} \cmidrule(lr){{8-9}}
        & & never alarms & {heads} & frozen & adapted \\
        \midrule
    {chr(10).join(lines)}
        \bottomrule
      \end{{tabular}}
    \end{{table}}"""
        notes = [f"`{n}` is not scored yet; its column prints --." for n in names if views[n] is None]
        data = pd.DataFrame([{"scorer": n, "auc": views[n]["auc"], "best_accuracy": views[n]["best_accuracy"]} for n in names if views[n]])
        return register_table("tab:baselines-vs-probes", latex, notes)

    table_baselines = baselines_table()
    return (table_baselines,)


@app.cell
def _(fx, median_range, metrics, pd, pm, register_table, seeds, signed):
    def strategies_table():
        val = metrics[(metrics["model"] == "apertus") & (metrics["split"] == "val") & (metrics["target_negative_fpr"] == 0.01)]
        # Each seed reads the selectable depth with its highest validation Warn_256.
        chosen = val[val["selectable"]].sort_values("warning_recall_256", ascending=False).groupby(["recipe", "seed"]).head(1)
        seeds_per_recipe = val.groupby("recipe")["seed"].nunique()

        def group(r):
            if r["lora"] != "none":
                return "adapters"
            if r["label"] == "frontier_soft":
                return "soft"
            return "horizon" if r["horizon"] > 0 else "selection"

        def name(r, g):
            if g == "soft":
                kind = "exponential" if r["decay"].startswith("exponential") else "linear"
                return f"{kind}, decay ${r['decay'].removeprefix(kind)}$"
            text = rf"\texttt{{{r['selection'].replace('_', chr(92) + '_')}}}"
            if r["selection"] != "all_tokens":
                text += f" $W{{=}}{int(r['window'])}$"
            if g in ("horizon", "adapters"):
                if isinstance(r["anchor"], str):
                    text += " centred" if r["anchor"] == "centered" else " trailing"
                text += f" hard $K{{=}}{int(r['horizon'])}$"
            return text

        headers = {
            "selection": r"\textbf{Token selection} (frozen, hard label $K{=}0$)",
            "horizon": r"\textbf{Label horizon} (frozen, hard label pulled $K$ tokens before the onset)",
            "soft": r"\textbf{Soft labels} (frozen, \texttt{frontier\_window} $W{=}512$ centred, target fades in over the decay length)",
            "adapters": r"\textbf{Adapters} (LoRA rank 16 on blocks $1$ to $L$, probe read at layer $L$)",
        }
        columns = ["in_pattern_recall", "warning_recall_256", "median_offset", "fired_before_frontier", "never_fired_positives"]
        rows = []
        for recipe, part in chosen.groupby("recipe"):
            first = part.iloc[0]
            g = group(first)
            rows.append({"recipe": recipe, "group": g, "name": name(first, g), "layers": part["layer"].tolist(),
                         "steps": part["selected_step"].tolist(), "trained_seeds": int(seeds_per_recipe[recipe]), **seeds(part, columns)})
        frame = pd.DataFrame(rows).sort_values("warning_recall_256", ascending=False)
        no_selectable = sorted(set(seeds_per_recipe.index) - set(frame["recipe"]))
        lines, short = [], False
        for k, g in enumerate(headers):
            part = frame[frame["group"] == g]
            if part.empty:
                continue
            if lines:
                lines.append(r"\addlinespace")
            lines.append(rf"\multicolumn{{8}}{{l}}{{{headers[g]}}} \\")
            for _, r in part.iterrows():
                mark = "$^{\\ddag}$" if r["n"] < r["trained_seeds"] else ""
                short |= bool(mark)
                cells = [
                    r["name"] + mark, median_range(r["layers"]), median_range(r["steps"]),
                    pm(r["in_pattern_recall"], r["in_pattern_recall_std"], 3),
                    pm(r["warning_recall_256"], r["warning_recall_256_std"], 4),
                    pm(r["median_offset"], r["median_offset_std"], 0, fmt=signed, std_fmt=fx),
                    pm(r["fired_before_frontier"], r["fired_before_frontier_std"], 3),
                    pm(r["never_fired_positives"], r["never_fired_positives_std"], 1),
                ]
                lines.append(" & ".join(cells) + r" \\")
        dagger = (r" $\ddag$: fewer seeds than were trained, because the others produced no"
                  r" checkpoint satisfying the eligibility criterion.") if short else ""
        latex = rf"""\begin{{table}}[H]
      \caption{{Validation performance of the probe training strategies considered
        during model selection at $\beta=1\%$ and $m=1$. Mean $\pm$ one standard
        deviation over three seeds. \emph{{Layer}} and \emph{{Step}} report the median
        selected value and, in parentheses, the range across seeds. \emph{{Post}}:
        post-onset coverage; \emph{{Warn}}: $\mathrm{{Warn}}_{{256}}$; \emph{{Lead}}:
        median alarm offset from onset in tokens; \emph{{Early}}: fraction of
        degenerate responses alarmed before onset; \emph{{Never}}: number of
        degenerate responses for which no alarm is raised.{dagger}}}
      \label{{tab:strategies}}
      \centering
      \footnotesize
      \setlength{{\tabcolsep}}{{3pt}}
      \resizebox{{\linewidth}}{{!}}{{%
      \begin{{tabular}}{{lrrrrrrr}}
        \toprule
        Training strategy & Layer & Step & Post & Warn$_{{256}}$ & Lead & Early & Never \\
        \midrule
    {chr(10).join(lines)}
        \bottomrule
      \end{{tabular}}}}
    \end{{table}}"""
        notes = [
            "Each seed reads its own best selectable depth (validation Warn$_{256}$ at 1%); for the adapters each depth is a separate run, so the depth is chosen among the 31 runs of a seed.",
            "The new sweep has no regression or class-balance recipes; those groups are left out.",
        ]
        if no_selectable:
            notes.append("No seed has a selectable depth for: " + ", ".join(no_selectable))
        return register_table("tab:strategies", latex, notes)

    table_strategies = strategies_table()
    return (table_strategies,)


@app.cell
def _(mo):
    with_paper = mo.ui.switch(value=True, label="Put the paper's current version under each table")
    with_paper
    return (with_paper,)


@app.cell
def _(
    TABLE_DIR,
    compile_tex,
    io,
    mo,
    paper_source,
    re,
    table_baselines,
    table_blocks,
    table_budgeted,
    table_corpus,
    table_four_scorers,
    table_layers,
    table_metric_room,
    table_persistence,
    table_positional,
    table_rank_transfer,
    table_shortlist,
    table_strategies,
    with_paper,
):
    BOOKLET_PREAMBLE = r"""\documentclass{article}
    \usepackage[letterpaper,textwidth=5.5in,textheight=9in]{geometry}
    \usepackage[T1]{fontenc}
    \usepackage{times}
    \usepackage{amsmath,amsfonts,booktabs,multirow,graphicx,xcolor,float,nicefrac,xspace}
    \providecommand{\sd}[1]{{\scriptstyle\pm#1}}
    \newcommand{\method}{\textsc{Onward}\xspace}
    \setlength{\parindent}{0pt}
    \pagestyle{plain}
    \begin{document}
    """.replace("\n    ", "\n")

    def tex_text(note):
        """A markdown note as LaTeX text: `code` to typewriter, specials escaped outside math."""
        out = []
        for i, part in enumerate(note.split("`")):
            if i % 2:
                out.append(r"\texttt{" + re.sub(r"([_%&#{}$])", r"\\\1", part) + "}")
            else:
                pieces = part.split("$")
                for j, piece in enumerate(pieces):
                    pieces[j] = piece if j % 2 else re.sub(r"([_%&#])", r"\\\1", piece)
                out.append("$".join(pieces))
        return "".join(out)

    def booklet(tables, include_paper):
        """Every regenerated table on its own page, in the order the paper prints them."""
        source = paper_source()
        current = table_blocks(source)
        order = {label: match.start() for label, match in current.items()}
        tables = sorted((t for t in tables if t), key=lambda t: order.get(t["label"], 1e9))
        pages = []
        for t in tables:
            page = [rf"{{\large\bfseries\ttfamily {t['label']}}}\par\medskip", "\\textcolor{gray}{New runs}\\par", t["latex"]]
            if t["notes"]:
                page.append("{\\footnotesize\\color{red!60!black}\\begin{itemize}\\setlength{\\itemsep}{0pt}")
                page += [r"\item " + tex_text(n) for n in t["notes"]]
                page.append("\\end{itemize}}")
            if include_paper and t["label"] in current:
                page += ["\\bigskip\\hrule\\bigskip", "\\textcolor{gray}{Paper now}\\par", current[t["label"]].group(0)]
            pages.append("\n".join(page))
        body = "\n\\clearpage\n".join(pages)
        pdf, log = compile_tex(BOOKLET_PREAMBLE + body + "\n\\end{document}\n", "paper_tables")
        if pdf is not None:
            (TABLE_DIR / "paper_tables.pdf").write_bytes(pdf.read_bytes())
        return pdf, log, tables

    _tables = [table_corpus, table_shortlist, table_four_scorers, table_positional, table_persistence,
               table_metric_room, table_rank_transfer, table_budgeted, table_layers, table_baselines, table_strategies]
    _pdf, _log, _done = booklet(_tables, with_paper.value)
    mo.vstack([
        mo.md(
            f"{len(_done)} tables. One PDF with all of them: `{TABLE_DIR / 'paper_tables.pdf'}`; "
            f"one cropped PDF per table in `{TABLE_DIR / 'pdf'}`; the LaTeX of each in `{TABLE_DIR}`."
        ),
        mo.pdf(src=io.BytesIO(_pdf.read_bytes()), width="100%", height="1150px") if _pdf
        else mo.md(f"```\n{_log[-4000:]}\n```"),
    ])
    return


@app.cell
def _(mo):
    build_button = mo.ui.run_button(label="Compile the paper with the new tables")
    build_button
    return (build_button,)


@app.cell
def _(
    PAPER_DIR,
    TABLE_DIR,
    build_button,
    compile_tex,
    io,
    mo,
    paper_source,
    shutil,
    table_blocks,
):
    mo.stop(not build_button.value, mo.md("Press the button to build `main.tex` with every regenerated table swapped in."))

    def build_paper():
        """Copy the paper, replace each table float by its regenerated version, compile."""
        text = paper_source()
        replaced = []
        for label, match in sorted(table_blocks(text).items(), key=lambda kv: -kv[1].start()):
            new = TABLE_DIR / f"{label.replace(':', '_')}.tex"
            if new.is_file():
                text = text[: match.start()] + new.read_text().rstrip("\n") + text[match.end():]
                replaced.append(label)
        work = TABLE_DIR / "paper_build"
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(PAPER_DIR, work, ignore=shutil.ignore_patterns(".git", "old-versions"))
        (work / "main.tex").write_text(text)
        # The paper uses \mathbbm without loading bbm, which Overleaf tolerates
        # and tectonic does not unless told to keep going.
        pdf, log = compile_tex(text, "main", extra=("-Z", "continue-on-errors"), directory=work)
        return pdf, log, replaced, work

    _pdf, _log, _replaced, _work = build_paper()
    mo.vstack([
        mo.md(f"Replaced {len(_replaced)} tables: " + ", ".join(f"`{x}`" for x in sorted(_replaced))),
        mo.pdf(src=io.BytesIO(_pdf.read_bytes()), width="100%", height="1100px") if _pdf else mo.md("compile failed"),
        mo.accordion({"Compile log": mo.md(f"```\n{_log[-6000:]}\n```")}),
    ])
    return


if __name__ == "__main__":
    app.run()
