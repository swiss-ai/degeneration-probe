import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.collections import LineCollection
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    REPO = Path(__file__).resolve().parents[1]
    STORE = Path("/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe")
    RUNS_ROOT = STORE / "outputs"
    CACHE_DIR = RUNS_ROOT / "analysis" / "paper_figures_cache"
    FIG_DIR = Path(__file__).resolve().parent / "figures" / "paper_figures"
    PAPER_DATA = Path(__file__).resolve().parent / "paper_figures_data.npz"
    DATASET_BUILD = {
        "apertus": STORE / "degeneration-dataset-apertus-8b-instruct",
        "llama": STORE / "degeneration-dataset-llama3p1-8b-instruct",
    }
    DATASET_NAME = {"apertus": "apertus-8b-instruct", "llama": "llama3p1-8b-instruct"}
    # Hidden states produced by scripts/extract_lora_linearity_activations.py, one
    # directory per condition ("frozen" for the base model, "lora" for the adapted one).
    EXTRACTED_ACTIVATIONS = RUNS_ROOT / "paper_figures" / "activations"

    # The probe layer every main-text figure reads, and the probes it reads there.
    LAYER = 15
    S4_RECIPE = dict(selection="frontier_window", window=512, anchor="centered", label="frontier_hard", horizon=256, loss="bce")
    DEPTH_RECIPE = dict(selection="all_tokens", label="frontier_hard", horizon=1024, loss="bce")
    DEPTH_LAYERS = [4, 8, 12, 15, 20, 30]
    BUDGETS = (0.01, 0.05, 0.10)
    PERSISTENCE_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
    # The rollout drawn by both the PCA and the parallel-coordinates figures.
    FIGURE_ROLLOUT = ("deepmath_103k", "deepmath_103k_00196", 8)
    WARNING_BAND = 128

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    return (
        BUDGETS,
        CACHE_DIR,
        DATASET_BUILD,
        DATASET_NAME,
        DEPTH_LAYERS,
        DEPTH_RECIPE,
        EXTRACTED_ACTIVATIONS,
        FIGURE_ROLLOUT,
        FIG_DIR,
        HandlerTuple,
        LAYER,
        LineCollection,
        Line2D,
        PAPER_DATA,
        PERSISTENCE_GRID,
        Patch,
        RUNS_ROOT,
        S4_RECIPE,
        WARNING_BAND,
        hashlib,
        json,
        mo,
        mpl,
        np,
        os,
        pd,
        pe,
        plt,
    )


@app.cell
def _(mo, os):
    source = mo.ui.radio(
        options=["auto", "paper"], value=os.environ.get("PAPER_FIGURES_SOURCE", "auto"), inline=True
    )
    source
    return (source,)


@app.cell
def _(mpl, plt):
    INK = "#0b0b0b"
    INK_SOFT = "#52514e"
    INK_MUTED = "#8a8880"
    GRID = "#e3e2dd"
    MODEL_COLOR = {"apertus": "#d62728", "llama": "#1f77b4"}

    _base = {k: v for k, v in mpl.rcParamsDefault.items() if not k.startswith(("backend", "interactive"))}

    def style(**overrides):
        """Matplotlib defaults plus the overrides of one figure, independent of the session."""
        return mpl.rc_context({**_base, **overrides})

    STYLE_BARS = {
        "figure.dpi": 110, "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.titlelocation": "left",
        "axes.titlepad": 8, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK_SOFT, "axes.spines.top": False,
        "axes.spines.right": False, "text.color": INK, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "grid.color": GRID, "grid.linewidth": 0.7, "lines.linewidth": 2.0, "lines.markersize": 5,
        "mathtext.fontset": "cm", "hatch.linewidth": 0.8,
    }
    STYLE_OPERATING = {
        "font.size": 10, "axes.labelsize": 11, "axes.labelcolor": INK_SOFT, "axes.edgecolor": INK_MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "text.color": INK, "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT, "xtick.labelsize": 10, "ytick.labelsize": 10, "grid.color": GRID,
        "grid.linewidth": 0.7, "legend.fontsize": 9.5,
    }
    STYLE_METRICS = {"hatch.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False}
    STYLE_STATES = {
        "font.size": 16, "axes.titlesize": 22, "axes.labelsize": 19, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 15, "text.color": "#111111", "axes.labelcolor": "#111111", "xtick.color": "#111111",
        "ytick.color": "#111111", "axes.edgecolor": "#3a3a3a", "axes.linewidth": 1.0, "grid.color": "#c9c9c9",
        "grid.linewidth": 0.8, "legend.framealpha": 0.92, "legend.edgecolor": "#c9c9c9",
    }
    STYLE_PARALLEL = {
        **STYLE_STATES, "font.size": 11, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 11,
    }
    STYLE_JUDGE = {
        "font.size": 8.8, "axes.titlesize": 9.5, "axes.labelsize": 9.0, "axes.titlelocation": "left",
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.labelcolor": INK_SOFT, "axes.spines.top": False,
        "axes.spines.right": False, "text.color": INK, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "xtick.labelsize": 9.0, "ytick.labelsize": 9.0, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
        "legend.fontsize": 8.0, "legend.frameon": False,
    }
    STYLE_SWEEP = {
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 12, "axes.linewidth": 1.0, "axes.edgecolor": "black",
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#b0b0b0", "grid.linewidth": 0.8,
        "grid.alpha": 0.6, "xtick.labelsize": 10, "ytick.labelsize": 10, "xtick.direction": "out",
        "ytick.direction": "out", "legend.fontsize": 11, "legend.frameon": True, "legend.edgecolor": "#888888",
        "legend.facecolor": "white", "lines.linewidth": 1.8, "pdf.fonttype": 42,
    }

    def page_figure(width_pt, height_pt, dpi=None):
        """A figure whose canvas is the final page, so axes can be placed in points."""
        fig = plt.figure(figsize=(width_pt / 72, height_pt / 72), dpi=dpi)
        fig._page = (width_pt, height_pt)
        return fig

    def axes_at(fig, x, y, w, h, **kwargs):
        """An axes at (x, y, w, h) in points from the bottom-left corner of the page."""
        W, H = fig._page
        return fig.add_axes([x / W, y / H, w / W, h / H], **kwargs)

    def save(fig, path, **kwargs):
        with mpl.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(path, **kwargs)

    return (
        GRID,
        INK,
        INK_MUTED,
        INK_SOFT,
        MODEL_COLOR,
        STYLE_BARS,
        STYLE_JUDGE,
        STYLE_METRICS,
        STYLE_OPERATING,
        STYLE_PARALLEL,
        STYLE_STATES,
        STYLE_SWEEP,
        axes_at,
        page_figure,
        save,
        style,
    )


@app.cell
def _(PAPER_DATA, np, pd):
    _npz = np.load(PAPER_DATA)

    # Detection and early-warning performance of the layer-15 probes on the
    # in-domain test split (paper, Table budgeted): (mean, std) per metric.
    _budgeted = {
        ("apertus", "frozen"): {0.01: ((0.664, 0.015), (0.067, 0.007), (48, 14), (0.432, 0.021)),
                                0.05: ((0.835, 0.002), (0.207, 0.005), (-159, 12), (0.655, 0.018)),
                                0.10: ((0.880, 0.001), (0.300, 0.004), (-234, 7), (0.735, 0.000))},
        ("apertus", "adapted"): {0.01: ((0.840, 0.026), (0.194, 0.007), (56, 15), (0.430, 0.032)),
                                 0.05: ((0.952, 0.003), (0.486, 0.016), (-202, 10), (0.667, 0.026)),
                                 0.10: ((0.981, 0.003), (0.631, 0.032), (-337, 32), (0.792, 0.020))},
        ("llama", "frozen"): {0.01: ((0.244, 0.027), (0.013, 0.001), (227, 21), (0.240, 0.008)),
                              0.05: ((0.511, 0.038), (0.088, 0.009), (-106, 17), (0.593, 0.019)),
                              0.10: ((0.689, 0.034), (0.194, 0.017), (-343, 20), (0.771, 0.017))},
        ("llama", "adapted"): {0.01: ((0.374, 0.078), (0.038, 0.009), (363, 16), (0.210, 0.024)),
                               0.05: ((0.852, 0.055), (0.281, 0.040), (-81, 25), (0.571, 0.028)),
                               0.10: ((0.958, 0.017), (0.532, 0.060), (-366, 19), (0.780, 0.004))},
    }
    BUDGETED_PAPER = {
        series: {b: dict(zip(("post_onset", "warning_256", "lead", "early"), values)) for b, values in rows.items()}
        for series, rows in _budgeted.items()
    }

    METRICS_PAPER = {
        "scorers": ["best\nAUC", "adapted", "token\ncap", "worst\nAUC"],
        "hatched": [False, True, False, False],
        **{k: (_npz[f"metrics_{k}"], _npz[f"metrics_{k}_sd"]) for k in ("auc", "acc", "post", "warn", "early")},
        "depth": {k: (_npz[f"depth_{k}"], _npz[f"depth_{k}_sd"]) for k in ("auc", "acc", "post", "warn", "early")},
    }

    STATES_PAPER = {
        regime: {
            "points": {g: _npz[f"pca_{regime}_{g}"] for g in ("pre", "warn", "post")},
            "variance": _npz[f"pca_{regime}_var"],
        }
        for regime in ("frozen", "adapted")
    }
    PARALLEL_PAPER = {regime: {g: _npz[f"parallel_{regime}_{g}"] for g in ("pre", "warn", "post")} for regime in ("frozen", "adapted")}

    _keys = _npz["sweep_keys"].astype(str)
    SWEEP_PAPER = pd.DataFrame(_npz["sweep_values"], columns=_npz["sweep_columns"].astype(str))
    SWEEP_PAPER.insert(0, "model", [k.split("|")[0] for k in _keys])
    SWEEP_PAPER.insert(1, "regime", [k.split("|")[1] for k in _keys])
    SWEEP_PAPER.insert(2, "budget", _npz["sweep_budget"])
    SWEEP_PAPER.insert(3, "m", _npz["sweep_m"])
    # The lead-time column of the paper figure was drawn with its own y range per row.
    SWEEP_PAPER_LEAD_YLIM = {0.01: (8.08, 423.12), 0.05: (-230.75, -29.60), 0.10: (-489.22, -138.35)}

    # Judge versus human onsets on the confirmed degenerate responses (paper,
    # Appendix, judge protocol): responses whose judge onset lies within each
    # tolerance of the human one, and the signed error buckets.
    JUDGE_PAPER = {
        "tolerances": ["exact", "32", "64", "128", "256", "512", "1024", "2048"],
        "within": {"apertus": ([119, 127, 134, 142, 154, 157, 159, 162], 163),
                   "llama": ([350, 352, 357, 361, 366, 369, 373, 373], 374)},
        "buckets": ["early > 128", "early 1-128", "exact", "late 1-128", "late 129-512", "late > 512"],
        "counts": {"apertus": [1, 3, 119, 20, 15, 5], "llama": [1, 7, 350, 4, 7, 5]},
    }
    return (
        BUDGETED_PAPER,
        JUDGE_PAPER,
        METRICS_PAPER,
        PARALLEL_PAPER,
        STATES_PAPER,
        SWEEP_PAPER,
        SWEEP_PAPER_LEAD_YLIM,
    )


@app.cell
def _(
    BUDGETS,
    CACHE_DIR,
    DATASET_BUILD,
    RUNS_ROOT,
    hashlib,
    json,
    np,
    pd,
):
    from pathlib import Path as _Path

    from degeneration_probe.analysis.unit_levels import best_accuracy
    from degeneration_probe.evaluation.protocol import alarm_staircase, choose_thresholds
    from degeneration_probe.evaluation.scores import read_scores, unjudged_rollouts
    from sklearn.metrics import roc_auc_score

    def discover_runs():
        """Every scored (run, layer) on store, with the recipe axes of its run."""
        rows = []
        for info in RUNS_ROOT.glob("*/2*/run_info.json"):
            run = json.loads(info.read_text())
            axes = run.get("axes", {})
            dataset = axes.get("dataset") or ""
            model = "apertus" if dataset.startswith("apertus-8b") else "llama" if dataset.startswith("llama3p1") else None
            if model is None:
                continue
            selection_file = info.parent / "checkpoint_selection.json"
            selectable = {}
            if selection_file.is_file():
                for depth in json.loads(selection_file.read_text()).get("depths", []):
                    selectable[int(depth["layer"])] = bool(depth.get("selectable", True))
            for layer_dir in sorted((info.parent / "selected").glob("layer_*")):
                scores = layer_dir / "scores"
                if not (scores / "val.parquet").is_file() or not (scores / "test_indomain.parquet").is_file():
                    continue
                layer = int(layer_dir.name.split("_")[1])
                rows.append({
                    "run": run["run_name"], "model": model,
                    "regime": "adapted" if axes.get("features") == "adapted" else "frozen",
                    "selection": axes.get("selection"), "window": axes.get("window"), "anchor": axes.get("anchor"),
                    "label": axes.get("label"), "horizon": axes.get("horizon") or 0, "decay": axes.get("decay"),
                    "loss": axes.get("loss"), "pos_weight": axes.get("pos_weight"), "seed": axes.get("seed"),
                    "layer": layer, "selectable": selectable.get(layer, True), "scores_dir": str(scores),
                })
        columns = ["run", "model", "regime", "selection", "window", "anchor", "label", "horizon", "decay", "loss",
                   "pos_weight", "seed", "layer", "selectable", "scores_dir"]
        return pd.DataFrame(rows, columns=columns)

    def runs_of(runs, model, regime, layer, recipe):
        """The seeds of one recipe at one layer, at most one scored attempt per seed."""
        sel = runs[(runs["model"] == model) & (runs["regime"] == regime) & (runs["layer"] == layer)]
        for key, value in recipe.items():
            sel = sel[sel[key] == value]
        return sel.drop_duplicates("seed").sort_values("seed")

    _unjudged = {}

    def load_split(scores_dir, split, model):
        if model not in _unjudged:
            _unjudged[model] = unjudged_rollouts(DATASET_BUILD[model] / "onset_labels" / "onset_labels.parquet")
        return read_scores(f"{scores_dir}/{split}.parquet", split=split, unjudged=_unjudged[model])

    def cached(kind, scores_dir, params, compute):
        """Results keyed by the score files they were computed from, so a rescored run is recomputed."""
        stamps = [(p.name, p.stat().st_mtime_ns) for p in sorted(_Path(scores_dir).glob("*.parquet"))]
        key = hashlib.sha1(json.dumps([kind, str(scores_dir), params, stamps]).encode()).hexdigest()
        path = CACHE_DIR / f"{kind}_{key}.json"
        if path.is_file():
            return json.loads(path.read_text())
        result = compute()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        return result

    def response_metrics(frame):
        """Response-level AUC and best accuracy, pooling token scores with the maximum."""
        labels = frame["is_positive"].to_numpy(bool)
        peaks = np.array([np.max(s) if len(s) else 0.0 for s in frame["scores"]], dtype=np.float64)
        return {"auc": float(roc_auc_score(labels, peaks)), "acc": float(best_accuracy(labels.astype(int), peaks))}

    def onset_metrics(val, test, persistence):
        """Post-onset coverage, Warn_256, median lead and early share at each budget.

        Thresholds are calibrated on validation for runs of `persistence` tokens, the
        alarm is the token that completes the first such run, and coverage counts the
        raw tokens at or above the calibrated threshold.
        """
        thresholds = choose_thresholds(val, BUDGETS, persistence=persistence)
        positives = test[test["is_positive"].astype(bool)]
        scores = [np.asarray(s, np.float64) for s in positives["scores"]]
        onsets = positives["onset_position"].astype(int).to_numpy()
        stairs = [alarm_staircase(s, persistence) for s in scores]
        rows = []
        for threshold in thresholds:
            post_hits = post_total = warn_hits = warn_total = 0
            offsets = []
            for (positions, levels), s, onset in zip(stairs, scores, onsets):
                above = s >= threshold.tau
                post_hits += int(above[onset:].sum())
                post_total += s.size - onset
                band = above[max(0, onset - 256):onset]
                warn_hits += int(band.sum())
                warn_total += band.size
                k = int(np.searchsorted(levels, threshold.tau, side="left"))
                if k < levels.size:
                    offsets.append(int(positions[k]) + persistence - 1 - onset)
            offsets = np.asarray(offsets, float)
            rows.append({
                "budget": threshold.target_negative_fpr, "m": persistence,
                "post_onset": post_hits / post_total, "warning_256": warn_hits / warn_total,
                "lead": float(np.median(offsets)) if offsets.size else float("nan"),
                "early": float((offsets < 0).mean()) if offsets.size else float("nan"),
            })
        return rows

    def run_metrics(row, persistences):
        """Response-level and onset-aligned metrics of one scored (run, layer) on the in-domain test split."""
        def compute():
            val = load_split(row["scores_dir"], "val", row["model"])
            test = load_split(row["scores_dir"], "test_indomain", row["model"])
            onset_rows = [r for m in persistences for r in onset_metrics(val, test, m)]
            return {"test": response_metrics(test), "onset": onset_rows}
        return cached("run_metrics", row["scores_dir"], list(persistences), compute)

    def val_response(row):
        return cached("val_response", row["scores_dir"], [], lambda: response_metrics(load_split(row["scores_dir"], "val", row["model"])))

    def token_cap_metrics(model="apertus"):
        """The token-cap baseline: score one on the last token of a capped response, zero elsewhere."""
        labels_path = DATASET_BUILD[model] / "onset_labels" / "onset_labels.parquet"
        def compute():
            labels = pd.read_parquet(labels_path)
            labels = labels[~labels["onset_resolution"].isin(["judge_failed", "not_found"])]
            def frame(split):
                part = labels[labels["split"] == split].copy()
                part["scores"] = [
                    np.r_[np.zeros(int(n) - 1), 1.0] if stop == "length" else np.zeros(int(n))
                    for n, stop in zip(part["num_tokens"], part["stop_reason"])
                ]
                part["is_positive"] = part["is_positive"].astype(bool)
                return part
            val, test = frame("val"), frame("test_indomain")
            return {"test": response_metrics(test), "onset": onset_metrics(val, test, 1)}
        return cached("token_cap", str(labels_path.parent), [], compute)

    return (
        discover_runs,
        run_metrics,
        runs_of,
        token_cap_metrics,
        val_response,
    )


@app.cell
def _(
    BUDGETS,
    DEPTH_LAYERS,
    DEPTH_RECIPE,
    LAYER,
    PERSISTENCE_GRID,
    S4_RECIPE,
    discover_runs,
    np,
    pd,
    run_metrics,
    runs_of,
    token_cap_metrics,
    val_response,
):
    RUNS = discover_runs()
    SERIES = [("apertus", "frozen"), ("apertus", "adapted"), ("llama", "frozen"), ("llama", "adapted")]
    MISSING = {}

    def seed_table(rows, persistences):
        """One row per seed and (budget, m) with the onset metrics, plus the response-level ones."""
        records = []
        for _, row in rows.iterrows():
            metrics = run_metrics(row, persistences)
            for r in metrics["onset"]:
                records.append({"seed": row["seed"], **r, **{f"test_{k}": v for k, v in metrics["test"].items()}})
        return pd.DataFrame(records)

    def summarise(table, keys):
        cols = ["post_onset", "warning_256", "lead", "early"]
        grouped = table.groupby(keys)[cols]
        out = grouped.mean().join(grouped.std(ddof=1), rsuffix="_sd").reset_index()
        return out

    # S4 recipe at the read layer, every model and regime, every budget and persistence.
    _s4 = {}
    for _model, _regime in SERIES:
        _rows = runs_of(RUNS, _model, _regime, LAYER, S4_RECIPE)
        if len(_rows) >= 2:
            _s4[(_model, _regime)] = summarise(seed_table(_rows, PERSISTENCE_GRID), ["budget", "m"])
        else:
            MISSING.setdefault("s4", []).append(f"{_model} {_regime} layer {LAYER}: {len(_rows)} seed(s)")
    S4_LIVE = _s4 if len(_s4) == len(SERIES) else None

    BUDGETED_LIVE = None
    SWEEP_LIVE = None
    if S4_LIVE is not None:
        BUDGETED_LIVE = {}
        for _series, _summary in S4_LIVE.items():
            BUDGETED_LIVE[_series] = {}
            for _b in BUDGETS:
                _r = _summary[(np.isclose(_summary["budget"], _b)) & (_summary["m"] == 1)].iloc[0]
                BUDGETED_LIVE[_series][_b] = {k: (_r[k], _r[f"{k}_sd"]) for k in ("post_onset", "warning_256", "lead", "early")}
        SWEEP_LIVE = pd.concat(
            [s.assign(model=m, regime=r) for (m, r), s in S4_LIVE.items()], ignore_index=True
        )

    # The four scorers of the metrics figure: the frozen probes ranked first and last by
    # validation AUC at the read layer, the adapted S4 probe, and the token-cap baseline.
    METRICS_LIVE = None
    _adapted = runs_of(RUNS, "apertus", "adapted", LAYER, S4_RECIPE)
    _depth = {layer: runs_of(RUNS, "apertus", "frozen", layer, DEPTH_RECIPE) for layer in DEPTH_LAYERS}
    if len(_adapted) < 2:
        MISSING.setdefault("metrics", []).append(f"adapted S4 probe at layer {LAYER}: {len(_adapted)} seed(s)")
    _thin = [layer for layer, rows in _depth.items() if len(rows) < 2]
    if _thin:
        MISSING.setdefault("metrics", []).append(f"all_tokens h1024 frozen at layers {_thin}")
    _pool = []
    if "metrics" not in MISSING:
        _frozen = RUNS[(RUNS["model"] == "apertus") & (RUNS["regime"] == "frozen") & (RUNS["layer"] == LAYER) & RUNS["selectable"]]
        _recipe_cols = ["selection", "window", "anchor", "label", "horizon", "decay", "loss", "pos_weight"]
        for _recipe, _rows in _frozen.groupby(_recipe_cols, dropna=False):
            _rows = _rows.drop_duplicates("seed")
            if len(_rows) >= 2:
                _pool.append((float(np.mean([val_response(r)["auc"] for _, r in _rows.iterrows()])), _rows))
        if len(_pool) < 2:
            MISSING.setdefault("metrics", []).append(f"{len(_pool)} frozen recipe(s) with two or more seeds at layer {LAYER}")
    if "metrics" not in MISSING:
        _pool.sort(key=lambda item: item[0], reverse=True)

        def _scorer(rows):
            table = seed_table(rows, (1,))
            table = table[np.isclose(table["budget"], 0.01)]
            return {k: (table[c].mean(), table[c].std(ddof=1)) for k, c in (
                ("auc", "test_auc"), ("acc", "test_acc"), ("post", "post_onset"), ("warn", "warning_256"), ("early", "early"))}

        _cap = token_cap_metrics("apertus")
        _cap_onset = [r for r in _cap["onset"] if np.isclose(r["budget"], 0.01)][0]
        _scorers = [
            ("best\nAUC", False, _scorer(_pool[0][1])),
            ("adapted", True, _scorer(_adapted)),
            ("token\ncap", False, {"auc": (_cap["test"]["auc"], 0.0), "acc": (_cap["test"]["acc"], 0.0),
                                   "post": (_cap_onset["post_onset"], 0.0), "warn": (_cap_onset["warning_256"], 0.0),
                                   "early": (_cap_onset["early"], 0.0)}),
            ("worst\nAUC", False, _scorer(_pool[-1][1])),
        ]
        _scorers.sort(key=lambda s: s[2]["auc"][0], reverse=True)
        _depth_stats = {k: ([], []) for k in ("auc", "acc", "post", "warn", "early")}
        for layer in DEPTH_LAYERS:
            stats = _scorer(_depth[layer])
            for k in _depth_stats:
                _depth_stats[k][0].append(stats[k][0])
                _depth_stats[k][1].append(stats[k][1])
        METRICS_LIVE = {
            "scorers": [s[0] for s in _scorers],
            "hatched": [s[1] for s in _scorers],
            **{k: (np.array([s[2][k][0] for s in _scorers]), np.array([s[2][k][1] for s in _scorers]))
               for k in ("auc", "acc", "post", "warn", "early")},
            "depth": {k: (np.array(v[0]), np.array(v[1])) for k, v in _depth_stats.items()},
        }
    return BUDGETED_LIVE, METRICS_LIVE, MISSING, SWEEP_LIVE


@app.cell
def _(
    CACHE_DIR,
    DATASET_BUILD,
    EXTRACTED_ACTIVATIONS,
    FIGURE_ROLLOUT,
    LAYER,
    WARNING_BAND,
    np,
    pd,
):
    from safetensors import safe_open
    from sklearn.decomposition import PCA

    _labels = pd.read_parquet(DATASET_BUILD["apertus"] / "onset_labels" / "onset_labels.parquet")

    def onset_of(prompt_id, rollout_idx):
        row = _labels[(_labels["prompt_id"] == prompt_id) & (_labels["rollout_idx"] == rollout_idx)].iloc[0]
        return int(row["onset_position"])

    def activation_file(regime, domain, prompt_id, rollout_idx, training_cache):
        """Hidden states of one rollout from a forward pass, or from the training cache for the frozen model.

        The PCA figure reads the frozen model from the fp16 training cache, as the paper's
        figure did. The parallel-coordinates figure compares two bf16 forward passes of the
        same rollout, so both regimes carry the same rounding.
        """
        if regime == "frozen" and training_cache:
            cache = DATASET_BUILD["apertus"] / "activations" / domain / prompt_id / f"rollout_{rollout_idx}.safetensors"
            return cache if cache.is_file() else None
        extracted = EXTRACTED_ACTIVATIONS / ("lora" if regime == "adapted" else "frozen") / domain / prompt_id / f"rollout_{rollout_idx}.safetensors"
        return extracted if extracted.is_file() else None

    def groups(onset, n_tokens, tokens=None):
        tokens = np.arange(n_tokens) if tokens is None else tokens
        return {"pre": tokens[tokens < onset - WARNING_BAND], "warn": tokens[(tokens >= onset - WARNING_BAND) & (tokens < onset)],
                "post": tokens[tokens >= onset]}

    def states_live(regime):
        """The two leading principal components of every token at the read layer (slot layer + 1)."""
        domain, prompt_id, rollout_idx = FIGURE_ROLLOUT
        path = activation_file(regime, domain, prompt_id, rollout_idx, training_cache=True)
        if path is None:
            return None
        cache = CACHE_DIR / f"states_{regime}_{prompt_id}_{rollout_idx}_layer{LAYER}_{path.stat().st_mtime_ns}.npz"
        if cache.is_file():
            stored = np.load(cache)
            z, variance = stored["z"], stored["variance"]
        else:
            with safe_open(str(path), "np") as f:
                hidden = f.get_slice("hidden_states")[LAYER + 1].astype(np.float64)
            centred = hidden - hidden.mean(0)
            _, singular, vt = np.linalg.svd(centred, full_matrices=False)
            z = centred @ vt[:2].T
            variance = (singular ** 2 / (singular ** 2).sum())[:2]
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, z=z, variance=variance)
        index = groups(onset_of(prompt_id, rollout_idx), z.shape[0])
        return {"points": {g: z[idx] for g, idx in index.items()}, "variance": variance, "source": str(path)}

    def parallel_live(regime):
        """Per-layer first principal component of every fifth token, scaled to [0, 1] per layer.

        Each axis is oriented so the post-onset tokens sit above the pre-onset ones
        (comparing medians), and scaled by the range over all tokens of the response.
        """
        domain, prompt_id, rollout_idx = FIGURE_ROLLOUT
        path = activation_file(regime, domain, prompt_id, rollout_idx, training_cache=False)
        if path is None:
            return None
        cache = CACHE_DIR / f"parallel_{regime}_{prompt_id}_{rollout_idx}_{path.stat().st_mtime_ns}.npy"
        onset = onset_of(prompt_id, rollout_idx)
        if cache.is_file():
            values = np.load(cache)
        else:
            with safe_open(str(path), "np") as f:
                hidden = f.get_slice("hidden_states")
                n_slots = hidden.get_shape()[0]
                values = np.zeros((hidden.get_shape()[1], n_slots - 1))
                for layer in range(n_slots - 1):
                    c = PCA(n_components=1, random_state=0).fit_transform(hidden[layer + 1]).ravel()
                    if np.median(c[onset:]) < np.median(c[:onset - WARNING_BAND]):
                        c = -c
                    values[:, layer] = (c - c.min()) / (c.max() - c.min())
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, values)
        index = groups(onset, values.shape[0], np.arange(0, values.shape[0], 5))
        return {**{g: values[idx] for g, idx in index.items()}, "source": str(path)}

    STATES_LIVE = {regime: states_live(regime) for regime in ("frozen", "adapted")}
    PARALLEL_LIVE = {regime: parallel_live(regime) for regime in ("frozen", "adapted")}
    return PARALLEL_LIVE, STATES_LIVE


@app.cell
def _(
    BUDGETED_LIVE,
    BUDGETED_PAPER,
    METRICS_LIVE,
    METRICS_PAPER,
    PARALLEL_LIVE,
    PARALLEL_PAPER,
    STATES_LIVE,
    STATES_PAPER,
    SWEEP_LIVE,
    SWEEP_PAPER,
    source,
):
    _auto = source.value == "auto"

    def pick(live, paper):
        return (live, "live") if (_auto and live is not None) else (paper, "paper")

    DATA = {
        "budgeted": pick(BUDGETED_LIVE, BUDGETED_PAPER),
        "metrics": pick(METRICS_LIVE, METRICS_PAPER),
        "sweep": pick(SWEEP_LIVE, SWEEP_PAPER),
        **{f"states_{r}": pick(STATES_LIVE[r], STATES_PAPER[r]) for r in ("frozen", "adapted")},
        **{f"parallel_{r}": pick(PARALLEL_LIVE[r], PARALLEL_PAPER[r]) for r in ("frozen", "adapted")},
    }
    return (DATA,)


@app.cell
def _(
    DATA,
    FIG_DIR,
    INK,
    INK_SOFT,
    GRID,
    MODEL_COLOR,
    Patch,
    STYLE_BARS,
    axes_at,
    page_figure,
    save,
    style,
):
    def budget_bars(data, budget=0.05):
        models, regimes = ["apertus", "llama"], ["frozen", "adapted"]
        panels = [
            ("post_onset", r"(a) post-onset coverage ($\uparrow$)"),
            ("warning_256", r"(b) Warn₂₅₆ ($\uparrow$)"),
            ("early", r"(c) early warning rate ($\uparrow$)"),
            ("lead", r"(d) median lead time ($\downarrow$)"),
        ]
        with style(**STYLE_BARS):
            fig = page_figure(666.72, 174.56)
            axes = [axes_at(fig, 39.653125, 32.92, 131.7880695, 122.44)]
            axes += [axes_at(fig, x, 32.92, 131.7880695, 122.44, sharey=axes[0]) for x in (177.8211945, 315.9892641)]
            axes += [axes_at(fig, 521.142527, 32.92, 138.377473, 122.44)]
            offset, width = 0.39 / 2, 0.39 * 0.86
            for ax, (key, title) in zip(axes, panels):
                for mi, model in enumerate(models):
                    for ri, regime in enumerate(regimes):
                        mean, sd = data[(model, regime)][budget][key]
                        x = mi + (ri - 0.5) * 2 * offset
                        kw = dict(color=MODEL_COLOR[model], edgecolor=INK, linewidth=0.7, zorder=3)
                        if regime == "adapted":
                            ax.bar(x, mean, width, hatch="//", alpha=1.0, **kw)
                        else:
                            ax.bar(x, mean, width, alpha=0.9, **kw)
                        ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
                        if key == "lead":
                            ax.annotate(f"{mean:.0f}", (x, mean - sd), xytext=(0, -10), textcoords="offset points",
                                        ha="center", va="baseline", fontsize=6.5, color=INK_SOFT)
                        else:
                            ax.annotate(f"{mean:.2f}", (x, mean + sd), xytext=(0, 3), textcoords="offset points",
                                        ha="center", va="baseline", fontsize=6.5, color=INK_SOFT)
                ax.set_title(title, loc="center", pad=4)
                ax.set_xticks(range(len(models)), models)
                ax.tick_params(axis="x", length=0)
                ax.set_xlim(-0.62, 1.62)
                ax.grid(axis="y")
                ax.set_axisbelow(True)
                if key == "lead":
                    ax.axhline(0, color=INK_SOFT, lw=0.9)
                    lo = min(data[s][budget]["lead"][0] for s in data)
                    if lo < 0:
                        ax.set_ylim(lo * 1.22, -lo * 0.04)
                    ax.set_ylabel("alarm offset in tokens\n(negative is before onset)")
            axes[0].set_ylim(0, 1.06)
            axes[0].set_ylabel("share")
            for ax in axes[1:3]:
                ax.tick_params(axis="y", labelleft=False)
            handles = [
                Patch(facecolor=MODEL_COLOR["apertus"], edgecolor=INK, linewidth=0.7, label="apertus"),
                Patch(facecolor=MODEL_COLOR["llama"], edgecolor=INK, linewidth=0.7, label="llama"),
                Patch(facecolor="white", edgecolor=INK, linewidth=0.7, label="frozen"),
                Patch(facecolor="white", edgecolor=INK, linewidth=0.7, hatch="//", label="adapted"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, edgecolor=GRID, handlelength=1.6, columnspacing=1.8)
            save(fig, FIG_DIR / "fig5_budget_bars_5pct.pdf")
            save(fig, FIG_DIR / "fig5_budget_bars_5pct.png", dpi=300)
        return fig

    budget_bars(DATA["budgeted"][0])
    return


@app.cell
def _(
    BUDGETS,
    DATA,
    FIG_DIR,
    GRID,
    INK,
    INK_MUTED,
    INK_SOFT,
    Line2D,
    MODEL_COLOR,
    STYLE_OPERATING,
    axes_at,
    page_figure,
    pe,
    save,
    style,
):
    def operating_points(data, kind):
        series = [("apertus", "adapted"), ("apertus", "frozen"), ("llama", "adapted"), ("llama", "frozen")]
        xkey = "lead" if kind == "lead" else "early"
        rect = (51.84363636, 44.48, 339.4763636, 202.84) if kind == "lead" else (51.84363636, 44.48, 338.7234483, 202.84)
        with style(**STYLE_OPERATING):
            fig = page_figure(403.2, 259.2)
            ax = axes_at(fig, *rect)
            points = {s: [(data[s][b][xkey], data[s][b]["warning_256"]) for b in BUDGETS] for s in series}
            ymax = max(y + ye for pts in points.values() for (_, _), (y, ye) in pts)
            if kind == "lead":
                xmax = max(abs(x) + xe for pts in points.values() for (x, xe), _ in pts)
                xlim = max(460, 1.08 * xmax)
                ax.axvspan(0, xlim, color=GRID, alpha=0.55, lw=0, zorder=0)
                ax.axvline(0, color=INK_MUTED, lw=0.9)
                ax.text(0.25, 0.97, "first-alarm before onset", transform=ax.transAxes, ha="center", va="top",
                        fontsize=9, style="italic", color=INK_SOFT)
                ax.text(0.75, 0.97, "first-alarm after onset", transform=ax.transAxes, ha="center", va="top",
                        fontsize=9, style="italic", color=INK_SOFT)
                ax.set_xlim(-xlim, xlim)
                ax.set_ylim(0, max(0.78, ymax + 0.1))
                ax.set_xlabel("median lead time")
            else:
                lo = min(x - xe for pts in points.values() for (x, xe), _ in pts)
                hi = max(x + xe for pts in points.values() for (x, xe), _ in pts)
                ax.set_xlim(min(0.05, lo - 0.03), max(0.92, hi + 0.03))
                ax.set_ylim(0, max(0.72, ymax + 0.05))
                ax.set_xlabel("early warning rate")
            for (model, regime) in series:
                pts = points[(model, regime)]
                ax.errorbar([p[0][0] for p in pts], [p[1][0] for p in pts], xerr=[p[0][1] for p in pts],
                            yerr=[p[1][1] for p in pts], fmt="none", color=MODEL_COLOR[model], elinewidth=0.8, capsize=2, zorder=3)
            for (model, regime) in series:
                pts = points[(model, regime)]
                ax.plot([p[0][0] for p in pts], [p[1][0] for p in pts], color=MODEL_COLOR[model], lw=1.4,
                        ls="--" if regime == "adapted" else "-", marker="o", markersize=4.5, zorder=3)
            for (model, regime) in series:
                for b, ((x, _), (y, _)) in zip(BUDGETS, points[(model, regime)]):
                    above = regime == "adapted" or (b == 0.01 and (kind == "lead" or model == "llama"))
                    ax.annotate(f"{b:.0%}", (x, y), xytext=(0, 9 if above else -15), textcoords="offset points",
                                ha="center", va="baseline", fontsize=8.5, color=MODEL_COLOR[model],
                                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
            ax.set_ylabel(r"Warn$_{256}$")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            blank = Line2D([], [], linestyle="none")
            handles = [blank, Line2D([], [], color=MODEL_COLOR["apertus"], lw=2), Line2D([], [], color=MODEL_COLOR["llama"], lw=2),
                       blank, Line2D([], [], color=INK, lw=2), Line2D([], [], color=INK, lw=2, ls="--")]
            labels = ["Model", "apertus", "llama", "Regime", "frozen", "adapted"]
            where = dict(loc="upper right", bbox_to_anchor=(1, 0.92)) if kind == "lead" else dict(loc="upper left")
            legend = ax.legend(handles, labels, ncol=2, frameon=True, edgecolor=INK_MUTED, handlelength=1.9,
                               borderpad=0.6, columnspacing=1.4, labelspacing=0.45, **where)
            legend.get_frame().set_linewidth(0.8)
            for text in legend.get_texts():
                if text.get_text() in ("Model", "Regime"):
                    text.set_fontweight("bold")
            name = "fig0_intro_summary" if kind == "lead" else "fig0_coverage_vs_early"
            save(fig, FIG_DIR / f"{name}.pdf")
            save(fig, FIG_DIR / f"{name}.png", dpi=300)
        return fig

    operating_points(DATA["budgeted"][0], "lead")
    return (operating_points,)


@app.cell
def _(DATA, operating_points):
    operating_points(DATA["budgeted"][0], "early")
    return


@app.cell
def _(
    DATA,
    DEPTH_LAYERS,
    FIG_DIR,
    HandlerTuple,
    Line2D,
    Patch,
    STYLE_METRICS,
    axes_at,
    np,
    page_figure,
    save,
    style,
):
    COLORS = {"auc": "#c4c4c4", "acc": "#8a8a8a", "post": "#2c6fa8", "warn": "#c0392b", "early": "#2e7d52"}
    LABEL_INK, ORDER_INK = "#2f3437", "#8a8a8a"
    LINE_STYLE = {
        "auc": dict(lw=1.7, marker="s", ms=4.6),
        "acc": dict(lw=1.5, ls="--", marker="o", ms=4.2, mfc="white", mew=1.3),
        "post": dict(lw=1.9, marker="^", ms=5.5),
        "early": dict(lw=1.9, marker="D", ms=5.5),
        "warn": dict(lw=1.9, marker="o", ms=5.5),
    }

    def ratio_label(values):
        r = max(values) / min(values)
        return f"{r:.0f}$\\times$" if r >= 10 else f"{r:.1f}$\\times$"

    def depth_lines(ax, depth, keys):
        for key in keys:
            m, s = (np.asarray(v, float) for v in depth[key])
            ax.fill_between(DEPTH_LAYERS, m - s, m + s, color=COLORS[key], alpha=0.18, lw=0)
        for key in keys:
            ax.plot(DEPTH_LAYERS, depth[key][0], color=COLORS[key], **LINE_STYLE[key])

    def metrics_figure(data, variant):
        scorers, hatched = data["scorers"], data["hatched"]
        depth = data["depth"]
        x = np.arange(len(scorers))
        with style(**STYLE_METRICS):
            if variant == "zoom":
                fig = page_figure(604.8, 259.2)
                y0, h = 51.84, 163.296
            else:
                fig = page_figure(604.8, 230.4)
                y0, h = 50.688, 135.936
            axa = axes_at(fig, 30.24, y0, 136.9320926966, h)
            axb = axes_at(fig, 210.2216207865, y0, 180.1738061798, h)
            if variant == "zoom":
                axu = axes_at(fig, 433.4449550562, 180.1239931389, 144.1390449438, 35.0120068611)
                axc = axes_at(fig, 433.4449550562, 51.84, 144.1390449438, 119.0408233276)
            else:
                axc = axes_at(fig, 433.4449550562, y0, 144.1390449438, h)
            # (a) response level
            for k, key in enumerate(["auc", "acc"]):
                axa.bar(x + (k - 0.5) * 0.36, data[key][0], 0.36, color=COLORS[key], edgecolor="white", linewidth=1.0,
                        hatch=["//" if hh else "" for hh in hatched])
            axa.annotate("", xy=(3.45, -0.265), xytext=(-0.45, -0.265), xycoords=("data", "axes fraction"),
                         arrowprops=dict(arrowstyle="->", color=ORDER_INK, lw=1.0, mutation_scale=9), annotation_clip=False)
            for k, key in enumerate(["auc", "acc"]):
                axa.errorbar(x + (k - 0.5) * 0.36, data[key][0], yerr=data[key][1], fmt="none", ecolor=LABEL_INK,
                             elinewidth=1.1, capsize=3, zorder=4)
            for i in range(len(scorers)):
                for k, key in enumerate(["auc", "acc"]):
                    axa.text(x[i] + (k - 0.5) * 0.36, 0.52, f"{data[key][0][i]:.4f}", rotation=90, ha="center", va="center",
                             fontsize=7.8, color=LABEL_INK, zorder=4,
                             bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.93))
            axa.set_xticks(x, scorers, fontsize=8.5)
            axa.set_yticks([0, 0.5, 1.0])
            axa.tick_params(axis="y", labelsize=9)
            axa.set_ylim(0, 1.08)
            axa.grid(alpha=0.25)
            axa.set_axisbelow(True)
            axa.set_title("(a) response level", loc="left", fontsize=10.5)
            axa.text(0.5, -0.25, "ordered by AUC [desc]", transform=axa.transAxes, ha="center", va="bottom", fontsize=8.2, color=ORDER_INK)
            # (b) token level
            keys = ["post", "warn", "early"]
            for k, key in enumerate(keys):
                axb.bar(x + (k - 1) * 0.27, data[key][0], 0.27, color=COLORS[key], edgecolor="white", linewidth=1.0,
                        hatch=["//" if hh else "" for hh in hatched])
            axb.annotate("", xy=(3.45, -0.265), xytext=(-0.45, -0.265), xycoords=("data", "axes fraction"),
                         arrowprops=dict(arrowstyle="->", color=ORDER_INK, lw=1.0, mutation_scale=9), annotation_clip=False)
            for k, key in enumerate(keys):
                axb.errorbar(x + (k - 1) * 0.27, data[key][0], yerr=data[key][1], fmt="none", ecolor=LABEL_INK,
                             elinewidth=1.1, capsize=3, zorder=4)
            for k, key in enumerate(keys):
                mean, sd = data[key]
                for i in range(len(scorers)):
                    text = f"{mean[i]:.3f}" if key == "warn" else f"{mean[i]:.2f}"
                    if mean[i] > 0.25:
                        axb.text(x[i] + (k - 1) * 0.27, mean[i] / 2, text, rotation=90, ha="center", va="center", fontsize=8,
                                 color=COLORS[key], zorder=4, bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.93))
                    else:
                        axb.text(x[i] + (k - 1) * 0.27, mean[i] + sd[i] + 0.03, text, rotation=90, ha="center", va="bottom",
                                 fontsize=8, color=COLORS[key], zorder=4)
            axb.set_xticks(x, scorers, fontsize=8.5)
            axb.set_yticks([0, 0.5, 1.0])
            axb.tick_params(axis="y", labelsize=9)
            axb.set_ylim(0, 1.05)
            axb.grid(alpha=0.25)
            axb.set_axisbelow(True)
            axb.set_title("(b) token level", loc="left", fontsize=10.5)
            axb.text(0.5, -0.25, "ordered by AUC [desc]", transform=axb.transAxes, ha="center", va="bottom", fontsize=8.2, color=ORDER_INK)
            # (c) across layers
            low = min(1.2e-3, 0.8 * float(np.min(np.asarray(depth["warn"][0]) - np.asarray(depth["warn"][1]))))
            if variant == "zoom":
                depth_lines(axu, depth, ["auc", "acc"])
                lowest = float(np.min(np.asarray(depth["acc"][0]) - np.asarray(depth["acc"][1])))
                axu.set_ylim(min(0.988, lowest - 0.002), 1.0016)
                axu.set_xlim(2, 48)
                axu.set_xticks(DEPTH_LAYERS)
                axu.tick_params(axis="x", labelbottom=False)
                axu.set_yticks([0.99, 1.0])
                axu.tick_params(axis="y", labelsize=8)
                axu.grid(alpha=0.25)
                axu.set_axisbelow(True)
                axu.text(31.0, depth["auc"][0][-1], "AUC", fontsize=8, color=LABEL_INK, va="center")
                axu.text(31.0, depth["acc"][0][-1], "accuracy", fontsize=8, color=LABEL_INK, va="center")
                axu.set_title("(c) across layers", loc="left", fontsize=10.5)
            depth_lines(axc, depth, ["auc", "acc", "post", "early", "warn"])
            axc.set_yscale("log")
            axc.set_ylim(low, 1.9 if variant == "zoom" else 2.8)
            axc.set_xlim(2, 48)
            axc.set_xticks(DEPTH_LAYERS, [str(layer) for layer in DEPTH_LAYERS], fontsize=7)
            axc.tick_params(axis="y", labelsize=8.5)
            axc.grid(alpha=0.25)
            axc.set_axisbelow(True)
            axc.set_xlabel("layer", fontsize=9)
            axc.set_ylabel("log scale", fontsize=8.5)
            for key, xa in (("warn", 33.0), ("early", 36.5), ("post", 40.0)):
                m = depth[key][0]
                hi, lo = max(m), min(m)
                axc.annotate("", xy=(xa, lo), xytext=(xa, hi), arrowprops=dict(arrowstyle="<->", color=COLORS[key], lw=1.3, mutation_scale=9))
                axc.text(xa + 0.8, (hi * lo) ** 0.5, ratio_label(m), fontsize=8.2, color=COLORS[key], va="center")
            if variant != "zoom":
                a, c = depth["auc"][0], depth["acc"][0]
                axc.text(0.02, 0.985, f"AUC {max(a) / min(a):.3f}$\\times$,  accuracy {max(c) / min(c):.3f}$\\times$",
                         transform=axc.transAxes, fontsize=7.6, color=LABEL_INK, va="top")
                axc.set_title("(c) across layers", loc="left", fontsize=10.5)
            handles = [
                (Patch(facecolor=COLORS["auc"], edgecolor="none"), Line2D([], [], color=COLORS["auc"], **LINE_STYLE["auc"])),
                (Patch(facecolor=COLORS["acc"], edgecolor="none"), Line2D([], [], color=COLORS["acc"], **LINE_STYLE["acc"])),
                Patch(facecolor=COLORS["post"], edgecolor="none"), Patch(facecolor=COLORS["warn"], edgecolor="none"),
                Patch(facecolor=COLORS["early"], edgecolor="none"),
                Patch(facecolor="white", edgecolor=LABEL_INK, hatch="//"),
            ]
            labels = ["AUC", "accuracy", "post-onset coverage", "warning coverage", "early-warning rate", "adapted (LoRA)"]
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=6, frameon=False, fontsize=8.3,
                       handlelength=2.4, handletextpad=0.6, columnspacing=1.0, handler_map={tuple: HandlerTuple(ndivide=None)})
            name = "fig_metrics_zoom" if variant == "zoom" else "fig_metrics_inline"
            save(fig, FIG_DIR / f"{name}.pdf")
            save(fig, FIG_DIR / f"{name}.png", dpi=300)
        return fig

    metrics_figure(DATA["metrics"][0], "zoom")
    return (metrics_figure,)


@app.cell
def _(DATA, metrics_figure):
    metrics_figure(DATA["metrics"][0], "inline")
    return


@app.cell
def _(FIG_DIR, LAYER, STYLE_STATES, axes_at, page_figure, save, style):
    def states_figure(data, regime):
        pts, variance = data["points"], data["variance"]
        with style(**STYLE_STATES):
            fig = page_figure(705.6, 604.8)
            ax = axes_at(fig, 93.26, 63.56, 592.9, 489.24)
            ax.scatter(pts["pre"][:, 0], pts["pre"][:, 1], s=25.9, c="#2ca02c", alpha=0.55, edgecolors="none", label="pre-onset tokens")
            ax.scatter(pts["post"][:, 0], pts["post"][:, 1], s=29.6, c="#d62728", alpha=0.5, edgecolors="none", label="post-onset tokens", zorder=3)
            ax.scatter(pts["warn"][:, 0], pts["warn"][:, 1], s=85.1, c="#ff7f0e", alpha=0.95, edgecolors="white", linewidths=0.6,
                       label=f"{len(pts['warn'])} warning band tokens", zorder=4)
            ax.margins(0.06)
            ax.grid(True)
            ax.set_axisbelow(True)
            ax.set_xlabel(f"PC1 ({variance[0]:.1%} var)")
            ax.set_ylabel(f"PC2 ({variance[1]:.1%} var)")
            ax.set_title(f"layer {LAYER} - apertus {regime} - warning band 128 tokens", pad=16)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([handles[0], handles[2], handles[1]], [labels[0], labels[2], labels[1]], loc="upper left", borderpad=0.5,
                      handlelength=1.4, handletextpad=0.6, labelspacing=0.35, fancybox=False, scatteryoffsets=[0.5])
            name = f"pca2d_layer{LAYER}_deepmath_103k_00196_r8_{regime}_w128_dots2"
            save(fig, FIG_DIR / f"{name}.pdf")
            save(fig, FIG_DIR / f"{name}.png", dpi=200)
        return fig

    return (states_figure,)


@app.cell
def _(DATA, states_figure):
    states_figure(DATA["states_frozen"][0], "frozen")
    return


@app.cell
def _(DATA, states_figure):
    states_figure(DATA["states_adapted"][0], "adapted")
    return


@app.cell
def _(
    FIG_DIR,
    LineCollection,
    Line2D,
    STYLE_PARALLEL,
    axes_at,
    np,
    page_figure,
    save,
    style,
):
    GROUP_STYLE = (("pre", "#2ca02c", 0.6, 0.35, 2), ("post", "#d62728", 0.5, 0.13, 3), ("warn", "#ff7f0e", 1.0, 0.75, 4))

    def parallel_figure(data, regime):
        n_layers = data["pre"].shape[1]
        x = np.arange(n_layers)
        with style(**STYLE_PARALLEL):
            fig = page_figure(1152, 532.8)
            ax = axes_at(fig, 15.12, 51.32, 1121.76, 438.68)
            for group, color, lw, alpha, z in GROUP_STYLE:
                segments = [np.column_stack([x, row]) for row in data[group]]
                ax.add_collection(LineCollection(segments, colors=color, linewidths=lw, alpha=alpha, zorder=z))
            ax.set_xlim(-0.6, n_layers - 0.4)
            ax.set_ylim(-0.03, 1.17)
            ax.set_xticks(x)
            ax.set_yticks([])
            for i in x:
                ax.axvline(i, color="#c9c9c9", lw=0.8, zorder=0)
            ax.set_xlabel("Apertus-8B-Instruct-2509 layer")
            ax.set_title(f"apertus {regime} - warning band 128 tokens", pad=14)
            handles = [Line2D([], [], color=c, lw=2.6) for _, c, *_ in (GROUP_STYLE[0], GROUP_STYLE[2], GROUP_STYLE[1])]
            ax.legend(handles, ["pre-onset tokens", "128 warning band tokens", "post-onset tokens"], loc="upper center",
                      borderpad=0.5, handlelength=1.4, handletextpad=0.6, labelspacing=0.35, fancybox=False)
            save(fig, FIG_DIR / f"parallel_{regime}_w128.pdf")
            save(fig, FIG_DIR / f"parallel_{regime}_w128.png", dpi=200)
        return fig

    return (parallel_figure,)


@app.cell
def _(DATA, parallel_figure):
    parallel_figure(DATA["parallel_frozen"][0], "frozen")
    return


@app.cell
def _(DATA, parallel_figure):
    parallel_figure(DATA["parallel_adapted"][0], "adapted")
    return


@app.cell
def _(FIG_DIR, INK_SOFT, JUDGE_PAPER, STYLE_JUDGE, np, plt, save, style):
    def judge_figure(data):
        blue, orange = "#2a78d6", "#eb6834"
        dpi, width, height = 300, 2163, 844

        def px_axes(fig, x0, x1, top, bottom):
            return fig.add_axes([x0 / width, (height - bottom) / height, (x1 - x0) / width, (bottom - top) / height])

        with style(**STYLE_JUDGE):
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            axl = px_axes(fig, 183.49, 957.4, 86.03, 696.25)
            axr = px_axes(fig, 1332.0, 2106.1, 86.03, 696.25)
            x = np.arange(len(data["tolerances"]))
            for name, color, marker, label in (("apertus", blue, "o", "Apertus-8B"), ("llama", orange, "s", "Llama-3.1-8B")):
                k, n = data["within"][name]
                axl.plot(x, 100 * np.array(k) / n, color=color, lw=2.0, marker=marker, ms=4.5, mfc="white", mew=1.6,
                         label=f"{label} (n={n})")
            for name, color, i, dy in (("apertus", blue, 0, -13), ("apertus", blue, 3, -13), ("llama", orange, 0, 6), ("llama", orange, 3, 6)):
                k, n = data["within"][name]
                v = 100 * k[i] / n
                axl.annotate(f"{v:.0f}%", (x[i], v), xytext=(4, dy), textcoords="offset points", color=color, fontsize=8.5, fontweight="bold")
            axl.set_xticks(x, data["tolerances"])
            axl.set_xlim(-0.2, len(x) - 0.8)
            axl.set_ylim(60, 101)
            axl.set_yticks([60, 70, 80, 90, 100])
            axl.grid(axis="y")
            axl.set_axisbelow(True)
            axl.set_xlabel("onset tolerance (tokens)")
            axl.set_ylabel("rollouts within tolerance (%)")
            axl.set_title("Onset agreement with the human reviewer")
            axl.legend(loc="lower right", handlelength=1.6)
            y = np.arange(len(data["buckets"]))
            h = 0.36
            for j, (name, color, label) in enumerate((("apertus", blue, "Apertus-8B"), ("llama", orange, "Llama-3.1-8B"))):
                counts = np.array(data["counts"][name])
                share = 100 * counts / counts.sum()
                axr.barh(y + (j - 0.5) * h, share, h, color=color, label=label)
                for yy, p, c in zip(y + (j - 0.5) * h, share, counts):
                    axr.text(p + 1.5, yy, str(c), va="center", fontsize=7.5, color=INK_SOFT)
            axr.set_yticks(y, data["buckets"], fontsize=9.0)
            axr.invert_yaxis()
            axr.set_xlim(0, 108)
            axr.spines["left"].set_visible(False)
            axr.tick_params(axis="y", length=0)
            axr.grid(axis="x")
            axr.set_axisbelow(True)
            axr.set_xlabel("confirmed rollouts (%)")
            axr.set_title("When wrong, the judge usually fires late")
            axr.legend(loc="lower right", handlelength=1.2)
            save(fig, FIG_DIR / "judge_onset_agreement_v2.png", dpi=dpi)
            save(fig, FIG_DIR / "judge_onset_agreement_v2.pdf")
        return fig

    judge_figure(JUDGE_PAPER)
    return


@app.cell
def _(
    DATA,
    FIG_DIR,
    Line2D,
    MODEL_COLOR,
    PERSISTENCE_GRID,
    STYLE_SWEEP,
    SWEEP_PAPER_LEAD_YLIM,
    np,
    plt,
    save,
    style,
):
    def sweep_figure(data, paper):
        budgets = [0.01, 0.05, 0.10]
        metrics = [("post_onset", "Post-onset coverage"), ("warning_256", "Warning@256"), ("lead", "Lead time (tokens)"), ("early", "Early share")]
        regime_style = {"frozen": "-", "adapted": (0, (5, 3))}
        with style(**STYLE_SWEEP):
            fig, axes = plt.subplots(3, 4, figsize=(16, 8.3), sharex=True)
            for col in (0, 1, 3):
                for row in (1, 2):
                    axes[row, col].sharey(axes[0, col])
            for row, b in enumerate(budgets):
                rows = data[np.isclose(data["budget"], b)]
                for col, (stub, label) in enumerate(metrics):
                    ax = axes[row, col]
                    for model in ("apertus", "llama"):
                        for regime in ("frozen", "adapted"):
                            s = rows[(rows["model"] == model) & (rows["regime"] == regime)].sort_values("m")
                            if s.empty:
                                continue
                            m, mu, sd = s["m"].to_numpy(), s[stub].to_numpy(), s[f"{stub}_sd"].to_numpy()
                            ax.plot(m, mu, color=MODEL_COLOR[model], ls=regime_style[regime], lw=1.8)
                            ax.fill_between(m, mu - sd, mu + sd, color=MODEL_COLOR[model], alpha=0.2, lw=0)
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(PERSISTENCE_GRID)
                    ax.set_xticklabels([str(m) for m in PERSISTENCE_GRID])
                    if stub == "lead" and paper:
                        ax.set_ylim(*SWEEP_PAPER_LEAD_YLIM[b])
                    if row == 0:
                        ax.set_title(label, fontsize=13)
                    if col == 0:
                        ax.set_ylabel(f"beta = {b:.0%}", fontsize=12)
                    if row == 2:
                        ax.set_xlabel("Persistence window m", fontsize=11)
            blank = Line2D([], [], color="none")
            handles = [blank, Line2D([], [], color=MODEL_COLOR["apertus"], lw=2.2), Line2D([], [], color=MODEL_COLOR["llama"], lw=2.2),
                       blank, Line2D([], [], color="black", lw=2.2, ls=regime_style["frozen"]),
                       Line2D([], [], color="black", lw=2.2, ls=regime_style["adapted"])]
            labels = ["Model", "apertus", "llama", "Regime", "frozen", "adapted"]
            legend = fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.04),
                                handletextpad=0.8, columnspacing=1.6)
            for text, l in zip(legend.get_texts(), labels):
                if l in ("Model", "Regime"):
                    text.set_fontweight("bold")
            fig.tight_layout(rect=(0, 0.08, 1, 0.95))
            fig.savefig(FIG_DIR / "new_sweep_m.png", dpi=200, bbox_inches="tight")
            fig.savefig(FIG_DIR / "new_sweep_m.pdf", bbox_inches="tight")
        return fig

    sweep_figure(DATA["sweep"][0], DATA["sweep"][1] == "paper")
    return


@app.cell
def _(DATA, MISSING, pd):
    _needs = {
        "budgeted": ("fig5_budget_bars_5pct, fig0_intro_summary, fig0_coverage_vs_early", "s4"),
        "sweep": ("new_sweep_m", "s4"),
        "metrics": ("fig_metrics_zoom, fig_metrics_inline", "metrics"),
        "states_frozen": ("pca2d ... frozen", None), "states_adapted": ("pca2d ... adapted", None),
        "parallel_frozen": ("parallel_frozen_w128", None), "parallel_adapted": ("parallel_adapted_w128", None),
    }
    _rows = []
    for _key, (_figures, _missing_key) in _needs.items():
        _data, _source = DATA[_key]
        _detail = ""
        if _source == "live" and isinstance(_data, dict) and "source" in _data:
            _detail = _data["source"]
        elif _source == "paper":
            _detail = "; ".join(MISSING.get(_missing_key, [])) if _missing_key else "no extracted hidden states for this regime"
        _rows.append({"figures": _figures, "source": _source, "detail": _detail})
    _rows.append({"figures": "judge_onset_agreement_v2", "source": "paper", "detail": "human review counts"})
    pd.DataFrame(_rows)
    return


if __name__ == "__main__":
    app.run()
