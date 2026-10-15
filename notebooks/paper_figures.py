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
    # The depth panel reads the frozen S4 probe, the frozen twin of the adapted probe.
    DEPTH_RECIPE = S4_RECIPE
    DEPTH_LAYERS = [4, 8, 12, 15, 20, 30]
    # The budget and split of the metrics figure (response-level against onset-aligned).
    METRICS_BUDGET = 0.05
    METRICS_SPLIT = "test_indomain"
    BUDGETS = (0.01, 0.05, 0.10)
    PERSISTENCE_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
    # The rollout drawn by both the PCA and the parallel-coordinates figures.
    FIGURE_ROLLOUT = ("deepmath_103k", "deepmath_103k_00196", 8)
    WARNING_BAND = 128
    # Labels of the test splits: "human" reads the human-verified labels
    # (scripts/evaluate_human_labels.py), the ones the paper reports; "judge"
    # reads the judge's. Validation is always judge-labelled.
    LABELS = os.environ.get("PAPER_FIGURES_LABELS", "human")
    ANALYSIS_SUFFIX = "_human" if LABELS == "human" else ""

    def read_onset_labels(build_dir):
        """A build's onset labels, with the reviewers' verdicts and onsets on the test splits when LABELS is human."""
        labels = pd.read_parquet(Path(build_dir) / "onset_labels" / "onset_labels.parquet")
        if LABELS != "human":
            return labels
        dataset = Path(build_dir).name.replace("degeneration-dataset-", "")
        reviewed = pd.read_csv(STORE / "human_review" / "labels" / f"labels_{dataset}.csv").set_index("id")
        keys = labels["prompt_id"] + "::" + labels["rollout_idx"].astype(str)
        labels = labels[~keys.isin(reviewed.index[reviewed["drop"]])].copy()
        keys = labels["prompt_id"] + "::" + labels["rollout_idx"].astype(str)
        hit = keys.isin(reviewed.index).to_numpy()
        labels.loc[hit, "is_positive"] = reviewed.loc[keys[hit], "is_positive"].to_numpy(bool)
        labels.loc[hit, "onset_position"] = reviewed.loc[keys[hit], "onset_position"].to_numpy(float)
        return labels

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    return (
        ANALYSIS_SUFFIX,
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
        METRICS_BUDGET,
        METRICS_SPLIT,
        PAPER_DATA,
        PERSISTENCE_GRID,
        Patch,
        RUNS_ROOT,
        S4_RECIPE,
        WARNING_BAND,
        hashlib,
        read_onset_labels,
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
def _(Patch, mpl, plt):
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

    # The figures of the revised paper share one style. They are drawn at the text width of
    # the NeurIPS template, so one point of the canvas is one point on the page, and set in
    # Times like the body text (STIX is matplotlib's Times clone).
    TEXT_WIDTH = 397.5
    # Regions of a degenerate response and the metrics that count them. The hues stay apart
    # under protanopia, deuteranopia and tritanopia (Machado et al. 2009, CIEDE2000 >= 20).
    PAPER_COLOR = {
        "pre": "#56B4E9", "warn": "#E69F00", "post": "#882255", "early": "#0072B2", "det": "#cc8fb0",
        "auc": "#c4c4c4", "acc": "#8a8a8a", "lead": "#6b6b6b",
    }
    # One colour per model, for figures that give each model its own panel (Paul Tol's muted set).
    PAPER_MODEL_COLOR = {"apertus": "#44AA99", "llama3p1": "#332288", "mistral": "#CC6677"}
    PAPER_SIZE = {"tick": 7.0, "label": 7.5, "legend": 7.5, "title": 8.0, "note": 6.5}
    STYLE_PAPER = {
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "pdf.fonttype": 42,
        "font.size": PAPER_SIZE["label"], "axes.unicode_minus": True,
        "axes.titlesize": PAPER_SIZE["title"], "axes.titlelocation": "left", "axes.titlepad": 3.5,
        "axes.labelsize": PAPER_SIZE["label"], "axes.labelpad": 2.5, "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.6,
        "axes.labelcolor": INK_SOFT, "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
        "xtick.labelsize": PAPER_SIZE["tick"], "ytick.labelsize": PAPER_SIZE["tick"], "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "xtick.labelcolor": INK_SOFT, "ytick.labelcolor": INK_SOFT, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0, "legend.fontsize": PAPER_SIZE["legend"], "legend.frameon": False,
        "legend.handlelength": 1.8, "legend.handleheight": 0.8, "legend.columnspacing": 1.4,
        "legend.handletextpad": 0.5, "legend.borderaxespad": 0.0, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.2, "hatch.linewidth": 0.7, "hatch.color": "white",
    }
    PAPER_BAR = dict(edgecolor="white", linewidth=0.5, zorder=3)
    PAPER_HATCH = "//////"
    PAPER_ERR = dict(fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
    PAPER_DASH = (0, (2.2, 1.6))

    def adapted_handle():
        """Legend entry for the hatching that marks adapted probes."""
        return Patch(facecolor=INK_MUTED, edgecolor="white", linewidth=0.5, hatch=PAPER_HATCH)

    def bar_value(ax, x, mean, sd, text, top=1.0):
        """The value of a bar, rotated: above the bar when it fits, otherwise inside it."""
        if mean + sd > 0.75 * top:
            ax.text(x, min(mean / 2, 0.52 * top), text, rotation=90, ha="center", va="center",
                    fontsize=PAPER_SIZE["note"], color=INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
        else:
            ax.text(x, mean + sd + 0.025 * top, text, rotation=90, ha="center", va="bottom",
                    fontsize=PAPER_SIZE["note"], color=INK_SOFT, zorder=5)

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
        PAPER_BAR,
        PAPER_COLOR,
        PAPER_DASH,
        PAPER_ERR,
        PAPER_HATCH,
        PAPER_MODEL_COLOR,
        PAPER_SIZE,
        STYLE_BARS,
        STYLE_JUDGE,
        STYLE_OPERATING,
        STYLE_PAPER,
        STYLE_PARALLEL,
        STYLE_STATES,
        STYLE_SWEEP,
        TEXT_WIDTH,
        adapted_handle,
        axes_at,
        bar_value,
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
    ANALYSIS_SUFFIX,
    BUDGETS,
    CACHE_DIR,
    DATASET_BUILD,
    RUNS_ROOT,
    hashlib,
    json,
    np,
    pd,
    read_onset_labels,
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
        """Detection rate, post-onset coverage, Warn_256, median lead and early share at each budget.

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
                    offsets.append(int(positions[k]) - onset)
            offsets = np.asarray(offsets, float)
            rows.append({
                "budget": threshold.target_negative_fpr, "m": persistence,
                "post_onset": post_hits / post_total, "warning_256": warn_hits / warn_total,
                "lead": float(np.median(offsets)) if offsets.size else float("nan"),
                "early": float((offsets < 0).mean()) if offsets.size else float("nan"),
                "detection": offsets.size / len(scores) if scores else float("nan"),
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
            labels = read_onset_labels(labels_path.parent.parent)
            # The probes' own population: unjudged responses are dropped, and capped responses
            # the judge rules healthy stay in as healthy answers, so the rule raises a false
            # alarm on each of them.
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
        return cached("token_cap_capped_healthy_kept" + ANALYSIS_SUFFIX, str(labels_path.parent), [], compute)

    return (
        discover_runs,
        run_metrics,
        runs_of,
        token_cap_metrics,
        val_response,
    )


@app.cell
def _(
    ANALYSIS_SUFFIX,
    BUDGETS,
    DEPTH_LAYERS,
    DEPTH_RECIPE,
    LAYER,
    METRICS_BUDGET,
    METRICS_SPLIT,
    PERSISTENCE_GRID,
    RUNS_ROOT,
    S4_RECIPE,
    discover_runs,
    np,
    pd,
    read_onset_labels,
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

    # The four scorers of the metrics figure and its depth panel, read from the
    # aggregated evaluations, which hold every depth under the protocol's own alarm
    # convention: the frozen probes ranked first and last by validation AUROC at
    # the read layer, the adapted S4 probe, and the token-cap baseline. The depth
    # panel follows the frozen S4 probe across layers.
    METRICS_LIVE = None
    _table_path = RUNS_ROOT / "analysis" / f"selected_metrics{ANALYSIS_SUFFIX}.csv"
    _cols = {"auc": "rollout_auc", "acc": "best_accuracy", "det": "detection_rate", "post": "in_pattern_recall",
             "warn": "warning_recall_256", "early": "fired_before_frontier"}
    if not _table_path.is_file():
        MISSING.setdefault("metrics", []).append(f"no {_table_path}")
    else:
        _table = pd.read_csv(_table_path)
        _table = _table[_table["model"] == "apertus"]
        _table["detection_rate"] = _table["detected_positives"] / (_table["detected_positives"] + _table["never_fired_positives"])
        _table["scorer_dir"] = [f"{RUNS_ROOT}/{r}/{a}/selected/layer_{int(l):02d}"
                                for r, a, l in zip(_table["run_name"], _table["attempt"], _table["layer"])]
        _pool_path = RUNS_ROOT / "analysis" / f"response_metrics_pool{ANALYSIS_SUFFIX}.csv"
        if _pool_path.is_file():
            _table = _table.merge(pd.read_csv(_pool_path)[["scorer_dir", "split", "best_accuracy"]],
                                  on=["scorer_dir", "split"], how="left")
        else:
            _table["best_accuracy"] = np.nan
            MISSING.setdefault("metrics", []).append("no response_metrics_pool.csv (best accuracy)")
        _at_budget = _table[np.isclose(_table["target_negative_fpr"], METRICS_BUDGET)]
        _val = _at_budget[(_at_budget["split"] == "val") & (_at_budget["layer"] == LAYER) & _at_budget["lora"].eq("none")]
        _rank = _val.groupby("recipe").agg(auc=("rollout_auc", "mean"), seeds=("seed", "nunique"), ok=("selectable", "all"))
        _rank = _rank[(_rank["seeds"] >= 2) & _rank["ok"]].sort_values("auc", ascending=False)
        _test = _at_budget[_at_budget["split"] == METRICS_SPLIT]

        def _stats(rows):
            return {k: (rows[c].mean(), rows[c].std(ddof=1)) for k, c in _cols.items()}

        def _chance_early():
            """Early-warning rate of random alarms at the same budget (scripts/random_alarm_reference.py)."""
            labels = read_onset_labels(RUNS_ROOT.parent / "degeneration-dataset-apertus-8b-instruct")
            labels = labels[~labels["onset_resolution"].isin(["judge_failed", "not_found"])]
            healthy = labels[(labels["split"] == "val") & ~labels["is_positive"].astype(bool)]["num_tokens"].to_numpy(float)
            positives = labels[(labels["split"] == METRICS_SPLIT) & labels["is_positive"].astype(bool)]
            low, high = 0.0, 1.0
            for _ in range(100):
                q = (low + high) / 2
                low, high = (low, q) if np.mean(1 - (1 - q) ** healthy) > METRICS_BUDGET else (q, high)
            alarm = 1 - (1 - low) ** positives["num_tokens"].to_numpy(float)
            return float((1 - (1 - low) ** positives["onset_position"].to_numpy(float)).sum() / alarm.sum())

        _adapted = _test[(_test["layer"] == LAYER) & (_test["recipe"] == "frontier_window W=512 centered K=256 +LoRA")]
        _depth_rows = _test[_test["lora"].eq("none") & (_test["recipe"] == "frontier_window W=512 centered K=256")]
        if len(_rank) < 2 or len(_adapted) < 2 or _depth_rows.empty:
            MISSING.setdefault("metrics", []).append("too few scored seeds for the four scorers or the depth sweep")
        else:
            _cap = token_cap_metrics("apertus")
            _cap_onset = [r for r in _cap["onset"] if np.isclose(r["budget"], METRICS_BUDGET)][0]
            _scorers = [
                ("best\nAUROC", False, _stats(_test[(_test["layer"] == LAYER) & (_test["recipe"] == _rank.index[0])])),
                ("adapted", True, _stats(_adapted)),
                ("token\ncap", False, {"auc": (_cap["test"]["auc"], 0.0), "acc": (_cap["test"]["acc"], 0.0),
                                       "det": (_cap_onset["detection"], 0.0),
                                       "post": (_cap_onset["post_onset"], 0.0), "warn": (_cap_onset["warning_256"], 0.0),
                                       "early": (_cap_onset["early"], 0.0)}),
                ("worst\nAUROC", False, _stats(_test[(_test["layer"] == LAYER) & (_test["recipe"] == _rank.index[-1])])),
            ]
            _scorers.sort(key=lambda item: item[2]["auc"][0], reverse=True)
            # One point every five layers, plus the first and the last, keeps the panel readable.
            _layers = [layer for layer in (1, 5, 10, 15, 20, 25, 31) if layer in set(_depth_rows["layer"])]
            _by_layer = _depth_rows.groupby("layer")
            METRICS_LIVE = {
                "scorers": [item[0] for item in _scorers],
                "hatched": [item[1] for item in _scorers],
                **{k: (np.array([item[2][k][0] for item in _scorers]), np.array([item[2][k][1] for item in _scorers]))
                   for k in _cols},
                "layers": _layers,
                "depth": {k: (_by_layer[c].mean().reindex(_layers).to_numpy(), _by_layer[c].std(ddof=1).reindex(_layers).to_numpy())
                          for k, c in _cols.items()},
                "chance_early": _chance_early(),
                "source": f"best {_rank.index[0]}, worst {_rank.index[-1]}, {METRICS_SPLIT} at {METRICS_BUDGET:.0%}",
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
    INK,
    INK_SOFT,
    INK_MUTED,
    Line2D,
    PAPER_BAR,
    PAPER_COLOR,
    PAPER_DASH,
    PAPER_ERR,
    PAPER_HATCH,
    PAPER_SIZE,
    Patch,
    STYLE_PAPER,
    TEXT_WIDTH,
    adapted_handle,
    axes_at,
    bar_value,
    mpl,
    np,
    page_figure,
    save,
    style,
):
    def metrics_figure(data):
        """Four scorers on the three monitoring tasks (a-c) and the frozen probe across layers (d)."""
        C, size = PAPER_COLOR, PAPER_SIZE
        scorers, hatched, depth = data["scorers"], data["hatched"], data["depth"]
        layers = list(data.get("layers", DEPTH_LAYERS))
        x = np.arange(len(scorers))
        width = 0.38
        with style(**STYLE_PAPER):
            height = 192.0
            fig = page_figure(TEXT_WIDTH, height)
            y0, h = 40.0, 100.0
            axa = axes_at(fig, 22.0, y0, 84.0, h)
            axb = axes_at(fig, 110.0, y0, 84.0, h, sharey=axa)
            axc = axes_at(fig, 198.0, y0, 84.0, h, sharey=axa)
            axu = axes_at(fig, 310.0, y0 + 62.0, 85.0, h - 62.0)
            axd = axes_at(fig, 310.0, y0, 85.0, 56.0)

            def task_panel(ax, keys, title, digits):
                for k, key in enumerate(keys):
                    xs = x + (k - (len(keys) - 1) / 2) * width
                    ax.bar(xs, data[key][0], width, color=C[key], hatch=[PAPER_HATCH if hh else "" for hh in hatched],
                           **PAPER_BAR)
                    ax.errorbar(xs, data[key][0], yerr=data[key][1], **PAPER_ERR)
                    for xi, m, s in zip(xs, *data[key]):
                        bar_value(ax, xi, m, s, f"{m:.{digits}f}")
                ax.set_xticks(x, scorers)
                ax.tick_params(axis="x", length=0, labelsize=size["note"])
                ax.set_xlim(-0.6, len(scorers) - 0.4)
                ax.set_ylim(0, 1.08)
                ax.set_yticks([0, 0.5, 1.0])
                ax.grid(axis="y")
                ax.set_axisbelow(True)
                ax.set_title(title)

            fd_keys = ["det", "post"] if "det" in data else ["post"]
            task_panel(axa, ["auc", "acc"], "(a) response\nclassification (↑)", 4)
            task_panel(axb, fd_keys, "(b) failure\ndetection (↑)", 2)
            task_panel(axc, ["warn", "early"], "(c) early\nwarning (↑)", 2)
            for ax in (axb, axc):
                ax.tick_params(axis="y", labelleft=False)
            if "chance_early" in data:
                for xi in x + 0.5 * width:
                    axc.plot([xi - 0.25, xi + 0.25], [data["chance_early"]] * 2, color=INK, lw=0.9, ls=PAPER_DASH,
                             zorder=6)
            # One arrow under the three task panels, pointing from the highest to the lowest AUROC.
            fig.text(152.0 / TEXT_WIDTH, 9.0 / height, "scorers ordered by AUROC, highest first", ha="center",
                     va="bottom", fontsize=size["note"], color=INK_MUTED)
            fig.add_artist(mpl.patches.FancyArrowPatch((22.0 / TEXT_WIDTH, 5.0 / height), (282.0 / TEXT_WIDTH, 5.0 / height),
                                                       transform=fig.transFigure, arrowstyle="->", mutation_scale=7,
                                                       lw=0.8, color=INK_MUTED))

            # (d) across layers: a zoom on the response-level metrics above, every metric on a log scale below,
            # with each metric's best-to-worst ratio in a strip right of the last layer.
            line = {"auc": dict(marker="s", ms=2.6), "acc": dict(ls="--", marker="o", ms=2.4, mfc="white", mew=0.8),
                    "post": dict(marker="^", ms=3.0), "early": dict(marker="D", ms=2.6), "warn": dict(marker="o", ms=2.8)}
            gap = 0.085 * (layers[-1] - layers[0])
            arrow_x = {"warn": layers[-1] + 1.3 * gap, "early": layers[-1] + 2.5 * gap, "post": layers[-1] + 3.7 * gap}
            xlim = (layers[0] - 2, layers[-1] + 4.6 * gap)
            # Flat caps keep the shortest range bar (early-warning rate, about 1.4x) readable.
            arrow = dict(arrowstyle="|-|,widthA=0.3,widthB=0.3", lw=0.9, mutation_scale=6, shrinkA=0, shrinkB=0)

            def lines(ax, keys):
                for key in keys:
                    m, s = (np.asarray(v, float) for v in depth[key])
                    ax.fill_between(layers, m - s, m + s, color=C[key], alpha=0.2, lw=0)
                    ax.plot(layers, m, color=C[key], lw=1.1, **line[key])

            lines(axu, ["auc", "acc"])
            lowest = min(float(np.min(np.asarray(depth[k][0]) - np.asarray(depth[k][1]))) for k in ("auc", "acc"))
            axu.set_ylim(min(0.955, lowest - 0.01), 1.003)
            axu.set_yticks([0.97, 1.0])
            axu.set_xlim(*xlim)
            axu.set_xticks(layers)
            axu.tick_params(axis="x", labelbottom=False)
            axu.grid(axis="y")
            for key, xa in (("auc", arrow_x["early"]), ("acc", arrow_x["post"])):
                m = depth[key][0]
                axu.annotate("", xy=(xa, min(m)), xytext=(xa, max(m)), arrowprops=dict(arrow, color=C[key]))
            a, c = depth["auc"][0], depth["acc"][0]
            for k, text in enumerate((f"AUROC {max(a) / min(a):.3f}×", f"best accuracy {max(c) / min(c):.3f}×")):
                axu.text((layers[0] + layers[-1]) / 2 + 3.5, 0.47 - 0.25 * k, text, transform=axu.get_xaxis_transform(),
                         fontsize=size["note"], color=INK_SOFT, ha="center", va="center")
            axu.set_title("(d) across\nlayers (↑)")

            lines(axd, ["auc", "acc", "post", "early", "warn"])
            axd.set_yscale("log")
            axd.set_ylim(0.5 * float(np.min(np.asarray(depth["warn"][0]) - np.asarray(depth["warn"][1]))), 2.2)
            axd.set_xlim(*xlim)
            axd.set_xticks(layers, [str(layer) for layer in layers])
            axd.yaxis.set_minor_locator(mpl.ticker.NullLocator())
            axd.grid(axis="y")
            axd.set_xlabel("layer")
            for key, xa in arrow_x.items():
                m = depth[key][0]
                axd.annotate("", xy=(xa, min(m)), xytext=(xa, max(m)), arrowprops=dict(arrow, color=C[key]))
                r = max(m) / min(m)
                axd.text(xa, min(m) / 1.12, f"{r:.0f}×" if r >= 9.95 else f"{r:.1f}×", fontsize=size["note"],
                         color=C[key], ha="center", va="top")

            entries = [(Patch(color=C["auc"]), "AUROC"), (Patch(color=C["acc"]), "best accuracy")]
            if "det" in data:
                entries.append((Patch(color=C["det"]), "Detection"))
            entries += [(Patch(color=C["post"]), "Post"), (Patch(color=C["warn"]), r"$\mathrm{Warn}_{256}$"),
                        (Patch(color=C["early"]), "Early")]
            if "chance_early" in data:
                entries.append((Line2D([], [], color=INK, lw=0.9, ls=PAPER_DASH), "random alarms"))
            entries.append((adapted_handle(), "adapted"))
            fig.legend(*zip(*entries), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, columnspacing=1.6)
            save(fig, FIG_DIR / "fig_metrics.pdf")
            save(fig, FIG_DIR / "fig_metrics.png", dpi=300)
        return fig

    metrics_figure(DATA["metrics"][0])
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
def _(
    FIG_DIR,
    GRID,
    INK,
    INK_MUTED,
    INK_SOFT,
    Line2D,
    PAPER_BAR,
    PAPER_COLOR,
    PAPER_DASH,
    PAPER_ERR,
    PAPER_HATCH,
    PAPER_MODEL_COLOR,
    PAPER_SIZE,
    Patch,
    RUNS_ROOT,
    STYLE_BARS,
    STYLE_PAPER,
    TEXT_WIDTH,
    ANALYSIS_SUFFIX,
    adapted_handle,
    axes_at,
    bar_value,
    mpl,
    np,
    page_figure,
    pd,
    pe,
    read_onset_labels,
    save,
    style,
):
    # Figures of the revised paper's Q2 (early warning per model) and Q4 (where the
    # flags fall), read from the aggregated evaluations and the onset profile.
    REB_PROBES = {
        "apertus": [("frozen", 15), ("adapted", 15), ("adapted", 30)],
        "llama3p1": [("frozen", 15), ("adapted", 15), ("adapted", 6)],
        "mistral": [("frozen", 5), ("adapted", 5), ("adapted", 11)],
    }
    REB_DATASET = {"apertus": "apertus-8b-instruct", "llama3p1": "llama3p1-8b-instruct", "mistral": "mistral-7b-instruct-v0p1"}
    REB_LABEL = {"apertus": "Apertus", "llama3p1": "Llama", "mistral": "Mistral"}
    REB_COLOR = {"apertus": "#d62728", "llama3p1": "#1f77b4", "mistral": "#e69f00"}
    REB_SPLIT, REB_BUDGET = "test_indomain", 0.05

    _table = pd.read_csv(RUNS_ROOT / "analysis" / f"selected_metrics{ANALYSIS_SUFFIX}.csv")
    _s4 = _table[_table["recipe"].str.startswith("frontier_window W=512 centered K=256")].copy()
    _s4["regime"] = np.where(_s4["lora"].eq("none"), "frozen", "adapted")
    _keys = {"post": "in_pattern_recall", "warn": "warning_recall_256", "early": "fired_before_frontier", "lead": "median_offset",
             "pre": "preonset_coverage_256"}

    def reb_stats(model, regime, layer, split=REB_SPLIT, budget=REB_BUDGET):
        rows = _s4[(_s4["model"] == model) & (_s4["regime"] == regime) & (_s4["layer"] == layer)
                   & (_s4["split"] == split) & np.isclose(_s4["target_negative_fpr"], budget)]
        return {k: (rows[c].mean(), rows[c].std()) for k, c in _keys.items()}

    def chance_early(model, split=REB_SPLIT, budget=REB_BUDGET):
        """Early-warning rate of alarms placed at random under the same budget (scripts/random_alarm_reference.py)."""
        labels = read_onset_labels(RUNS_ROOT.parent / f"degeneration-dataset-{REB_DATASET[model]}")
        labels = labels[~labels["onset_resolution"].isin(["judge_failed", "not_found"])]
        healthy = labels[(labels["split"] == "val") & ~labels["is_positive"].astype(bool)]["num_tokens"].to_numpy(float)
        test = labels[(labels["split"] == split) & labels["is_positive"].astype(bool)]
        low, high = 0.0, 1.0
        for _ in range(100):
            q = (low + high) / 2
            low, high = (low, q) if np.mean(1 - (1 - q) ** healthy) > budget else (q, high)
        alarm = 1 - (1 - low) ** test["num_tokens"].to_numpy(float)
        return float((1 - (1 - low) ** test["onset_position"].to_numpy(float)).sum() / alarm.sum())

    def q2_figure():
        """Post, Warn_256 (with Pre), Early and Lead of the frozen probe and the adapted probe at its own layer."""
        C, size = PAPER_COLOR, PAPER_SIZE
        models = list(REB_PROBES)
        probes = {m: [REB_PROBES[m][0], REB_PROBES[m][2]] for m in models}
        panels = [("post", "(a) Post (↑)"), ("warn", r"(b) $\mathrm{Warn}_{256}$ (↑)"), ("early", "(c) Early (↑)"),
                  ("lead", "(d) Lead (↓)")]
        pre_effect = [pe.Stroke(linewidth=2.6, foreground=INK), pe.Normal()]
        with style(**STYLE_PAPER):
            fig = page_figure(TEXT_WIDTH, 142.0)
            y0, h = 18.0, 98.0
            axes = [axes_at(fig, 30.0, y0, 80.0, h)]
            axes += [axes_at(fig, x, y0, 80.0, h, sharey=axes[0]) for x in (114.0, 198.0)]
            axes += [axes_at(fig, 314.0, y0, 80.0, h)]
            offset, width = 0.2, 0.36
            for ax, (key, title) in zip(axes, panels):
                values = []
                for mi, model in enumerate(models):
                    for pi, (regime, layer) in enumerate(probes[model]):
                        stats = reb_stats(model, regime, layer)
                        mean, sd = stats[key]
                        xb = mi + (2 * pi - 1) * offset
                        ax.bar(xb, mean, width, color=C[key], hatch=PAPER_HATCH if regime == "adapted" else "", **PAPER_BAR)
                        ax.errorbar(xb, mean, yerr=sd, **PAPER_ERR)
                        values.append((xb, mean, sd))
                        if key == "warn":
                            # Coverage of the text before the warning window, as a tick on the bar.
                            ax.plot([xb - 0.45 * width, xb + 0.45 * width], [stats["pre"][0]] * 2, color=C["pre"],
                                    lw=1.6, solid_capstyle="butt", zorder=6, path_effects=pre_effect)
                    if key == "early":
                        ax.plot([mi - 0.42, mi + 0.42], [chance_early(model)] * 2, color=INK, lw=0.9, ls=PAPER_DASH,
                                zorder=6)
                for xb, mean, sd in values:
                    if key == "lead":
                        va, y = ("top", mean - sd - 8) if mean < 0 else ("bottom", mean + sd + 8)
                        ax.text(xb, y, f"{mean:+.0f}".replace("-", "−"), rotation=90, ha="center", va=va,
                                fontsize=size["note"], color=INK_SOFT)
                    else:
                        bar_value(ax, xb, mean, sd, f"{mean:.2f}")
                ax.set_title(title)
                ax.set_xticks(range(len(models)), [REB_LABEL[m] for m in models])
                ax.tick_params(axis="x", length=0)
                ax.set_xlim(-0.5, len(models) - 0.5)
                ax.grid(axis="y")
                ax.set_axisbelow(True)
            axes[0].set_ylim(0, 1.08)
            axes[0].set_yticks([0, 0.5, 1.0])
            axes[0].set_ylabel("share")
            for ax in axes[1:3]:
                ax.tick_params(axis="y", labelleft=False)
            axes[3].axhline(0, color=INK_MUTED, lw=0.6)
            axes[3].set_ylim(-420, 270)
            axes[3].set_yticks([-400, -200, 0, 200])
            axes[3].set_ylabel("tokens")
            handles = [Patch(color=INK_MUTED), adapted_handle(),
                       Line2D([], [], color=C["pre"], lw=1.6, solid_capstyle="butt", path_effects=pre_effect),
                       Line2D([], [], color=INK, lw=0.9, ls=PAPER_DASH)]
            fig.legend(handles, ["frozen", "adapted", "Pre", "random alarms"], loc="upper center",
                       bbox_to_anchor=(0.5, 1.0), ncol=4, columnspacing=2.0)
            save(fig, FIG_DIR / "fig_q2_warning_per_model.pdf")
            save(fig, FIG_DIR / "fig_q2_warning_per_model.png", dpi=300)
        return fig

    POSITION_TABLES = RUNS_ROOT / "analysis" / f"position_analysis{ANALYSIS_SUFFIX}"

    def q4_figure():
        """Flag rate on degenerate responses by distance from the onset (top) and on healthy responses by
        absolute position (bottom), on one y scale. Colour marks the model, line style the probe."""
        profile = pd.read_csv(POSITION_TABLES / "onset_profile.csv")
        healthy = pd.read_csv(POSITION_TABLES / "healthy_position.csv")
        healthy = healthy[healthy["scope"] == "all"]
        width = int(np.diff(np.unique(healthy["lo"])).min())
        styles = [dict(ls="-", lw=1.2), dict(ls=(0, (3.5, 1.4)), lw=1.1), dict(ls=(0, (1, 1.1)), lw=1.3)]
        far_span = (-1250, -1090)
        with style(**STYLE_PAPER):
            fig = page_figure(TEXT_WIDTH, 262.0)
            xs = [38.0 + i * 121.0 for i in range(3)]
            top = [axes_at(fig, x, 150.0, 104.0, 82.0) for x in xs]
            bottom = [axes_at(fig, x, 54.0, 104.0, 70.0, sharey=top[0]) for x in xs]
            strips = [axes_at(fig, x, 26.0, 104.0, 22.0, sharex=bottom[i]) for i, x in enumerate(xs)]
            for ax, low, strip, model in zip(top, bottom, strips, REB_PROBES):
                probes = REB_PROBES[model]
                color = PAPER_MODEL_COLOR[model]
                ax.axvspan(-256, 0, color=PAPER_COLOR["warn"], alpha=0.25, lw=0, zorder=0)
                ax.axvline(0, color=INK, lw=0.6, zorder=1)
                ax.axvline(-1050, color=GRID, lw=0.8, zorder=1)
                for (regime, layer), st in zip(probes, styles):
                    name = f"{model} {regime} L{layer}"
                    part = profile[profile["scorer"] == name]
                    before = part[(part["side"] == "before") & (part["lo"] < 1024)].sort_values("lo", ascending=False)
                    after = part[part["side"] == "after"].sort_values("lo")
                    curve = pd.concat([before, after])
                    x = np.concatenate([-(before["lo"].to_numpy() + 16), after["lo"].to_numpy() + 16])
                    ax.fill_between(x, curve["ci_low"], curve["ci_high"], color=color, alpha=0.15, lw=0, zorder=2)
                    ax.plot(x, curve["flag_rate"], color=color, zorder=3, **st)
                    # Every position further than 1,024 tokens before the onset, pooled into one bin.
                    far = part[(part["side"] == "before") & (part["lo"] == 1024)].iloc[0]
                    ax.fill_between(far_span, far["ci_low"], far["ci_high"], color=color, alpha=0.15, lw=0, zorder=2)
                    ax.plot(far_span, [far["flag_rate"]] * 2, color=color, zorder=3, **st)
                    rows = healthy[healthy["scorer"] == name].sort_values("lo")
                    hx = rows["lo"].to_numpy() + width / 2
                    low.fill_between(hx, rows["ci_low"], rows["ci_high"], color=color, alpha=0.15, lw=0, zorder=2)
                    low.plot(hx, rows["flag_rate"], color=color, zorder=3, **st)
                ax.set_title(f"{REB_LABEL[model]}, layers " + " / ".join(str(layer) for _, layer in probes))
                ax.set_xlim(-1290, 290)
                ax.set_xticks([-1170, -512, -256, 0, 256], ["<−1,024", "−512", "−256", "0", "256"])
                ax.set_xlabel("tokens from the onset")
                # How many healthy responses are still running at each position, on a log scale.
                running = healthy[healthy["scorer"] == f"{model} {probes[0][0]} L{probes[0][1]}"].sort_values("lo")
                strip.bar(running["lo"], running["responses"], width=width, align="edge", color=INK_MUTED, alpha=0.45,
                          edgecolor="white", linewidth=0.4)
                for (_, row), align in zip(running.iloc[[0, -1]].iterrows(), ("left", "center")):
                    xt = row["lo"] + (0.15 if align == "left" else 0.5) * width
                    strip.text(xt, row["responses"] * 1.3, f"{int(row['responses']):,}", ha=align, va="bottom",
                               fontsize=PAPER_SIZE["note"], color=INK_SOFT)
                strip.set_yscale("log")
                strip.set_ylim(10, 4e4)
                strip.set_yticks([100, 1000], ["100", "1k"])
                strip.yaxis.set_minor_locator(mpl.ticker.NullLocator())
                last = int(healthy["lo"].max()) + width
                strip.set_xlim(0, last)
                strip.set_xticks(range(0, last + 1, 1024), [f"{t:,}" for t in range(0, last + 1, 1024)])
                strip.set_xticks(range(512, last, 1024), minor=True)
                strip.tick_params(axis="x", which="minor", length=1.5, width=0.6)
                strip.set_xlabel("position in the response")
                low.tick_params(axis="x", labelbottom=False)
                for a in (ax, low):
                    a.set_ylim(0, 1.0)
                    a.set_yticks([0, 0.5, 1.0])
                    a.grid(axis="y")
                    a.set_axisbelow(True)
            top[0].set_ylabel("degenerate responses\nshare flagged")
            bottom[0].set_ylabel("healthy responses\nshare flagged")
            strips[0].set_ylabel("running", fontsize=PAPER_SIZE["note"])
            for a in top[1:] + bottom[1:] + strips[1:]:
                a.tick_params(axis="y", labelleft=False)
            handles = [Line2D([], [], color=INK_SOFT, **st) for st in styles]
            handles += [Patch(color=PAPER_COLOR["warn"], alpha=0.25), Line2D([], [], color=INK, lw=0.6)]
            fig.legend(handles, ["frozen", "adapted, same layer", "adapted, own layer", "warning window", "onset"],
                       loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, handlelength=2.4)
            save(fig, FIG_DIR / "fig_q4_onset_profile.pdf")
            save(fig, FIG_DIR / "fig_q4_onset_profile.png", dpi=300)
        return fig

    def false_alarm_figure():
        """Share of healthy tokens flagged by absolute position, over the healthy answers still running there."""
        profile = pd.read_csv(POSITION_TABLES / "healthy_position.csv")
        profile = profile[profile["scope"] == "all"]
        styles = [dict(ls="-", lw=1.5), dict(ls=(0, (4, 1.6)), lw=1.4), dict(ls=(0, (1, 1.3)), lw=1.6)]
        width = int(np.diff(np.unique(profile["lo"])).min())
        top = min(1.0, 1.08 * profile["ci_high"].max())
        with style(**STYLE_BARS):
            fig = page_figure(666.72, 196.0)
            axes, strips = [], []
            for i in range(3):
                axes.append(axes_at(fig, 45.0 + i * 210.0, 76.0, 190.0, 92.0))
                strips.append(axes_at(fig, 45.0 + i * 210.0, 38.0, 190.0, 26.0, sharex=axes[-1]))
            for ax, strip, model in zip(axes, strips, REB_PROBES):
                for (regime, layer), st in zip(REB_PROBES[model], styles):
                    part = profile[profile["scorer"] == f"{model} {regime} L{layer}"].sort_values("lo")
                    x = part["lo"].to_numpy() + width / 2
                    ax.fill_between(x, part["ci_low"], part["ci_high"], color=REB_COLOR[model], alpha=0.13, lw=0, zorder=2)
                    ax.plot(x, part["flag_rate"], color=REB_COLOR[model], label=f"{regime}, L{layer}", zorder=3, **st)
                running = profile[profile["scorer"] == f"{model} {REB_PROBES[model][0][0]} L{REB_PROBES[model][0][1]}"].sort_values("lo")
                strip.bar(running["lo"], running["responses"], width=width, align="edge", color=INK_MUTED, alpha=0.45,
                          edgecolor="white", linewidth=0.5)
                for (_, row), align in zip(running.iloc[[0, -1]].iterrows(), ("left", "center")):
                    x = row["lo"] + (0.15 if align == "left" else 0.5) * width
                    strip.text(x, row["responses"] * 1.25, f"{int(row['responses']):,}", ha=align, va="bottom", fontsize=6.3,
                               color=INK_SOFT)
                ax.set_title(REB_LABEL[model], loc="center", pad=4)
                ax.set_ylim(0, top)
                ax.grid(axis="y")
                ax.set_axisbelow(True)
                ax.tick_params(axis="x", labelbottom=False)
                ax.legend(loc="upper left", handlelength=2.6)
                strip.set_yscale("log")
                strip.set_ylim(10, 2.5e4)
                strip.set_yticks([100, 1000], ["100", "1k"])
                strip.minorticks_off()
                strip.set_xlim(0, running["lo"].max() + width)
                strip.set_xticks(range(0, int(running["lo"].max() + width) + 1, 512))
                strip.set_xlabel("token position in the healthy answer")
            axes[0].set_ylabel("share of tokens flagged")
            strips[0].set_ylabel("answers\nrunning", fontsize=7.5)
            for ax in axes[1:] + strips[1:]:
                ax.tick_params(axis="y", labelleft=False)
            save(fig, FIG_DIR / "fig_false_alarm_profile.pdf")
            save(fig, FIG_DIR / "fig_false_alarm_profile.png", dpi=300)
        return fig

    HUMAN_LABELS = RUNS_ROOT.parent / "human_review" / "labels"
    # Bins of the signed onset error, log-spaced and the same on both sides of zero. The last
    # edge is past the token cap, so the outer bin holds every error of 1,024 tokens or more.
    ONSET_ERROR_EDGES = [1, 64, 256, 1024, 4097]

    def judge_human_onsets(model):
        """Every capped test response the judge ruled on, with the judge's and the reviewer's verdict and onset."""
        labels = pd.read_parquet(RUNS_ROOT.parent / f"degeneration-dataset-{REB_DATASET[model]}" / "onset_labels" / "onset_labels.parquet")
        labels["id"] = labels["prompt_id"] + "::" + labels["rollout_idx"].astype(str)
        judge = labels[["id", "onset_resolution", "onset_position"]].rename(columns={"onset_position": "judge_onset"})
        frame = pd.read_csv(HUMAN_LABELS / f"labels_{REB_DATASET[model]}.csv").merge(judge, on="id")
        # Responses the judge failed on are left out of every split, so they are left out here too.
        frame = frame[frame["onset_resolution"].isin(["ok", "not_degenerating"]) & ~frame["drop"]].copy()
        # Positive when the judge places the onset later than the reviewer; defined where both call it degenerate.
        frame["error"] = frame["judge_onset"] - frame["onset_position"]
        return frame

    def judge_onset_table():
        """Verdict and onset agreement between the judge and the reviewers, per model."""
        rows = []
        for model in REB_PROBES:
            frame = judge_human_onsets(model)
            judged = frame[frame["judge_positive"]]
            error = judged.loc[judged["is_positive"], "error"]
            rows.append({
                "model": REB_LABEL[model], "ruled on": len(frame), "judge degenerate": len(judged),
                "confirmed": int(judged["is_positive"].sum()), "judge healthy": int((~frame["judge_positive"]).sum()),
                "judge healthy, reviewer degenerate": int((~frame["judge_positive"] & frame["is_positive"]).sum()),
                "same onset": int((error == 0).sum()), "|error| < 64": int((error.abs() < 64).sum()),
                "|error| >= 256": int((error.abs() >= 256).sum()), "judge later by >= 256": int((error >= 256).sum()),
                "median |error| of moved onsets": float(error[error != 0].abs().median()),
            })
        return pd.DataFrame(rows)

    def judge_onset_figure():
        """Signed distance from the reviewer's onset to the judge's, on the responses both call degenerate."""
        edges, size = ONSET_ERROR_EDGES, PAPER_SIZE
        bins, width, top = len(edges) - 1, 0.26, 12.0
        with style(**STYLE_PAPER):
            fig = page_figure(TEXT_WIDTH, 132.0)
            ax = axes_at(fig, 34.0, 28.0, 350.0, 84.0)
            handles, names = [], []
            for mi, model in enumerate(REB_PROBES):
                color = PAPER_MODEL_COLOR[model]
                frame = judge_human_onsets(model)
                error = frame.loc[frame["judge_positive"] & frame["is_positive"], "error"].to_numpy()
                for side in (-1, 1):
                    counts = np.array([np.sum((side * error >= lo) & (side * error < hi)) for lo, hi in zip(edges[:-1], edges[1:])])
                    xb = side * np.arange(1, bins + 1) + (mi - 1) * width
                    for x, n in zip(xb, counts):
                        if n:
                            share = 100 * n / len(error)
                            ax.bar(x, share, width, color=color, **PAPER_BAR)
                            bar_value(ax, x, share, 0.0, f"{share:.1f}", top=top)
                # Responses whose onset the reviewer kept, in the middle slot, so one model's numbers add up to 100.
                ax.text(0, top - 2.3 - 1.5 * mi, f"{100 * np.mean(error == 0):.1f}", ha="center", va="top",
                        fontsize=size["label"], color=color)
                handles.append(Patch(color=color))
                names.append(f"{REB_LABEL[model]} ($n={len(error)}$)")
            ax.text(0, top - 0.5, "same onset", ha="center", va="top", fontsize=size["note"], color=INK_SOFT)
            ax.text(-bins - 0.4, top - 0.5, "← judge onset earlier", ha="left", va="top", fontsize=size["note"], color=INK_SOFT)
            ax.text(bins + 0.4, top - 0.5, "judge onset later →", ha="right", va="top", fontsize=size["note"], color=INK_SOFT)
            ticks = np.arange(1, bins + 1) + 0.5
            labels = [f"{e:,}" for e in edges[1:-1]] + ["4,096"]
            ax.set_xticks(np.concatenate([-ticks[::-1], [0], ticks]),
                          ["−" + label for label in labels[::-1]] + ["0"] + labels)
            ax.set_xlim(-bins - 0.5, bins + 0.5)
            ax.set_xlabel("judge onset − human onset (tokens)")
            ax.set_ylim(0, top)
            ax.set_yticks([0, 5, 10])
            ax.set_ylabel("responses (%)")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=2.0)
            save(fig, FIG_DIR / "fig_judge_onset_error.pdf")
            save(fig, FIG_DIR / "fig_judge_onset_error.png", dpi=300)
        return fig

    q2_figure()
    q4_figure()
    false_alarm_figure()
    judge_onset_figure()
    return (judge_onset_table,)


@app.cell
def _(judge_onset_table):
    judge_onset_table()
    return



@app.cell
def _(FIG_DIR, RUNS_ROOT, STYLE_BARS, axes_at, np, page_figure, pd, save, style):
    # PCA of the residual stream at the frozen probe's layer, frozen against adapted, on
    # degenerate and long healthy test responses (scripts/extract_pca_states.py). One PCA
    # per model and regime, fitted on an equal number of tokens from each group so the
    # long post-onset stretches do not dominate it. Separation is measured in the plane of
    # the first two components: the distance between two group centroids over the
    # root-mean-square spread of the two groups around them. Tokens whose norm exceeds five
    # times the median norm of their response are left out: a few such tokens (the large,
    # nearly constant dimensions of early Mistral layers) otherwise take the first component.
    from safetensors.torch import load_file as _load_file
    from sklearn.decomposition import PCA as _PCA

    _root = RUNS_ROOT / "paper_figures" / "pca_states"
    _models = {"apertus": ("Apertus", 15), "llama3p1": ("Llama", 15), "mistral": ("Mistral", 5)}
    _groups = {"healthy": "#9a9a9a", "pre": "#2ca02c", "warn": "#ff7f0e", "post": "#d62728"}
    _names = {"healthy": "healthy", "pre": "degenerate, before the window", "warn": "warning window", "post": "after the onset"}
    _stride, _per_group, _rng = 4, 3000, np.random.default_rng(0)

    def _tokens(model, regime, dropped):
        manifest = pd.read_parquet(_root / model / "manifest.parquet")
        parts = {g: [] for g in _groups}
        for row in manifest.itertuples():
            path = _root / model / regime / row.domain / row.prompt_id / f"rollout_{row.rollout_idx}.safetensors"
            hidden = _load_file(str(path))["hidden_states"].float().numpy()
            position = np.arange(len(hidden))
            norm = np.linalg.norm(hidden, axis=1)
            typical = norm <= 5 * np.median(norm)
            dropped[0] += int((~typical).sum())
            dropped[1] += len(hidden)
            if row.is_positive:
                onset = int(row.onset_position)
                masks = {"pre": position < onset - 256, "warn": (position >= onset - 256) & (position < onset), "post": position >= onset}
            else:
                masks = {"healthy": np.ones(len(hidden), bool)}
            for group, mask in masks.items():
                parts[group].append(hidden[mask & typical][::_stride])
        return {g: np.concatenate(v) for g, v in parts.items() if v}

    def _separation(a, b):
        spread = np.sqrt((((a - a.mean(0)) ** 2).sum(1).mean() + ((b - b.mean(0)) ** 2).sum(1).mean()) / 2)
        return float(np.linalg.norm(a.mean(0) - b.mean(0)) / spread)

    _rows, _projected = [], {}
    for _model in _models:
        if not (_root / _model / "manifest.parquet").is_file():
            continue
        for _regime in ("frozen", "adapted"):
            _dropped = [0, 0]
            _tok = _tokens(_model, _regime, _dropped)
            _fit = np.concatenate([g[_rng.choice(len(g), min(_per_group, len(g)), replace=False)] for g in _tok.values()])
            _pca = _PCA(n_components=2, random_state=0).fit(_fit)
            _proj = {g: _pca.transform(v) for g, v in _tok.items()}
            _projected[_model, _regime] = _proj
            _rows.append({"model": _model, "regime": _regime, "pc1_var": _pca.explained_variance_ratio_[0],
                          "pc2_var": _pca.explained_variance_ratio_[1],
                          "pre_vs_healthy": _separation(_proj["pre"], _proj["healthy"]),
                          "warn_vs_pre": _separation(_proj["warn"], _proj["pre"]),
                          "post_vs_healthy": _separation(_proj["post"], _proj["healthy"]),
                          "dropped_share": _dropped[0] / _dropped[1]})
    PCA_SUMMARY = pd.DataFrame(_rows)
    if len(PCA_SUMMARY):
        PCA_SUMMARY.to_csv(RUNS_ROOT / "analysis" / "pca_states_summary.csv", index=False)

    if _projected:
        _shown = [m for m in _models if (m, "frozen") in _projected]
        with style(**STYLE_BARS):
            _fig = page_figure(430.0, 40.0 + 150.0 * len(_shown))
            for _i, _model in enumerate(_shown):
                for _j, _regime in enumerate(("frozen", "adapted")):
                    _ax = axes_at(_fig, 40.0 + _j * 200.0, 40.0 + (len(_shown) - 1 - _i) * 150.0, 170.0, 115.0)
                    # Same number of points and size for every group, drawn so that the
                    # groups compared with each other (healthy, before the window) stay visible.
                    for _group in ("post", "warn", "pre", "healthy"):
                        _pts = _projected[_model, _regime][_group]
                        _keep = _rng.choice(len(_pts), min(800, len(_pts)), replace=False)
                        _ax.scatter(_pts[_keep, 0], _pts[_keep, 1], s=3, c=_groups[_group], alpha=0.45, lw=0,
                                    label=_names[_group], rasterized=True)
                    _ax.set_xticks([])
                    _ax.set_yticks([])
                    _ax.set_title(f"{_models[_model][0]}, {_regime}, layer {_models[_model][1]}", loc="left", pad=3)
            _fig.axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.12), ncol=4, markerscale=3, frameon=False,
                                handletextpad=0.3, columnspacing=1.0)
            save(_fig, FIG_DIR / "fig_pca_states.pdf")
            save(_fig, FIG_DIR / "fig_pca_states.png", dpi=300)
    PCA_SUMMARY.round(3)
    return (PCA_SUMMARY,)


@app.cell
def _(
    CACHE_DIR,
    FIG_DIR,
    GRID,
    INK_SOFT,
    Line2D,
    PAPER_COLOR,
    PAPER_SIZE,
    RUNS_ROOT,
    STYLE_PAPER,
    TEXT_WIDTH,
    axes_at,
    np,
    page_figure,
    pd,
    save,
    style,
):
    # One degenerate test response per model, projected on the first two principal
    # components of its own residual stream: frozen at the frozen probe's layer, adapted at
    # that layer, and adapted at the adapted probe's own selected layer, with its own
    # adapters (scripts/extract_pca_states.py --rollouts ... --extra-adapted-layer ...).
    # Each panel has its own PCA over every token of the response, except tokens whose norm
    # exceeds five times the median norm (the large, nearly constant dimensions of early
    # Mistral layers). The responses were chosen among human-verified in-domain test
    # responses on which both adapted probes cover the warning window and the frozen probe
    # does not. Separation is the balanced accuracy of the best linear boundary in the plane
    # between the tokens after the onset and the tokens before the warning window.
    from safetensors.torch import load_file as _load_file
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA

    _root = RUNS_ROOT / "paper_figures" / "pca_rollouts"
    # model: (label, prompt, rollout, frozen probe's layer, adapted probe's own layer)
    _rollouts = {"apertus": ("Apertus", "numinamath_1_5_00388", 2, 15, 30),
                 "llama3p1": ("Llama", "numinamath_1_5_00044", 4, 15, 6),
                 "mistral": ("Mistral", "numinamath_1_5_00517", 9, 5, 11)}
    _names = {"pre": "before the warning window", "warn": "warning window", "post": "after the onset"}
    _columns = ("frozen", "adapted, same layer", "adapted, own layer")
    _window = 256

    def _project(path):
        """The first two principal components of one response's states, the positions kept, and their variance shares."""
        key = "_".join(path.relative_to(_root).with_suffix("").parts)
        cache = CACHE_DIR / f"pca_rollout_{key}_{path.stat().st_mtime_ns}.npz"
        if cache.is_file():
            stored = np.load(cache)
            return stored["z"], stored["position"], stored["variance"]
        hidden = _load_file(str(path))["hidden_states"].double().numpy()
        norm = np.linalg.norm(hidden, axis=1)
        position = np.flatnonzero(norm <= 5 * np.median(norm))
        centred = hidden[position] - hidden[position].mean(0)
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        z, variance = centred @ vt[:2].T, (singular ** 2 / (singular ** 2).sum())[:2]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, z=z, position=position, variance=variance)
        return z, position, variance

    _panels, _rows = {}, []
    for _model, (_label, _prompt, _k, _layer, _own) in _rollouts.items():
        if not (_root / _model / "manifest.parquet").is_file():
            continue
        _row = pd.read_parquet(_root / _model / "manifest.parquet").set_index(["prompt_id", "rollout_idx"]).loc[(_prompt, _k)]
        _onset = int(_row["onset_position"])
        for _j, (_condition, _read) in enumerate((("frozen", _layer), ("adapted", _layer), (f"adapted_L{_own}", _own))):
            _z, _pos, _var = _project(_root / _model / _condition / _row["domain"] / _prompt / f"rollout_{_k}.safetensors")
            _groups = {"pre": _pos < _onset - _window, "warn": (_pos >= _onset - _window) & (_pos < _onset), "post": _pos >= _onset}
            _panels[_model, _j] = (_z, _groups, _read)
            _ends = _groups["pre"] | _groups["post"]
            _y = _groups["post"][_ends].astype(int)
            _pred = _LDA().fit(_z[_ends], _y).predict(_z[_ends])
            _rows.append({"model": _model, "regime": _columns[_j], "layer": _read, "prompt": f"{_prompt} r{_k}", "onset": _onset,
                          "tokens_left_out": int(_row["num_tokens"]) - len(_pos), "pc1_var": _var[0], "pc2_var": _var[1],
                          "separation": 0.5 * ((_pred[_y == 1] == 1).mean() + (_pred[_y == 0] == 0).mean())})
    PCA_ROLLOUTS_SUMMARY = pd.DataFrame(_rows)

    _shown = [m for m in _rollouts if (m, 0) in _panels]
    if _shown:
        with style(**STYLE_PAPER):
            _row_h, _panel_h = 112.0, 96.0
            _fig = page_figure(TEXT_WIDTH, 30.0 + _row_h * len(_shown))
            for _i, _model in enumerate(_shown):
                for _j in range(3):
                    _z, _groups, _read = _panels[_model, _j]
                    _ax = axes_at(_fig, 16.0 + _j * 127.0, 4.0 + (len(_shown) - 1 - _i) * _row_h, 122.0, _panel_h)
                    # Warning-window tokens last, so the band stays visible on top of the other two groups.
                    for _group, _size, _alpha in (("pre", 2.0, 0.6), ("post", 2.0, 0.35), ("warn", 3.5, 0.9)):
                        _ax.scatter(_z[_groups[_group], 0], _z[_groups[_group], 1], s=_size, c=PAPER_COLOR[_group],
                                    alpha=_alpha, lw=0, rasterized=True)
                    # The axes of each panel are its own principal components, so they carry no ticks.
                    _ax.set_xticks([])
                    _ax.set_yticks([])
                    for _side in ("top", "right", "left", "bottom"):
                        _ax.spines[_side].set_visible(True)
                        _ax.spines[_side].set_color(GRID)
                    _var = next(r for r in _rows if r["model"] == _model and r["regime"] == _columns[_j])
                    _ax.set_title(f"layer {_read}, PC1 {_var['pc1_var']:.0%}, PC2 {_var['pc2_var']:.0%}",
                                  fontsize=PAPER_SIZE["note"], color=INK_SOFT, pad=2.5)
                    if _j == 0:
                        _ax.set_ylabel(_rollouts[_model][0], fontsize=PAPER_SIZE["title"], color="black", labelpad=3)
                    if _i == 0:
                        _ax.text(0.0, 1.13, _columns[_j], transform=_ax.transAxes, fontsize=PAPER_SIZE["title"])
            _handles = [Line2D([], [], ls="", marker="o", ms=4.0, mfc=PAPER_COLOR[g], mec="none") for g in ("pre", "warn", "post")]
            _fig.legend(_handles, [_names[g] for g in ("pre", "warn", "post")], loc="upper center",
                        bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.0)
            save(_fig, FIG_DIR / "fig_pca_rollouts.pdf")
            save(_fig, FIG_DIR / "fig_pca_rollouts.png", dpi=300)
    PCA_ROLLOUTS_SUMMARY.round(3)
    return (PCA_ROLLOUTS_SUMMARY,)


@app.cell
def _(FIG_DIR, GRID, RUNS_ROOT, STYLE_BARS, axes_at, np, page_figure, pd, save, style):
    # How far a degenerate response's residual stream sits from healthy responses to the
    # same prompt, at the same absolute positions, as the onset approaches. For each
    # degenerate response and 64-token bin relative to its onset, the mean state over the
    # bin is compared with the mean state of its healthy siblings over the same positions,
    # in units of the siblings' spread there. The reference applies the same measure to
    # one healthy sibling against the others, so it gives the distance expected between
    # two healthy responses to one prompt. Both use leave-one-sibling-out means, so they
    # average over the same number of sibling responses.
    from safetensors.torch import load_file as _load_file

    _root = RUNS_ROOT / "paper_figures" / "pca_states_paired"
    _models = {"apertus": ("Apertus", 15), "llama3p1": ("Llama", 15), "mistral": ("Mistral", 5)}
    _bins = list(range(-1024, 256, 64))
    _colors = {"frozen": "#0072B2", "adapted": "#E69F00"}

    def _distance(target, others, start, stop):
        """Distance of target's mean over [start, stop) to the others' mean there, over their spread."""
        pooled = np.concatenate([o[start:stop] for o in others])
        centre = pooled.mean(0)
        spread = np.sqrt(((pooled - centre) ** 2).sum(1).mean())
        return float(np.linalg.norm(target[start:stop].mean(0) - centre) / spread)

    def _profile(model, regime):
        manifest = pd.read_parquet(_root / model / "manifest.parquet")
        load = lambda r: _load_file(str(_root / model / regime / r.domain / r.prompt_id / f"rollout_{r.rollout_idx}.safetensors"))["hidden_states"].float().numpy()
        rows = []
        for prompt_id, group in manifest.groupby("prompt_id"):
            healthy = [load(r) for r in group[~group["is_positive"]].itertuples()]
            for degenerate in group[group["is_positive"]].itertuples():
                states, onset = load(degenerate), int(degenerate.onset_position)
                for lo in _bins:
                    start, stop = onset + lo, onset + lo + 64
                    if start < 0:
                        continue
                    covering = [h for h in healthy if len(h) >= stop]
                    if len(covering) < 3:
                        continue
                    for left_out in range(len(covering)):
                        rest = covering[:left_out] + covering[left_out + 1:]
                        rows.append({"bin": lo, "prompt": prompt_id, "kind": "degenerate",
                                     "distance": _distance(states, rest, start, stop)})
                        rows.append({"bin": lo, "prompt": prompt_id, "kind": "healthy",
                                     "distance": _distance(covering[left_out], rest, start, stop)})
        return pd.DataFrame(rows)

    _frames = []
    for _model in _models:
        if (_root / _model / "manifest.parquet").is_file():
            for _regime in ("frozen", "adapted"):
                _part = _profile(_model, _regime)
                _part["model"], _part["regime"] = _model, _regime
                _frames.append(_part)
    SIBLING_PROFILE = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()

    if len(SIBLING_PROFILE):
        # Mean over prompts first, so a prompt with two degenerate responses or more siblings
        # does not count more; the band is a 95% interval over prompts.
        _per_prompt = SIBLING_PROFILE.groupby(["model", "regime", "kind", "bin", "prompt"])["distance"].mean().reset_index()
        _summary = _per_prompt.groupby(["model", "regime", "kind", "bin"])["distance"].agg(["mean", "sem", "count"]).reset_index()
        _summary.to_csv(RUNS_ROOT / "analysis" / "sibling_distance_profile.csv", index=False)
        _shown = [m for m in _models if m in set(_summary["model"])]
        with style(**STYLE_BARS):
            _fig = page_figure(666.72, 175.0)
            _axes = [axes_at(_fig, 45.0 + _i * 210.0, 42.0, 190.0, 105.0) for _i in range(len(_shown))]
            for _ax, _model in zip(_axes, _shown):
                _ax.axvspan(-256, 0, color=GRID, alpha=0.7, lw=0, zorder=0)
                _ax.axvline(0, color="black", lw=0.6, zorder=1)
                for _regime in ("frozen", "adapted"):
                    for _kind, _ls, _lw in (("degenerate", "-", 1.5), ("healthy", ":", 1.1)):
                        _row = _summary[(_summary["model"] == _model) & (_summary["regime"] == _regime)
                                        & (_summary["kind"] == _kind) & (_summary["count"] >= 8)].sort_values("bin")
                        _x = _row["bin"].to_numpy() + 32
                        _ax.plot(_x, _row["mean"], color=_colors[_regime], ls=_ls, lw=_lw,
                                 label=f"{_regime}, {'degenerate' if _kind == 'degenerate' else 'healthy sibling'}")
                        _ax.fill_between(_x, _row["mean"] - 1.96 * _row["sem"], _row["mean"] + 1.96 * _row["sem"],
                                         color=_colors[_regime], alpha=0.15, lw=0)
                _ax.set_title(f"{_models[_model][0]}, layer {_models[_model][1]}", loc="center", pad=4)
                _ax.set_xlim(-1024, 256)
                _ax.set_xticks([-1024, -768, -512, -256, 0, 256], ["\u22121024", "\u2212768", "\u2212512", "\u2212256", "0", "256"])
                _ax.set_xlabel("tokens from the onset")
                _ax.grid(axis="y")
                _ax.set_axisbelow(True)
            _axes[0].set_ylabel("distance to healthy siblings")
            _axes[0].legend(loc="upper left", fontsize=6.5, handlelength=2.2)
            save(_fig, FIG_DIR / "fig_sibling_distance.pdf")
            save(_fig, FIG_DIR / "fig_sibling_distance.png", dpi=300)
    SIBLING_PROFILE.head()
    return (SIBLING_PROFILE,)


@app.cell
def _(DATA, MISSING, pd):
    _needs = {
        "budgeted": ("fig5_budget_bars_5pct, fig0_intro_summary, fig0_coverage_vs_early", "s4"),
        "sweep": ("new_sweep_m", "s4"),
        "metrics": ("fig_metrics", "metrics"),
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
