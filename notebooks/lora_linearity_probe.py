import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from matplotlib.collections import LineCollection
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from degeneration_probe.data.activation_store import (
        agrees_with_live_capture,
        load_probe_layer,
    )

    return (
        LineCollection,
        LogisticRegression,
        PCA,
        Path,
        load_probe_layer,
        mo,
        mpl,
        np,
        pd,
        plt,
        roc_auc_score,
        torch,
    )


@app.cell
def _(Path):
    REPO = Path("/iopsstor/scratch/cscs/mdenegri/degeneration-probe")
    OUTPUTS = REPO / "outputs"
    BUILD_ROOT = Path(
        "/capstor/store/cscs/swissai/infra01/users/mdenegri/degeneration-probe"
        "/degeneration-dataset-apertus-8b-instruct"
    )
    # Where the training-time cache's hidden states actually live. Usually
    # BUILD_ROOT/activations, but degeneration_probe.config.DatasetConfig now
    # supports pointing this elsewhere (activations_root), so this is read the
    # same way the extraction script reads it rather than assumed.
    BUILD_ACTIVATIONS_ROOT = BUILD_ROOT / "activations"
    ACTIVATIONS_ROOT = OUTPUTS / "lora_linearity_probe" / "activations"
    FIGURES = REPO / "notebooks" / "figures" / "lora_linearity_probe"
    FIGURES.mkdir(parents=True, exist_ok=True)

    # The layer both runs share: the frozen run trains a head at every depth,
    # the LoRA run trains a single head at this one depth, and this is the depth
    # extraction was run for. Every figure below is at this layer unless it says
    # otherwise.
    PROBE_LAYER = 15

    # Every figure and table below iterates this dict rather than naming "frozen"
    # or "lora" directly, so a third run (a shuffled-frontier LoRA control) can be
    # dropped in later by adding one entry here -- once its own extraction pass
    # (cluster/extract_lora_linearity.sbatch) has been run -- and nothing else in
    # the notebook needs to change.
    RUNS = {
        "frozen": {
            "label": "frozen",
            "description": "layer 15 of an all-layer run, activations read from the cache",
            "probe_dir": OUTPUTS
            / "apertus-8b-instruct_L1-31_all_tokens_hard1024_bce_lora-none_s42_7790c264"
            / "20260814T050453"
            / "checkpoint-950"
            / "layer_15",
            "activations_root": ACTIVATIONS_ROOT / "frozen",
        },
        "lora": {
            "label": "LoRA",
            "description": "single-depth run at layer 15, adapters loaded from checkpoint-1600",
            "probe_dir": OUTPUTS
            / "apertus-8b-instruct_L15_all_tokens_hard1024_bce_lora-all_s42_b9e9df34"
            / "20260816T085625"
            / "checkpoint-1600",
            "activations_root": ACTIVATIONS_ROOT / "lora",
        },
    }
    return ACTIVATIONS_ROOT, BUILD_ACTIVATIONS_ROOT, FIGURES, PROBE_LAYER, RUNS


@app.cell
def _(mo, mpl):
    # Same reference palette as notebooks/experiment_diary.py, reused unchanged
    # so a reader moving between the two notebooks is not asked to learn a
    # second color language.
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    RAMP = ["#bdd7f2", "#7fb2e5", "#2a78d6", "#17457d"]
    INK = "#0b0b0b"
    INK_SOFT = "#52514e"
    INK_MUTED = "#8a8880"
    GRID = "#e3e2dd"

    def use_paper_style():
        mpl.rcParams.update(
            {
                "figure.dpi": 110,
                "savefig.bbox": "tight",
                "font.size": 9,
                "axes.titlesize": 9.5,
                "axes.labelsize": 9,
                "axes.titlelocation": "left",
                "axes.titlepad": 8,
                "axes.edgecolor": INK_MUTED,
                "axes.labelcolor": INK_SOFT,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "text.color": INK,
                "xtick.color": INK_SOFT,
                "ytick.color": INK_SOFT,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "legend.frameon": False,
                "grid.color": GRID,
                "grid.linewidth": 0.7,
                "lines.linewidth": 2.0,
                "lines.markersize": 5,
            }
        )

    use_paper_style()

    def save(fig, name, figures_dir):
        for suffix in (".pdf", ".png"):
            fig.savefig((figures_dir / name).with_suffix(suffix))
        return fig

    def tidy(axis, *, xgrid=True):
        axis.grid(axis="x" if xgrid else "y", alpha=0.9)
        axis.set_axisbelow(True)
        return axis

    def missing(message):
        return mo.md(f"**Nothing to show yet.** {message}").callout(kind="warn")

    return INK_MUTED, RAMP, SERIES, missing, save, tidy


@app.cell
def _(mo):
    mo.md(r"""
    # Does a LoRA-adapted representation make degeneration linearly separable?

    Two runs share everything but one thing: whether the representation was
    allowed to move. Both are the `all_tokens` / `hard1024` / BCE recipe at
    seed 42, read at layer 15. One trains a linear probe over the frozen
    residual stream. The other trains the same probe while LoRA adapters
    reshape the residual stream underneath it, at every layer up to and
    including 15.

    A probe reads a token's residual state through its own learned
    LayerNorm and one linear map. If the LoRA run's probe separates
    degenerate from healthy tokens more cleanly than the frozen run's, that
    could mean the adapted representation makes the distinction more
    *accessible* along a direction PCA can find. It could also just mean
    the probe overfit a small, adapted space, or that some confound
    (position in the answer, which prompt, which domain) is doing the
    separating instead of degeneration itself. The figures below are built
    to tell those apart: every one that recolors the same points by a
    confound is a chance for the "it's about degeneration" story to fail.

    Everything below reads layer 15 unless a figure says otherwise. A
    closing section states plainly what this notebook can and cannot
    establish.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The 30 rollouts

    Three validation-split prompts, ten rollouts each (`K=10`), chosen from
    the same corpus `notebooks/experiment_diary.py` reads:

    - `if_sft_data_verified_00483` -- 5 of 10 rollouts degenerate
      (rollouts 1, 3, 5, 7, 8; frontiers at tokens 654, 70, 685, 0, 658).
    - `llama_nemotron_00398` -- 4 of 10 rollouts degenerate (rollouts 3, 4,
      7, 8; frontiers at tokens 931, 2648, 1498, 928).
    - `deepmath_103k_00001` -- 0 of 10 rollouts degenerate.

    The first two are different in-domain sources with a mix of healthy and
    degenerate siblings of the *same* prompt, which is what lets figure 1
    ask whether a separation is about the trajectory or about the prompt.
    The third supplies healthy tokens with no frontier at all. Every
    positive rollout's `is_positive` flag already implies a resolved
    frontier (`onset_resolution == "ok"`) in this corpus, and
    `scripts/extract_lora_linearity_activations.py` re-asserts that
    directly before extracting anything.

    The manifest below is read from disk, not retyped, so a mismatch here
    would mean the extraction was run for different rollouts than the text
    above describes.
    """)
    return


@app.cell
def _(ACTIVATIONS_ROOT, mo, pd):
    EXPECTED_PROMPT_IDS = [
        "if_sft_data_verified_00483",
        "llama_nemotron_00398",
        "deepmath_103k_00001",
    ]
    _manifest_path = ACTIVATIONS_ROOT / "manifest.parquet"
    if _manifest_path.is_file():
        ROLLOUT_MANIFEST = pd.read_parquet(_manifest_path)
        _found = sorted(ROLLOUT_MANIFEST["prompt_id"].unique())
        if _found != sorted(EXPECTED_PROMPT_IDS):
            raise ValueError(
                f"manifest at {_manifest_path} lists prompts {_found}, expected {EXPECTED_PROMPT_IDS}"
            )
        if len(ROLLOUT_MANIFEST) != 30:
            raise ValueError(f"expected 30 rollouts in the manifest, found {len(ROLLOUT_MANIFEST)}")
        _summary = (
            ROLLOUT_MANIFEST.groupby(["domain", "prompt_id"])
            .agg(rollouts=("rollout_idx", "size"), degenerate=("is_positive", "sum"))
            .reset_index()
        )
        mo.output.replace(
            mo.vstack([mo.ui.table(_summary, selection=None), mo.ui.table(ROLLOUT_MANIFEST, selection=None)])
        )
    else:
        ROLLOUT_MANIFEST = None
    return (ROLLOUT_MANIFEST,)


@app.cell
def _(ACTIVATIONS_ROOT, ROLLOUT_MANIFEST, missing, mo):
    mo.stop(
        ROLLOUT_MANIFEST is None,
        missing(
            f"No manifest at {ACTIVATIONS_ROOT / 'manifest.parquet'}. Run "
            "`sbatch cluster/extract_lora_linearity.sbatch --frozen-run-dir ... "
            "--lora-checkpoint ... --out-dir outputs/lora_linearity_probe/activations` first."
        ),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Extraction: one teacher-forced pass per condition

    The frozen run's activations already exist in the training-time cache.
    The LoRA run's do not, and re-generating them live in this notebook
    would tempt mixing a cached frozen vector with a freshly computed LoRA
    one for the same token, which is exactly the numerical seam that must
    not be in this comparison.

    Instead `scripts/extract_lora_linearity_activations.py` makes one
    forward pass per condition over the same 30 rollouts, teacher-forced
    over each rollout's *stored* generated token ids (never re-generated),
    with `output_hidden_states=True` so every layer is captured at once.
    Condition A is the base model with adapters disabled; condition B is
    the same weights with the LoRA adapter from `checkpoint-1600` loaded on
    top. Files are written in the same layout and tensor format the
    training-time cache uses, so they are read back below with the same
    `degeneration_probe.data.activation_store` helpers the training code
    uses, including their layer-order and slot-offset checks.

    Before writing anything else, the script compares its own
    recomputed frozen-condition, layer-15 activations against the existing
    training-time cache for a few rollouts and stops if they disagree. The
    next cell repeats that check independently, from inside this notebook,
    for one rollout of each of the three prompts -- so a reader does not
    have to trust the extraction log, only what is on disk right now.
    """)
    return


@app.cell
def _(
    ACTIVATIONS_ROOT,
    BUILD_ACTIVATIONS_ROOT,
    PROBE_LAYER,
    ROLLOUT_MANIFEST,
    load_probe_layer,
    mo,
    torch,
):
    # Agreement is judged by flattened cosine similarity (> 0.999), the same
    # criterion scripts/verify_cached_layer.py uses and the same one
    # scripts/extract_lora_linearity_activations.py's own sanity check uses.
    # A strict elementwise torch.allclose at bf16-scale tolerance fails even for
    # a verified-correct reproduction at this depth: the live forward pass runs
    # in bf16 (~7 mantissa bits), and that rounding compounds over 15 decoder
    # layers into elementwise disagreement across most of the tensor even while
    # every token's direction stays within a fraction of a degree of the fp16
    # cache. Cosine similarity is also what actually matters for a linear probe,
    # since it reads direction through a LayerNorm, not raw magnitude.
    def _cosine_agreement(a, b):
        return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))

    _rows = []
    if ROLLOUT_MANIFEST is not None:
        for _prompt_id, _group in ROLLOUT_MANIFEST.groupby("prompt_id"):
            _row = _group.iloc[0]
            _cached = load_probe_layer(
                BUILD_ACTIVATIONS_ROOT, _row["domain"], _row["prompt_id"], int(_row["rollout_idx"]), probe_layer=PROBE_LAYER
            )
            _recomputed = load_probe_layer(
                ACTIVATIONS_ROOT / "frozen",
                _row["domain"],
                _row["prompt_id"],
                int(_row["rollout_idx"]),
                probe_layer=PROBE_LAYER,
            )
            _n = min(_cached.shape[0], _recomputed.shape[0])
            _cosine = _cosine_agreement(_cached[:_n], _recomputed[:_n])
            _rows.append(
                {
                    "prompt_id": _row["prompt_id"],
                    "rollout_idx": int(_row["rollout_idx"]),
                    "tokens_compared": _n,
                    "cosine_similarity": round(_cosine, 6),
                    "matches_training_cache": _cosine > 0.999,
                }
            )
    SANITY_CHECK = mo.ui.table(_rows, selection=None) if _rows else None
    SANITY_OK = bool(_rows) and all(row["matches_training_cache"] for row in _rows)
    SANITY_CHECK
    return (SANITY_OK,)


@app.cell
def _(load_probe_layer, np, pd):
    def load_condition_tokens(activations_root, rollouts, layer):
        """Every pooled token of one condition at one layer, plus its metadata.

        Row *i* of the returned array and row *i* of the returned frame are the
        same token: both are built in the same rollout order, so any boolean
        mask on the frame indexes the array directly.
        """
        vector_chunks = []
        meta_rows = []
        for row in rollouts.itertuples():
            states = load_probe_layer(
                activations_root, row.domain, row.prompt_id, int(row.rollout_idx), probe_layer=layer
            ).numpy().astype(np.float32)
            n_tokens = states.shape[0]
            vector_chunks.append(states)
            positions = np.arange(n_tokens)
            onset = None if pd.isna(row.onset_position) else int(row.onset_position)
            if row.is_positive:
                token_class = np.where(positions < onset, "pre_frontier", "in_pattern")
                distance_to_frontier = (positions - onset).astype(np.float64)
            else:
                token_class = np.full(n_tokens, "healthy", dtype=object)
                distance_to_frontier = np.full(n_tokens, np.nan)
            meta_rows.append(
                pd.DataFrame(
                    {
                        "prompt_id": row.prompt_id,
                        "domain": row.domain,
                        "rollout_idx": int(row.rollout_idx),
                        "rollout_id": f"{row.prompt_id}/rollout_{row.rollout_idx}",
                        "is_positive": bool(row.is_positive),
                        "token_position": positions,
                        "token_class": token_class,
                        "distance_to_frontier": distance_to_frontier,
                        "onset_position": onset if onset is not None else np.nan,
                    }
                )
            )
        vectors = np.concatenate(vector_chunks, axis=0)
        meta = pd.concat(meta_rows, ignore_index=True)
        assert vectors.shape[0] == len(meta)
        return vectors, meta

    def apply_layernorm(x, weight, bias, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight + bias

    def load_probe_head(probe_dir, torch_module):
        linear = torch_module.load(probe_dir / "probe.bin", map_location="cpu", weights_only=True)
        norm_path = probe_dir / "normalization.bin"
        norm = (
            torch_module.load(norm_path, map_location="cpu", weights_only=True)
            if norm_path.is_file()
            else None
        )
        return {
            "linear_weight": linear["weight"].numpy().reshape(-1).astype(np.float32),
            "linear_bias": float(linear["bias"].numpy().reshape(-1)[0]),
            "ln_weight": norm["weight"].numpy().astype(np.float32) if norm is not None else None,
            "ln_bias": norm["bias"].numpy().astype(np.float32) if norm is not None else None,
        }

    return apply_layernorm, load_condition_tokens, load_probe_head


@app.cell
def _(
    PROBE_LAYER,
    ROLLOUT_MANIFEST,
    RUNS,
    SANITY_OK,
    apply_layernorm,
    load_condition_tokens,
    load_probe_head,
    missing,
    mo,
    np,
    torch,
):
    # This is the one real gate: if the sanity check above failed, CONDITION_DATA
    # is never built, and every downstream cell that reads it -- every figure,
    # every table -- stops with it rather than plotting untrustworthy activations.
    mo.stop(
        not SANITY_OK,
        missing(
            "The recomputed frozen-condition activations do not match the training-time cache. "
            "Every figure below is disabled until this is fixed: a mismatch here means the two "
            "conditions are not comparable, and nothing downstream should be trusted."
        ),
    )
    CONDITION_DATA = {}
    for _run_key, _run in RUNS.items():
        if not _run["activations_root"].is_dir() or not (_run["probe_dir"] / "probe.bin").is_file():
            CONDITION_DATA[_run_key] = {"ok": False}
            continue
        _raw, _meta = load_condition_tokens(_run["activations_root"], ROLLOUT_MANIFEST, PROBE_LAYER)
        _head = load_probe_head(_run["probe_dir"], torch)
        _norm = (
            apply_layernorm(_raw, _head["ln_weight"], _head["ln_bias"])
            if _head["ln_weight"] is not None
            else _raw
        )
        if _head["ln_weight"] is not None:
            _logits = _norm @ _head["linear_weight"] + _head["linear_bias"]
            _meta = _meta.copy()
            _meta["probe_score"] = 1.0 / (1.0 + np.exp(-_logits))
        CONDITION_DATA[_run_key] = {
            "ok": True,
            "raw": _raw,
            "norm": _norm,
            "meta": _meta,
            "head": _head,
        }
    return (CONDITION_DATA,)


@app.cell
def _(mo):
    mo.md(r"""
    ## PCA: fit per condition, projected separately

    Each condition gets its own PCA, fit on the pooled tokens of all 30
    rollouts of that condition alone, with no class weighting -- a
    degenerate token is not up-weighted just because it is rarer. Frozen
    and LoRA activations do not share a coordinate system (LoRA changes the
    weights that produced them), so a single shared PCA would silently
    compare two different bases; fitting separately is what makes "PC1-PC2
    of the frozen space" and "PC1-PC2 of the LoRA space" each a valid
    question, though never a directly comparable axis between the two
    panels of one figure.

    PCA is fit with randomized SVD, keeping 20 components, and fit twice
    per condition: once on the raw residual states, once after applying
    that condition's own learned probe LayerNorm. The LayerNorm is what the
    probe actually reads, and applying it removes the scale change LoRA
    introduces to the residual stream, which would otherwise show up as
    apparent separation that has nothing to do with degeneration. A toggle
    below switches every figure between the two, so both can be inspected.

    Every figure fits on all pooled tokens but plots every 4th (with alpha)
    to keep the scatters legible.
    """)
    return


@app.cell
def _(CONDITION_DATA, PCA):
    PCA_DATA = {}
    for _run_key, _data in CONDITION_DATA.items():
        if not _data.get("ok"):
            PCA_DATA[_run_key] = {"ok": False}
            continue
        PCA_DATA[_run_key] = {"ok": True}
        for _space, _key in (("raw", "raw"), ("norm", "norm")):
            _pca = PCA(n_components=20, svd_solver="randomized", random_state=0)
            _embedding = _pca.fit_transform(_data[_key])
            PCA_DATA[_run_key][_space] = {"pca": _pca, "embedding": _embedding}
    SUBSAMPLE_STRIDE = 4
    return PCA_DATA, SUBSAMPLE_STRIDE


@app.cell
def _(mo):
    space_toggle = mo.ui.radio(
        options=["probe LayerNorm (own head)", "raw (unnormalised)"],
        value="probe LayerNorm (own head)",
        label="representation space for figures 1, 2, 3 and 5, and the numbers table",
    )
    space_toggle
    return (space_toggle,)


@app.cell
def _(space_toggle):
    SPACE_KEY = "norm" if space_toggle.value.startswith("probe") else "raw"
    SPACE_TITLE = space_toggle.value
    return SPACE_KEY, SPACE_TITLE


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 1: does the class separate?

    **Under the hypothesis** (LoRA makes degeneration more linearly
    accessible): the LoRA panel shows pre-frontier and in-pattern tokens
    pulling away from healthy tokens more cleanly than the frozen panel
    does, in just the top two components.

    **Under the negation**: the two panels look about equally mixed, or
    the frozen panel already separates the classes about as well, in which
    case LoRA is not adding accessibility this figure can see.

    Coloring is by token class only. Figure 2 recolors the identical points
    by three confounds; if any of those separates as cleanly as this does,
    this figure is not evidence about degeneration.
    """)
    return


@app.cell
def _(
    CONDITION_DATA,
    FIGURES,
    PCA_DATA,
    RUNS,
    SERIES,
    SPACE_KEY,
    SPACE_TITLE,
    SUBSAMPLE_STRIDE,
    missing,
    mo,
    np,
    plt,
    save,
    tidy,
):
    CLASS_ORDER = ["healthy", "pre_frontier", "in_pattern"]
    CLASS_COLORS = dict(zip(CLASS_ORDER, [SERIES[0], SERIES[1], SERIES[2]]))
    CLASS_LABELS = {
        "healthy": "healthy rollout",
        "pre_frontier": "pre-frontier (degenerate rollout)",
        "in_pattern": "at/after frontier (degenerate rollout)",
    }

    def class_scatter_figure():
        available = [key for key in RUNS if CONDITION_DATA.get(key, {}).get("ok")]
        if not available:
            return missing("No condition has extracted activations to plot yet.")
        fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4.6), squeeze=False)
        axes = axes[0]
        for axis, run_key in zip(axes, available):
            meta = CONDITION_DATA[run_key]["meta"]
            embedding = PCA_DATA[run_key][SPACE_KEY]["embedding"]
            keep = np.arange(len(meta)) % SUBSAMPLE_STRIDE == 0
            for cls in CLASS_ORDER:
                mask = keep & (meta["token_class"].values == cls)
                axis.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    s=8, alpha=0.3 if cls == "healthy" else 0.55,
                    color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], linewidths=0,
                )
            axis.set_title(f"{RUNS[run_key]['label']} ({RUNS[run_key]['description']})", fontsize=8.5)
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            tidy(axis, xgrid=False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), fontsize=8)
        fig.suptitle(f"PC1-PC2 by token class -- {SPACE_TITLE}", x=0.005, ha="left", fontsize=10, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        return mo.vstack(
            [
                save(fig, "fig1_class_scatter", FIGURES),
                mo.md(
                    f"_Every 4th token of all 30 rollouts, fit on all of them. "
                    f"Space: {SPACE_TITLE}._"
                ),
            ]
        )

    class_scatter_figure()
    return (CLASS_COLORS,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 2: confound checks

    The same projection as figure 1, recolored three times by things that
    have nothing to do with degeneration on their own: how far into the
    answer a token is, which domain it came from, which prompt it came
    from.

    **Under the hypothesis**: none of these three recolorings separates as
    cleanly as the class coloring in figure 1 did. A cluster of points
    being "late in the answer" or "from `llama_nemotron`" is expected --
    degenerate tokens usually are late, and each prompt has its own cluster
    of siblings -- but that clustering should not, on its own, reproduce
    the pre-frontier/in-pattern/healthy split.

    **Under the negation**: one of these panels looks about as separated as
    figure 1. If position alone divides the plot, figure 1 could be
    reading "this token is late" rather than "this token is heading into a
    loop." If prompt identity alone divides it, figure 1 could be reading
    "this is prompt X" rather than anything about the trajectory. Either
    way the text below says so plainly rather than letting figure 1 stand
    unqualified.
    """)
    return


@app.cell
def _(
    CONDITION_DATA,
    FIGURES,
    PCA_DATA,
    RUNS,
    SPACE_KEY,
    SPACE_TITLE,
    SUBSAMPLE_STRIDE,
    missing,
    mo,
    mpl,
    np,
    plt,
    save,
):
    def confound_scatter_figure():
        available = [key for key in RUNS if CONDITION_DATA.get(key, {}).get("ok")]
        if not available:
            return missing("No condition has extracted activations to plot yet.")
        confounds = ["token_position", "domain", "prompt_id"]
        fig, axes = plt.subplots(
            len(confounds), len(available), figsize=(5.0 * len(available), 3.6 * len(confounds)), squeeze=False
        )
        for row_idx, confound in enumerate(confounds):
            for col_idx, run_key in enumerate(available):
                axis = axes[row_idx][col_idx]
                meta = CONDITION_DATA[run_key]["meta"]
                embedding = PCA_DATA[run_key][SPACE_KEY]["embedding"]
                keep = np.arange(len(meta)) % SUBSAMPLE_STRIDE == 0
                xy = embedding[keep]
                values = meta[confound].values[keep]
                if confound == "token_position":
                    scatter = axis.scatter(xy[:, 0], xy[:, 1], c=values, cmap="viridis", s=8, alpha=0.55, linewidths=0)
                    if col_idx == len(available) - 1:
                        fig.colorbar(scatter, ax=axis, pad=0.02, shrink=0.9).set_label("token position", fontsize=7)
                else:
                    categories = sorted(pd_series_unique(values))
                    cmap = mpl.colormaps["tab10"]
                    for cat_idx, category in enumerate(categories):
                        mask = values == category
                        axis.scatter(
                            xy[mask, 0], xy[mask, 1], s=8, alpha=0.55, linewidths=0,
                            color=cmap(cat_idx % 10), label=str(category),
                        )
                    if col_idx == len(available) - 1:
                        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.5)
                if row_idx == 0:
                    axis.set_title(RUNS[run_key]["label"], fontsize=9)
                if col_idx == 0:
                    axis.set_ylabel(confound, fontsize=8)
                axis.set_xticks([])
                axis.set_yticks([])
        fig.suptitle(f"Figure 1's projection, recolored by confounds -- {SPACE_TITLE}", x=0.005, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return mo.vstack([save(fig, "fig2_confounds", FIGURES), mo.md("_Same points, same axes as figure 1, three colorings._")])

    def pd_series_unique(values):
        seen = []
        for value in values:
            if value not in seen:
                seen.append(value)
        return seen

    confound_scatter_figure()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 3: distance-to-frontier bands

    A single pre-frontier color, as in figure 1, can only show a step: a
    token is either flagged as "before the loop" or it is not. The actual
    claim worth testing is different -- that the representation drifts
    *gradually* as a rollout approaches its frontier, not that it snaps at
    some boundary. This figure colors pre-frontier tokens by which band of
    distance-to-frontier they fall in, from far (`[-1024,-512)`, lightest)
    to near (`[-128,0)`, darkest), with in-pattern tokens in a separate
    color and tokens more than 1024 before their frontier shown muted as
    context.

    **Under the hypothesis**: the bands trace a visible gradient toward the
    in-pattern cluster, darkest band closest.

    **Under the negation**: the bands are interleaved with no visible
    order, in which case whatever separation figure 1 showed is closer to
    a step at the frontier than an approach.
    """)
    return


@app.cell
def _(
    CLASS_COLORS,
    CONDITION_DATA,
    FIGURES,
    INK_MUTED,
    PCA_DATA,
    RAMP,
    RUNS,
    SPACE_KEY,
    SPACE_TITLE,
    SUBSAMPLE_STRIDE,
    missing,
    mo,
    np,
    plt,
    save,
    tidy,
):
    BAND_EDGES = [(-1024, -512), (-512, -256), (-256, -128), (-128, 0)]
    BAND_LABELS = [f"[{lo},{hi})" for lo, hi in BAND_EDGES]

    def band_scatter_figure():
        available = [key for key in RUNS if CONDITION_DATA.get(key, {}).get("ok")]
        if not available:
            return missing("No condition has extracted activations to plot yet.")
        fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4.6), squeeze=False)
        axes = axes[0]
        for axis, run_key in zip(axes, available):
            meta = CONDITION_DATA[run_key]["meta"]
            embedding = PCA_DATA[run_key][SPACE_KEY]["embedding"]
            keep = np.arange(len(meta)) % SUBSAMPLE_STRIDE == 0
            dist = meta["distance_to_frontier"].values
            cls = meta["token_class"].values

            far_mask = keep & (cls == "pre_frontier") & (dist < BAND_EDGES[0][0])
            axis.scatter(
                embedding[far_mask, 0], embedding[far_mask, 1], s=6, alpha=0.15,
                color=INK_MUTED, linewidths=0, label="more than 1024 before frontier",
            )
            for (lo, hi), color, label in zip(BAND_EDGES, RAMP, BAND_LABELS):
                mask = keep & (cls == "pre_frontier") & (dist >= lo) & (dist < hi)
                axis.scatter(
                    embedding[mask, 0], embedding[mask, 1], s=10, alpha=0.75,
                    color=color, linewidths=0, label=label,
                )
            in_pattern_mask = keep & (cls == "in_pattern")
            axis.scatter(
                embedding[in_pattern_mask, 0], embedding[in_pattern_mask, 1], s=8, alpha=0.5,
                color=CLASS_COLORS["in_pattern"], linewidths=0, label="in-pattern",
            )
            axis.set_title(RUNS[run_key]["label"], fontsize=9.5)
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            tidy(axis, xgrid=False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1), fontsize=7.5)
        fig.suptitle(f"Distance to frontier, banded -- {SPACE_TITLE}", x=0.005, ha="left", fontsize=10, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.82))
        return mo.vstack([save(fig, "fig3_distance_bands", FIGURES), mo.md("_Darkest = closest to the frontier. Healthy rollouts are not shown here; figure 1 already covers them._")])

    band_scatter_figure()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 4: one rollout's path

    Figures 1 to 3 pool tokens across rollouts, which can hide what a
    single generation actually does over time. This figure draws one
    rollout's path through PC1-PC2 in generation order, colored by a
    gradient over time, with the frontier marked if it has one. A prompt
    selector picks one of the two prompts with both healthy and degenerate
    siblings, so a degenerate rollout can be shown next to a healthy
    sibling of the *same* prompt.

    **Under the hypothesis**: the degenerate path visibly bends or
    accelerates toward the in-pattern region as it nears its frontier,
    while the healthy path wanders without doing that.

    **Under the negation**: the two paths look similarly shaped, in which
    case whatever figure 1 and figure 3 show is not visible at the level of
    an individual trajectory.
    """)
    return


@app.cell
def _(ROLLOUT_MANIFEST, mo):
    _mixed_prompts = (
        ROLLOUT_MANIFEST.groupby("prompt_id")["is_positive"].agg(["sum", "count"])
        if ROLLOUT_MANIFEST is not None
        else None
    )
    _options = (
        sorted(_mixed_prompts[(_mixed_prompts["sum"] > 0) & (_mixed_prompts["sum"] < _mixed_prompts["count"])].index)
        if _mixed_prompts is not None
        else []
    )
    trajectory_prompt = mo.ui.dropdown(
        options=_options or ["none"], value=(_options[0] if _options else "none"), label="prompt"
    )
    trajectory_prompt
    return (trajectory_prompt,)


@app.cell
def _(ROLLOUT_MANIFEST, mo, trajectory_prompt):
    if ROLLOUT_MANIFEST is not None and trajectory_prompt.value != "none":
        _rows = ROLLOUT_MANIFEST[ROLLOUT_MANIFEST["prompt_id"] == trajectory_prompt.value]
        _degenerate = sorted(_rows.loc[_rows["is_positive"], "rollout_idx"].tolist())
        _healthy = sorted(_rows.loc[~_rows["is_positive"], "rollout_idx"].tolist())
    else:
        _degenerate, _healthy = [0], [0]
    trajectory_degenerate = mo.ui.dropdown(
        options=[str(v) for v in _degenerate], value=str(_degenerate[0]), label="degenerate rollout"
    )
    trajectory_healthy = mo.ui.dropdown(
        options=[str(v) for v in _healthy], value=str(_healthy[0]), label="healthy sibling"
    )
    mo.hstack([trajectory_degenerate, trajectory_healthy], justify="start", gap=2)
    return trajectory_degenerate, trajectory_healthy


@app.cell
def _(
    CONDITION_DATA,
    FIGURES,
    LineCollection,
    PCA_DATA,
    RUNS,
    SPACE_KEY,
    SPACE_TITLE,
    missing,
    mo,
    mpl,
    plt,
    save,
    tidy,
    trajectory_degenerate,
    trajectory_healthy,
    trajectory_prompt,
):
    def one_path(meta, embedding, prompt_id, rollout_idx):
        mask = (meta["prompt_id"].values == prompt_id) & (meta["rollout_idx"].values == rollout_idx)
        return meta[mask].reset_index(drop=True), embedding[mask]

    def trajectory_figure():
        available = [key for key in RUNS if CONDITION_DATA.get(key, {}).get("ok")]
        if not available or trajectory_prompt.value == "none":
            return missing("No condition has extracted activations, or no mixed prompt is available yet.")
        degenerate_idx = int(trajectory_degenerate.value)
        healthy_idx = int(trajectory_healthy.value)
        fig, axes = plt.subplots(2, len(available), figsize=(5.2 * len(available), 8.4), squeeze=False)
        cmap = mpl.colormaps["plasma"]
        for col_idx, run_key in enumerate(available):
            meta = CONDITION_DATA[run_key]["meta"]
            embedding = PCA_DATA[run_key][SPACE_KEY]["embedding"][:, :2]
            for row_idx, (rollout_idx, kind) in enumerate(
                [(degenerate_idx, "degenerate"), (healthy_idx, "healthy")]
            ):
                axis = axes[row_idx][col_idx]
                path_meta, path_xy = one_path(meta, embedding, trajectory_prompt.value, rollout_idx)
                if len(path_meta) == 0:
                    axis.set_axis_off()
                    continue
                positions = path_meta["token_position"].values
                segments = LineCollection(
                    [path_xy[step : step + 2] for step in range(len(path_xy) - 1)],
                    cmap=cmap,
                    norm=mpl.colors.Normalize(vmin=0, vmax=max(1, positions.max())),
                    linewidth=1.4,
                )
                segments.set_array(positions[:-1])
                axis.add_collection(segments)
                axis.autoscale_view()
                if kind == "degenerate" and path_meta["onset_position"].notna().any():
                    onset = int(path_meta["onset_position"].iloc[0])
                    frontier_rows = path_meta[path_meta["token_position"] == onset]
                    if not frontier_rows.empty:
                        frontier_xy = path_xy[frontier_rows.index[0]]
                        axis.plot(frontier_xy[0], frontier_xy[1], "o", color="black", markersize=8, markerfacecolor="none", markeredgewidth=1.6)
                        axis.annotate("frontier", xy=frontier_xy, xytext=(6, 6), textcoords="offset points", fontsize=7)
                axis.set_title(f"{RUNS[run_key]['label']}: {kind} rollout {rollout_idx}", fontsize=9)
                axis.set_xlabel("PC1")
                axis.set_ylabel("PC2")
                tidy(axis, xgrid=False)
        fig.suptitle(
            f"{trajectory_prompt.value}: one path per generation order -- {SPACE_TITLE}",
            x=0.005, ha="left", fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return mo.vstack([save(fig, "fig4_trajectory", FIGURES), mo.md("_Color runs from generation start (dark) to end (light) along the plasma colormap._")])

    trajectory_figure()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 5 and numbers: the warning band only, position-matched

    Figures 1 to 4 compare pre-frontier tokens against *all* healthy
    tokens, which lets position sneak in as an unmatched confound: a
    pre-frontier token late in a 4096-token rollout might separate from a
    healthy token early in a 300-token rollout simply because most
    healthy answers are short. This section restricts to the tokens the
    deployment question is actually about -- the 256 tokens immediately
    before a frontier, `[frontier-256, frontier)` -- and matches them
    against healthy tokens from the *same absolute position range*: token
    positions are grouped into 64-token bins, and only healthy tokens
    whose bin also holds a warning-band token are kept. Without that
    matching this panel would just re-measure position, not degeneration.

    **Under the hypothesis**: even with position controlled for, the
    warning-band tokens separate from their position-matched healthy
    counterparts, more so in the LoRA condition than the frozen one.

    **Under the negation**: the separation figure 1 showed collapses once
    position is matched, in which case it was position all along.
    """)
    return


@app.cell
def _(CONDITION_DATA, RUNS, np, pd):
    WARNING_BAND = 256
    POSITION_BIN_WIDTH = 64

    def matched_populations(run_key):
        meta = CONDITION_DATA[run_key]["meta"]
        dist = meta["distance_to_frontier"].values
        cls = meta["token_class"].values
        warning_mask = (cls == "pre_frontier") & (dist >= -WARNING_BAND) & (dist < 0)
        healthy_mask = cls == "healthy"
        position_bin = (meta["token_position"].values // POSITION_BIN_WIDTH).astype(int)
        warning_bins = set(position_bin[warning_mask].tolist())
        matched_healthy_mask = healthy_mask & np.isin(position_bin, list(warning_bins))
        in_pattern_mask = cls == "in_pattern"
        return {
            "warning": np.flatnonzero(warning_mask),
            "matched_healthy": np.flatnonzero(matched_healthy_mask),
            "any_healthy": np.flatnonzero(healthy_mask),
            "in_pattern": np.flatnonzero(in_pattern_mask),
        }

    MATCHED = {key: matched_populations(key) for key in RUNS if CONDITION_DATA.get(key, {}).get("ok")}
    MATCH_SUMMARY = pd.DataFrame(
        [
            {
                "run": key,
                "warning-band tokens": len(pools["warning"]),
                "position-matched healthy tokens": len(pools["matched_healthy"]),
                "any healthy tokens": len(pools["any_healthy"]),
                "in-pattern tokens": len(pools["in_pattern"]),
            }
            for key, pools in MATCHED.items()
        ]
    )
    MATCH_SUMMARY
    return (MATCHED,)


@app.cell
def _(
    FIGURES,
    MATCHED,
    PCA_DATA,
    RUNS,
    SERIES,
    SPACE_KEY,
    SPACE_TITLE,
    missing,
    mo,
    plt,
    save,
    tidy,
):
    def warning_band_figure():
        available = [key for key in RUNS if key in MATCHED and len(MATCHED[key]["warning"]) and len(MATCHED[key]["matched_healthy"])]
        if not available:
            return missing("No condition has both warning-band and position-matched healthy tokens yet.")
        fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4.6), squeeze=False)
        axes = axes[0]
        for axis, run_key in zip(axes, available):
            embedding = PCA_DATA[run_key][SPACE_KEY]["embedding"]
            healthy_idx = MATCHED[run_key]["matched_healthy"]
            warning_idx = MATCHED[run_key]["warning"]
            axis.scatter(embedding[healthy_idx, 0], embedding[healthy_idx, 1], s=10, alpha=0.4, color=SERIES[0], linewidths=0, label="position-matched healthy")
            axis.scatter(embedding[warning_idx, 0], embedding[warning_idx, 1], s=10, alpha=0.6, color=SERIES[1], linewidths=0, label="warning band (last 256 before frontier)")
            axis.set_title(RUNS[run_key]["label"], fontsize=9.5)
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            tidy(axis, xgrid=False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), fontsize=8)
        fig.suptitle(f"Warning band vs. position-matched healthy -- {SPACE_TITLE}", x=0.005, ha="left", fontsize=10, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        return mo.vstack([save(fig, "fig5_warning_band_matched", FIGURES), mo.md("_All tokens plotted here, not subsampled: the matched pools are already small._")])

    warning_band_figure()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Numbers: separability, not just pictures

    A PCA scatter is suggestive but not evidence on its own. For each
    condition, an L2-regularized logistic regression is fit to separate
    two token populations, in the top-2 PCs, the top-10 PCs, and the full
    4096-dimensional space (of whichever representation the toggle above
    selects). The split for fitting and reporting is by **rollout**, never
    by token: a rollout's tokens are entirely in the training fold or
    entirely in the test fold, so a model cannot pass by memorizing one
    rollout's idiosyncrasies and being asked about its own tokens back.

    Two token-population pairs are reported:

    - **pre-frontier vs. position-matched healthy** (figure 5's
      populations): the actual question, with position controlled for.
    - **in-pattern vs. healthy** (any position): the easy reference. A
      probe that cannot separate the *obvious* case is not a probe worth
      reading anything else from.

    Given how few rollouts contribute to the warning-band population (9
    degenerate rollouts across 2 prompts), the train/test split holds out
    40% of each population's contributing rollouts, seeded for
    reproducibility rather than cross-validated -- there is not enough
    data here to do more than describe the direction of an effect.
    """)
    return


@app.cell
def _(np):
    def split_rollout_ids(rollout_ids, seed=0, test_fraction=0.4):
        ids = sorted(set(rollout_ids))
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids))
        n_test = max(1, round(len(ids) * test_fraction)) if len(ids) > 1 else 0
        test_ids = {ids[i] for i in order[:n_test]}
        train_ids = {ids[i] for i in order[n_test:]}
        return train_ids, test_ids

    def rollout_split_mask(meta, positive_idx, negative_idx, seed=0):
        """Row masks for a train/test split that holds whole rollouts out."""
        pos_rollouts = meta["rollout_id"].values[positive_idx]
        neg_rollouts = meta["rollout_id"].values[negative_idx]
        pos_train_ids, pos_test_ids = split_rollout_ids(pos_rollouts, seed=seed)
        neg_train_ids, neg_test_ids = split_rollout_ids(neg_rollouts, seed=seed + 1)
        pos_train = positive_idx[np.isin(pos_rollouts, list(pos_train_ids))]
        pos_test = positive_idx[np.isin(pos_rollouts, list(pos_test_ids))]
        neg_train = negative_idx[np.isin(neg_rollouts, list(neg_train_ids))]
        neg_test = negative_idx[np.isin(neg_rollouts, list(neg_test_ids))]
        return pos_train, pos_test, neg_train, neg_test

    return (rollout_split_mask,)


@app.cell
def _(LogisticRegression, np, roc_auc_score):
    def fit_and_score(features, train_pos, train_neg, test_pos, test_neg):
        if min(len(train_pos), len(train_neg), len(test_pos), len(test_neg)) == 0:
            return None
        train_y = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
        test_y = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
        if len(np.unique(test_y)) < 2:
            return None
        train_x = features[np.concatenate([train_pos, train_neg])]
        test_x = features[np.concatenate([test_pos, test_neg])]
        center = train_x.mean(axis=0, keepdims=True)
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
        model.fit(train_x - center, train_y)
        scores = model.decision_function(test_x - center)
        return float(roc_auc_score(test_y, scores))

    return (fit_and_score,)


@app.cell
def _(
    CONDITION_DATA,
    MATCHED,
    PCA_DATA,
    RUNS,
    SPACE_KEY,
    fit_and_score,
    mo,
    pd,
    rollout_split_mask,
):
    def separability_row(run_key):
        meta = CONDITION_DATA[run_key]["meta"]
        full_features = CONDITION_DATA[run_key][SPACE_KEY]
        pca_embedding = PCA_DATA[run_key][SPACE_KEY]["embedding"]
        pools = MATCHED[run_key]
        row = {"run": run_key}
        for pair_name, pos_idx, neg_idx in [
            ("pre-frontier vs matched healthy", pools["warning"], pools["matched_healthy"]),
            ("in-pattern vs healthy", pools["in_pattern"], pools["any_healthy"]),
        ]:
            splits = rollout_split_mask(meta, pos_idx, neg_idx, seed=0)
            for dims, features in [("top-2 PCs", pca_embedding[:, :2]), ("top-10 PCs", pca_embedding[:, :10]), ("full 4096-d", full_features)]:
                auc = fit_and_score(features, *splits)
                row[f"{pair_name} -- {dims}"] = None if auc is None else round(auc, 3)
        return row

    SEPARABILITY_TABLE = pd.DataFrame([separability_row(key) for key in RUNS if key in MATCHED])
    mo.ui.table(SEPARABILITY_TABLE, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cosine similarity: does PCA find what the probe uses?

    The figures above show whether PCA's first two directions separate the
    classes. That is a different question from whether the probe's own
    trained weight vector `w` lives anywhere near those directions. This
    table reports, per condition, the cosine similarity between `w` (in
    the probe-LayerNorm space, since that is the space `w` was trained to
    read) and each of the top 5 principal components, plus the fraction of
    `||w||^2` that lies inside the top-10 PC subspace.

    A high fraction means the probe's decision direction is close to the
    directions of largest variance, which is what would make it visible in
    a PCA scatter at all. A low fraction means the probe found a direction
    PCA does not foreground, in which case the figures above could look
    unconvincing even for a probe that works.
    """)
    return


@app.cell
def _(CONDITION_DATA, PCA_DATA, RUNS, mo, np, pd):
    def cosine_row(run_key):
        head = CONDITION_DATA[run_key]["head"]
        w = head["linear_weight"].astype(np.float64)
        w_norm = w / np.linalg.norm(w)
        components = PCA_DATA[run_key]["norm"]["pca"].components_.astype(np.float64)
        row = {"run": run_key}
        for k in range(5):
            row[f"cosine(w, PC{k + 1})"] = round(float(np.dot(w_norm, components[k])), 3)
        projection = components[:10] @ w
        row["fraction of ||w||^2 in top-10 PCs"] = round(float(np.sum(projection**2) / np.sum(w**2)), 3)
        return row

    COSINE_TABLE = pd.DataFrame([cosine_row(key) for key in RUNS if CONDITION_DATA.get(key, {}).get("ok")])
    mo.vstack(
        [
            mo.md("_Computed in the probe-LayerNorm space regardless of the toggle above, since `w` is only ever applied after that LayerNorm._"),
            mo.ui.table(COSINE_TABLE, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 6: scree

    Explained variance ratio per component, both conditions, both raw and
    probe-LayerNorm spaces. This is not toggle-controlled: seeing all four
    lines together is the point. If LoRA's normalized space spreads
    variance across more components than the frozen one, that on its own
    would say the adapted representation is less dominated by one or two
    directions -- worth knowing regardless of whether those directions
    happen to align with degeneration.
    """)
    return


@app.cell
def _(FIGURES, PCA_DATA, RUNS, SERIES, missing, mo, np, plt, save, tidy):
    def scree_figure():
        available = [key for key in RUNS if PCA_DATA.get(key, {}).get("ok")]
        if not available:
            return missing("No condition has a fitted PCA yet.")
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        styles = {"raw": "--", "norm": "-"}
        for idx, run_key in enumerate(available):
            for space, style in styles.items():
                ratio = PCA_DATA[run_key][space]["pca"].explained_variance_ratio_
                axis.plot(
                    np.arange(1, len(ratio) + 1), ratio, style,
                    color=SERIES[idx], linewidth=1.8,
                    label=f"{RUNS[run_key]['label']}, {'probe LayerNorm' if space == 'norm' else 'raw'}",
                )
        axis.set_xlabel("component")
        axis.set_ylabel("explained variance ratio")
        axis.legend(loc="best")
        tidy(axis, xgrid=False)
        fig.suptitle("Scree: variance explained per component", x=0.005, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return mo.vstack([save(fig, "fig6_scree", FIGURES), mo.md("_Dashed = raw states, solid = after that condition's own probe LayerNorm._")])

    scree_figure()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bonus: the same picture at a different layer

    Everything above is pinned to layer 15, the one depth both conditions
    share a trained probe for. Every layer was extracted, though, so this
    panel lets a reader pick any of them and see the raw-state PCA
    scatter, colored either by token class or by the layer-15 probe's
    score for that same token. The second option is a deliberate small
    inconsistency: the color comes from a probe that never saw *this*
    layer's representation, only layer 15's, but it is still the same
    token, so overlaying it here is a way to ask "does whatever the layer
    15 probe keys on line up with structure visible earlier or later in
    the stream", not a claim that the probe works at the displayed layer.

    Only raw states are shown here: neither condition has a trained
    LayerNorm at any layer but 15, so a normalized view at another layer
    would be applying the wrong head's normalization.
    """)
    return


@app.cell
def _(mo):
    layer_selector = mo.ui.slider(1, 31, value=15, label="layer")
    layer_color_mode = mo.ui.radio(
        options=["token class", "layer-15 probe score"], value="token class", label="color by"
    )
    mo.hstack([layer_selector, layer_color_mode], justify="start", gap=2)
    return layer_color_mode, layer_selector


@app.cell
def _(
    CONDITION_DATA,
    FIGURES,
    ROLLOUT_MANIFEST,
    RUNS,
    SUBSAMPLE_STRIDE,
    layer_color_mode,
    layer_selector,
    load_condition_tokens,
    missing,
    mo,
    np,
    plt,
    save,
    tidy,
):
    from sklearn.decomposition import PCA as _PCA

    def layer_explorer_figure():
        available = [key for key in RUNS if CONDITION_DATA.get(key, {}).get("ok")]
        if not available:
            return missing("No condition has extracted activations yet.")
        layer = layer_selector.value
        by_score = layer_color_mode.value == "layer-15 probe score"
        fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4.6), squeeze=False)
        axes = axes[0]
        last_scatter = None
        for axis, run_key in zip(axes, available):
            activations_root = RUNS[run_key]["activations_root"]
            vectors, meta = load_condition_tokens(activations_root, ROLLOUT_MANIFEST, layer)
            embedding = _PCA(n_components=2, svd_solver="randomized", random_state=0).fit_transform(vectors)
            keep = np.arange(len(meta)) % SUBSAMPLE_STRIDE == 0
            if by_score and "probe_score" in CONDITION_DATA[run_key]["meta"].columns:
                # Same rollouts, same iteration order, same per-rollout token count
                # at every layer, so the layer-15 probe score lines up with this
                # layer's tokens position-for-position even though the probe never
                # read this layer's representation.
                score = CONDITION_DATA[run_key]["meta"]["probe_score"].values
                last_scatter = axis.scatter(
                    embedding[keep, 0], embedding[keep, 1], c=score[keep],
                    cmap="viridis", vmin=0, vmax=1, s=8, alpha=0.55, linewidths=0,
                )
            else:
                for cls, color in zip(["healthy", "pre_frontier", "in_pattern"], ["#2a78d6", "#eb6834", "#1baf7a"]):
                    mask = keep & (meta["token_class"].values == cls)
                    axis.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.5, color=color, linewidths=0, label=cls)
            axis.set_title(f"{RUNS[run_key]['label']}, layer {layer} (raw)", fontsize=9)
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            tidy(axis, xgrid=False)
        if by_score and last_scatter is not None:
            fig.colorbar(last_scatter, ax=axes.tolist(), pad=0.02, shrink=0.85).set_label("layer-15 probe score", fontsize=7)
        else:
            axes[0].legend(loc="best", fontsize=7)
        fig.suptitle(f"Layer {layer}, raw states, colored by {layer_color_mode.value}", x=0.005, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return mo.vstack([save(fig, "bonus_layer_explorer", FIGURES), mo.md("_PCA is refit at every layer choice; this panel is exploratory, not part of the numbered claims above._")])

    layer_explorer_figure()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What this notebook can and cannot establish

    A probe's own LayerNorm and linear map are trained, under supervision,
    specifically to separate the label they were given. So a
    representation that a trained probe separates well is not surprising
    on its own, in either condition: that is what training does, whether
    or not the underlying representation contains anything a human would
    call "structure." What this notebook can speak to is narrower and more
    useful than that: whether the *same* supervision, applied to a frozen
    representation versus one LoRA was allowed to reshape, finds a more
    linearly *accessible* signal in one than the other, once position and
    the easy in-pattern case are controlled for. That is a claim about
    accessibility to a linear readout, not a claim that either
    representation "understands" degeneration, and not a claim that would
    survive a probe trained to find structure in noise.

    That last risk is exactly why a shuffled-frontier LoRA control is
    required before any of this supports a claim. A LoRA run trained
    against frontiers randomly reassigned to the wrong rollouts would still
    be adapting the representation and training a probe on top of it; if
    *that* control also shows separation in these same figures, the
    separation shown above is a property of supervised training meeting a
    flexible representation, not of degeneration specifically. `RUNS` at
    the top of this notebook is built so that control is one more
    dictionary entry and one more extraction pass away, and every figure
    and table above already iterates over `RUNS` rather than naming
    `frozen` or `lora` by hand.
    """)
    return


if __name__ == "__main__":
    app.run()
