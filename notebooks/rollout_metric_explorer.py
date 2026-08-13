import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import anywidget
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import traitlets
    from sklearn.cluster import KMeans

    from degeneration_probe.data.activation_store import activation_path, load_probe_layer
    from degeneration_probe.dataset_gen import paths
    from degeneration_probe.dataset_gen.config import DatasetGenConfig
    from degeneration_probe.probes.linear_probe import CachedFeatureProbe, read_probe_config
    from degeneration_probe.utils.tokenization import find_string_in_tokens

    return (
        CachedFeatureProbe,
        DatasetGenConfig,
        KMeans,
        Path,
        activation_path,
        anywidget,
        find_string_in_tokens,
        json,
        load_probe_layer,
        mo,
        np,
        paths,
        pd,
        plt,
        read_probe_config,
        torch,
        traitlets,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Rollout metric explorer

    Pick one rollout, then watch a per-token metric move across it: how it
    behaves before the degeneration frontier, at the frontier itself, and
    once inside a repetition loop -- and whether it settles into place once
    it's there.

    The frontier and the loop band are picked by clicking rows in a
    per-token table (Section 4) rather than typing a token index, so the
    position you mark is always an exact token id. Both start with a row
    already selected, prefilled from signals `label.py` already computes
    for this rollout (the digit-run-normalized onset position for the
    frontier, the first exact-repeat span `find_longest_repeated_substring`
    finds for the band) -- clear or change the selection if you disagree
    with the default.

    Two of the metrics (probe score, representation cluster) read a chosen
    layer's cached hidden states directly off disk -- no model, no GPU, just
    the activations `cache_activations.py` already wrote for this dataset.
    """)
    return


@app.cell
def _(Path):
    import os

    REPO_ROOT = Path("/iopsstor/scratch/cscs/mdenegri/degeneration-probe")
    FIGURES_DIR = REPO_ROOT / "notebooks" / "figures" / "rollout_metrics"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR = REPO_ROOT / "outputs"

    DATASET_CONFIG_PATHS = {
        "apertus-8b-instruct": REPO_ROOT / "configs" / "dataset" / "builds" / "degeneration-dataset-apertus-8b-instruct.yaml",
        "apertus1p5-capfilter-linear-it8816": REPO_ROOT / "configs" / "dataset" / "builds" / "degeneration-dataset-apertus1p5-capfilter-linear-it8816.yaml",
        "apertus1p5-sft256k-4200": REPO_ROOT / "configs" / "dataset" / "builds" / "degeneration-dataset-apertus1p5-sft256k-4200.yaml",
    }
    DATASET_TAGS = list(DATASET_CONFIG_PATHS)

    # The tokenizer cache generate.py filled in lives on a different filesystem
    # than REPO_ROOT -- point HF_HOME there and load offline, since the login
    # node has no guaranteed internet access.
    HF_CACHE_DIR = Path(f"/capstor/scratch/cscs/{os.environ['USER']}/degeneration-probe/hf_cache")
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return DATASET_CONFIG_PATHS, DATASET_TAGS, FIGURES_DIR, OUTPUTS_DIR


@app.cell
def _(DATASET_CONFIG_PATHS, DatasetGenConfig, paths):
    DATASET_CONFIGS = {tag: DatasetGenConfig.from_yaml(p) for tag, p in DATASET_CONFIG_PATHS.items()}
    DATASET_DOMAINS = {tag: sorted(paths.configured_domain_names(cfg)) for tag, cfg in DATASET_CONFIGS.items()}
    return DATASET_CONFIGS, DATASET_DOMAINS


@app.cell
def _(DATASET_CONFIGS):
    from transformers import AutoTokenizer

    _tokenizer_cache = {}

    def get_tokenizer(tag):
        if tag not in _tokenizer_cache:
            config = DATASET_CONFIGS[tag]
            tokenizer_name = config.tokenizer_name or config.model_name
            _tokenizer_cache[tag] = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        return _tokenizer_cache[tag]

    return (get_tokenizer,)


@app.cell
def _(anywidget, traitlets):
    # Two clicks select the loop band: the first click is both the band start
    # and the frontier, the second click is the band end. A third click starts
    # a new selection. Selection happens directly in the rendered text instead
    # of a separate token-index table, so there's no risk of picking the wrong
    # row -- and there's no drag to get wrong either.
    _ROLLOUT_TEXT_SELECTOR_ESM = """
    function render({ model, el }) {
      el.classList.add("rts-root");

      const toolbar = document.createElement("div");
      toolbar.className = "rts-toolbar";
      const info = document.createElement("span");
      info.className = "rts-info";
      const clearBtn = document.createElement("button");
      clearBtn.textContent = "clear selection";
      clearBtn.className = "rts-btn";
      toolbar.append(info, clearBtn);

      const legend = document.createElement("div");
      legend.className = "rts-legend";
      legend.innerHTML = `
        <span><i class="rts-swatch rts-swatch-quote"></i> llm-judge onset_quote</span>
        <span><i class="rts-swatch rts-swatch-frontier"></i> 1st click: frontier / band start</span>
        <span><i class="rts-swatch rts-swatch-band"></i> 2nd click: band end</span>
      `;

      const textEl = document.createElement("pre");
      textEl.className = "rts-text";

      el.append(toolbar, legend, textEl);

      const tokens = model.get("tokens");
      const tooltips = model.get("tooltips");
      const spans = tokens.map((tok, i) => {
        const span = document.createElement("span");
        span.className = "rts-tok";
        span.textContent = tok.length ? tok : "\\u200b";
        span.dataset.idx = String(i);
        if (tooltips && tooltips[i]) span.title = tooltips[i];
        textEl.appendChild(span);
        return span;
      });

      // Whether the next click sets the band end (true) or starts a fresh
      // selection (false). Local UI state, not synced to Python.
      let awaitingEnd = model.get("band_start") >= 0 && model.get("band_end") < 0;

      function updateInfo() {
        const f = model.get("frontier_index");
        const bs = model.get("band_start");
        const be = model.get("band_end");
        const parts = [];
        parts.push(f >= 0 ? `frontier: ${f}` : "frontier: none");
        parts.push(bs >= 0 && be >= 0 ? `band: ${bs}-${be}` : awaitingEnd ? "band: click the end token" : "band: none");
        info.textContent = parts.join("   ·   ");
      }

      function applyHighlights() {
        const hs = model.get("highlight_start");
        const he = model.get("highlight_end");
        const f = model.get("frontier_index");
        const bs = model.get("band_start");
        const be = model.get("band_end");
        spans.forEach((span, i) => {
          span.classList.toggle("rts-quote", hs >= 0 && he > hs && i >= hs && i < he);
          span.classList.toggle("rts-band", bs >= 0 && be >= 0 && i >= bs && i <= be);
          span.classList.toggle("rts-frontier", i === f);
        });
        updateInfo();
      }

      function onTextClick(e) {
        const span = e.target.closest(".rts-tok");
        if (!span) return;
        const idx = Number(span.dataset.idx);
        if (!awaitingEnd) {
          model.set("frontier_index", idx);
          model.set("band_start", idx);
          model.set("band_end", -1);
          awaitingEnd = true;
        } else {
          const start = model.get("band_start");
          model.set("band_start", Math.min(start, idx));
          model.set("band_end", Math.max(start, idx));
          model.set("frontier_index", Math.min(start, idx));
          awaitingEnd = false;
        }
        model.save_changes();
        applyHighlights();
      }

      function onClear() {
        model.set("frontier_index", -1);
        model.set("band_start", -1);
        model.set("band_end", -1);
        awaitingEnd = false;
        model.save_changes();
        applyHighlights();
      }

      textEl.addEventListener("click", onTextClick);
      clearBtn.addEventListener("click", onClear);

      model.on("change:frontier_index", applyHighlights);
      model.on("change:band_start", applyHighlights);
      model.on("change:band_end", applyHighlights);
      model.on("change:highlight_start", applyHighlights);
      model.on("change:highlight_end", applyHighlights);

      applyHighlights();
    }
    export default { render };
    """

    _ROLLOUT_TEXT_SELECTOR_CSS = """
    .rts-toolbar { display:flex; align-items:center; gap:0.9em; margin-bottom:0.35em; font-size:0.8em; }
    .rts-info { color:#888; font-variant-numeric: tabular-nums; }
    .rts-btn { font-size:0.85em; padding:1px 8px; border-radius:4px; border:1px solid #8883; background:transparent; cursor:pointer; }
    .rts-btn:hover { background:#8882; }
    .rts-legend { display:flex; gap:1.2em; font-size:0.75em; color:#888; margin-bottom:0.35em; flex-wrap:wrap; }
    .rts-legend span { display:inline-flex; align-items:center; gap:0.35em; }
    .rts-swatch { display:inline-block; width:0.9em; height:0.9em; border-radius:2px; }
    .rts-swatch-quote { background:rgba(253,191,111,0.7); }
    .rts-swatch-frontier { background:transparent; border:2px dotted purple; }
    .rts-swatch-band { background:rgba(128,0,128,0.28); }
    .rts-text {
      white-space: pre-wrap;
      font-family: monospace;
      font-size: 0.85em;
      max-height: 420px;
      overflow-y: auto;
      padding: 0.5em;
      border: 1px solid #8883;
      border-radius: 4px;
      cursor: pointer;
      user-select: none;
      margin: 0;
    }
    .rts-tok { border-radius:2px; }
    .rts-tok:hover { background: rgba(120,120,120,0.2); }
    .rts-quote { background: rgba(253,191,111,0.55); }
    .rts-band { background: rgba(128,0,128,0.22); }
    .rts-frontier { outline: 2px dotted purple; outline-offset: -1px; background: rgba(128,0,128,0.4); }
    """

    class RolloutTextSelector(anywidget.AnyWidget):
        _esm = _ROLLOUT_TEXT_SELECTOR_ESM
        _css = _ROLLOUT_TEXT_SELECTOR_CSS

        tokens = traitlets.List(traitlets.Unicode()).tag(sync=True)
        tooltips = traitlets.List(traitlets.Unicode()).tag(sync=True)
        highlight_start = traitlets.Int(-1).tag(sync=True)
        highlight_end = traitlets.Int(-1).tag(sync=True)
        frontier_index = traitlets.Int(-1).tag(sync=True)
        band_start = traitlets.Int(-1).tag(sync=True)
        band_end = traitlets.Int(-1).tag(sync=True)

    return (RolloutTextSelector,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Pick a rollout
    """)
    return


@app.cell
def _(DATASET_TAGS, mo):
    dataset_dd = mo.ui.dropdown(options=DATASET_TAGS, value=DATASET_TAGS[0], label="dataset")
    dataset_dd
    return (dataset_dd,)


@app.cell
def _(DATASET_CONFIGS, DATASET_DOMAINS, dataset_dd, paths, pd):
    _tag = dataset_dd.value
    _config = DATASET_CONFIGS[_tag]
    _domains = DATASET_DOMAINS[_tag]

    _gen_frames = []
    _lbl_frames = []
    for _domain in _domains:
        _gen_path = paths.generations_shard_path(_config, _domain, 0)
        if _gen_path.exists():
            _gdf = pd.read_parquet(_gen_path)
            _gdf["domain"] = _domain
            _gen_frames.append(_gdf)
        _lbl_path = paths.labels_shard_path(_config, _domain, 0)
        if _lbl_path.exists():
            _ldf = pd.read_parquet(_lbl_path)
            _ldf["domain"] = _domain
            _lbl_frames.append(_ldf)
    generations_df = pd.concat(_gen_frames, ignore_index=True) if _gen_frames else pd.DataFrame()
    labels_df = pd.concat(_lbl_frames, ignore_index=True) if _lbl_frames else pd.DataFrame()

    _judge_dir = paths.llm_judge_dir(_config)
    _judge_files = sorted(_judge_dir.glob("results_*.parquet")) if _judge_dir.exists() else []
    _judge_frames = []
    for _f in _judge_files:
        _jdf = pd.read_parquet(_f)
        _jdf["backend"] = _f.stem.removeprefix("results_")
        _judge_frames.append(_jdf)
    llm_judge_results_df = pd.concat(_judge_frames, ignore_index=True) if _judge_frames else pd.DataFrame()

    # (prompt_id, rollout_idx) -> list of verdict dicts (one per backend with an "ok" result).
    llm_judge_lookup = {}
    if not llm_judge_results_df.empty:
        for _row in llm_judge_results_df[llm_judge_results_df["status"] == "ok"].itertuples(index=False):
            llm_judge_lookup.setdefault((_row.prompt_id, int(_row.rollout_idx)), []).append({
                "backend": _row.backend,
                "onset_quote": _row.onset_quote,
                "is_degenerating": bool(_row.is_degenerating),
            })

    def _verdict_label(entries):
        if not entries:
            return None
        _n = len(entries)
        _n_pos = sum(1 for _e in entries if _e["is_degenerating"])
        if _n_pos == _n:
            return "degenerate"
        if _n_pos == 0:
            return "non-degenerate"
        return f"mixed {_n_pos}/{_n}"

    rollouts_df = pd.DataFrame()
    if not generations_df.empty:
        rollouts_df = generations_df[["domain", "prompt_id", "rollout_idx", "num_tokens", "stop_reason"]].copy()
        rollouts_df["verdict"] = [
            _verdict_label(llm_judge_lookup.get((row.prompt_id, int(row.rollout_idx)), []))
            for row in rollouts_df.itertuples(index=False)
        ]
    return generations_df, labels_df, llm_judge_lookup, rollouts_df


@app.cell
def _(mo, rollouts_df):
    _domain_options = sorted(rollouts_df["domain"].unique()) if not rollouts_df.empty else []
    domain_dd = mo.ui.dropdown(
        options=_domain_options, value=(_domain_options[0] if _domain_options else None), label="domain"
    )
    domain_dd
    return (domain_dd,)


@app.cell
def _(domain_dd, mo, rollouts_df):
    _sub = rollouts_df[rollouts_df["domain"] == domain_dd.value] if not rollouts_df.empty else rollouts_df
    _prompt_ids = sorted(_sub["prompt_id"].unique()) if not _sub.empty else []

    def _prompt_label(prompt_id):
        _prows = _sub[_sub["prompt_id"] == prompt_id]
        _n_total = len(_prows)
        _n_judged = int(_prows["verdict"].notna().sum())
        _n_degenerate = int((_prows["verdict"] == "degenerate").sum())
        if _n_judged == 0:
            return f"{prompt_id}  (not judged, {_n_total} rollouts)"
        return f"{prompt_id}  ({_n_degenerate}/{_n_judged} judged degenerate, {_n_total} rollouts)"

    _options = {_prompt_label(pid): pid for pid in _prompt_ids}
    prompt_dd = mo.ui.dropdown(
        options=_options, value=(next(iter(_options)) if _options else None), label="prompt_id"
    )
    prompt_dd
    return (prompt_dd,)


@app.cell
def _(mo, prompt_dd, rollouts_df):
    _sub = (
        rollouts_df[rollouts_df["prompt_id"] == prompt_dd.value].sort_values("rollout_idx")
        if not rollouts_df.empty else rollouts_df
    )
    _options = {
        (f"{int(_row.rollout_idx)}  [{_row.verdict}]" if _row.verdict is not None else str(int(_row.rollout_idx))): int(_row.rollout_idx)
        for _row in _sub.itertuples(index=False)
    } if not _sub.empty else {}
    rollout_dd = mo.ui.dropdown(
        options=_options, value=(next(iter(_options)) if _options else None), label="rollout_idx"
    )
    rollout_dd
    return (rollout_dd,)


@app.cell
def _(
    dataset_dd,
    find_string_in_tokens,
    generations_df,
    get_tokenizer,
    labels_df,
    llm_judge_lookup,
    np,
    prompt_dd,
    rollout_dd,
    torch,
):
    _gen_row = generations_df.loc[
        (generations_df["prompt_id"] == prompt_dd.value) & (generations_df["rollout_idx"] == rollout_dd.value)
    ].iloc[0]
    _label_match = labels_df.loc[
        (labels_df["prompt_id"] == prompt_dd.value) & (labels_df["rollout_idx"] == rollout_dd.value)
    ]
    label_row = _label_match.iloc[0] if not _label_match.empty else None

    token_ids = [int(t) for t in _gen_row["generated_token_ids"]]
    num_tokens = int(_gen_row["num_tokens"])
    stop_reason = _gen_row["stop_reason"]
    entropy = np.asarray(label_row["entropy"], dtype=float) if label_row is not None else None

    # llm-judge onset_quote, verbatim-matched into this rollout's own tokens (same
    # binary-search lookup resolve_onset_position(metric="onset_quote") uses in
    # production) -- a paraphrased/hallucinated quote has no position and is skipped.
    onset_quote_text = None
    onset_quote_position = None
    onset_quote_end = None
    _tokenizer = get_tokenizer(dataset_dd.value)
    for _entry in llm_judge_lookup.get((prompt_dd.value, int(rollout_dd.value)), []):
        _quote = _entry.get("onset_quote")
        if not _quote:
            continue
        try:
            _span = find_string_in_tokens(_quote, torch.as_tensor(token_ids), _tokenizer)
        except (AssertionError, ValueError):
            continue
        onset_quote_text = _quote
        onset_quote_position = int(_span.start)
        onset_quote_end = int(_span.stop)
        break
    return (
        entropy,
        label_row,
        num_tokens,
        onset_quote_end,
        onset_quote_position,
        stop_reason,
        token_ids,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 2. Metric, layer, and clustering controls
    """)
    return


@app.cell
def _(mo):
    layer_dd = mo.ui.number(start=0, stop=31, step=1, value=30, label="representation layer")
    k_slider = mo.ui.slider(start=2, stop=12, step=1, value=6, label="clusters (k)", show_value=True)
    metric_radio = mo.ui.radio(
        options={
            "entropy": "entropy",
            "probe score (sigmoid)": "probe_sigmoid",
            "probe score (raw logit)": "probe_logit",
        },
        value="entropy",
        label="y-axis metric",
    )
    color_by_cluster_cb = mo.ui.checkbox(value=True, label="color points by representation cluster")
    show_onset_quote_cb = mo.ui.checkbox(value=False, label="mark llm-judge onset_quote position")
    mo.vstack([
        mo.hstack([layer_dd, k_slider]),
        metric_radio,
        mo.hstack([color_by_cluster_cb, show_onset_quote_cb]),
    ])
    return (
        color_by_cluster_cb,
        k_slider,
        layer_dd,
        metric_radio,
        show_onset_quote_cb,
    )


@app.cell
def _(DATASET_CONFIGS, OUTPUTS_DIR, dataset_dd, json):
    # (prompt_id, rollout_idx)-independent index of every finished, non-LoRA,
    # single-layer probe run trained on this dataset -- built once per dataset
    # switch, then filtered by layer below. LoRA runs need the live model (not
    # just cached activations) and multi-layer runs need their own head-per-depth
    # forward, so both are left out of this pass -- pick a different layer/dataset
    # if the one you want isn't showing up here.
    def _scan_probe_runs(build_root):
        by_layer = {}
        for _config_path in sorted(OUTPUTS_DIR.glob("*/*/resolved_config.json")):
            _run_dir = _config_path.parent
            if _run_dir.is_symlink():
                continue
            _info_path = _run_dir / "run_info.json"
            if not _info_path.is_file():
                continue
            try:
                _resolved = json.loads(_config_path.read_text())
                _info = json.loads(_info_path.read_text())
            except ValueError:
                continue
            if _info.get("status") != "finished":
                continue
            if str(_resolved.get("dataset", {}).get("build_root")) != str(build_root):
                continue
            _training = _resolved.get("training", {})
            if _training.get("lora", {}).get("enabled"):
                continue
            _probe_cfg = _training.get("probe", {})
            _layers = _probe_cfg.get("layers")
            if _layers is not None and len(_layers) > 1:
                continue
            _layer = _probe_cfg.get("layer")
            if _layer is None:
                continue
            _checkpoint = _run_dir / "final"
            if not (_checkpoint / "probe_config.json").is_file():
                continue
            _label = f"{_run_dir.parent.name} / {_run_dir.name}"
            by_layer.setdefault(int(_layer), []).append((_label, str(_checkpoint)))
        return by_layer

    probe_runs_by_layer = _scan_probe_runs(DATASET_CONFIGS[dataset_dd.value].output_root)
    return (probe_runs_by_layer,)


@app.cell
def _(layer_dd, mo, probe_runs_by_layer):
    _options = dict(probe_runs_by_layer.get(int(layer_dd.value), []))
    probe_dd = mo.ui.dropdown(
        options=_options,
        value=(next(iter(_options)) if _options else None),
        label=f"probe run at layer {int(layer_dd.value)}",
    )
    probe_dd
    return (probe_dd,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Rollout text -- click to select the loop band

    The frontier is the first token of the loop band, so one selection sets
    both: click a token to mark the band start (and frontier), then click
    another token to mark the band end. Both come out as exact token indices
    directly from the text, no separate table or typing needed. The orange
    highlight is the llm-judge's `onset_quote`. The selection starts prefilled
    from the signals `label.py` already computes for this rollout (the
    digit-run-normalized onset position for the start, the first exact-repeat
    span `find_longest_repeated_substring` finds for the end); click "clear
    selection" and reselect if you disagree with the default.
    """)
    return


@app.cell
def _(dataset_dd, entropy, get_tokenizer, label_row, np, token_ids):
    _tokenizer = get_tokenizer(dataset_dd.value)
    token_raw_texts = [_tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]

    _repetition_score = (
        np.asarray(label_row["repetition_score"], dtype=float) if label_row is not None else None
    )
    token_tooltips = [
        f"#{_i}"
        + (f" · entropy {entropy[_i]:.3f}" if entropy is not None else "")
        + (f" · rep {_repetition_score[_i]:.3f}" if _repetition_score is not None else "")
        for _i in range(len(token_ids))
    ]
    return token_raw_texts, token_tooltips


@app.cell
def _(label_row, pd):
    def _safe_int(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return int(x)

    onset_default = _safe_int(label_row["lrs_first_start_normalized_growing"]) if label_row is not None else None
    _has_lrs = label_row is not None and bool(_safe_int(label_row["lrs_length"]))
    band_start_default = _safe_int(label_row["lrs_region_starts"][0]) if _has_lrs else None
    band_end_default = _safe_int(label_row["lrs_region_ends"][0]) if _has_lrs else None
    return band_end_default, band_start_default, onset_default


@app.cell
def _(
    RolloutTextSelector,
    band_end_default,
    band_start_default,
    mo,
    onset_default,
    onset_quote_end,
    onset_quote_position,
    token_raw_texts,
    token_tooltips,
):
    text_selector = mo.ui.anywidget(
        RolloutTextSelector(
            tokens=token_raw_texts,
            tooltips=token_tooltips,
            highlight_start=(onset_quote_position if onset_quote_position is not None else -1),
            highlight_end=(onset_quote_end if onset_quote_end is not None else -1),
            frontier_index=(onset_default if onset_default is not None else -1),
            band_start=(band_start_default if band_start_default is not None else -1),
            band_end=(band_end_default if band_end_default is not None else -1),
        )
    )
    text_selector
    return (text_selector,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Rendered metric
    """)
    return


@app.cell
def _(
    CachedFeatureProbe,
    DATASET_CONFIGS,
    KMeans,
    activation_path,
    dataset_dd,
    domain_dd,
    k_slider,
    layer_dd,
    load_probe_layer,
    np,
    probe_dd,
    prompt_dd,
    read_probe_config,
    rollout_dd,
    torch,
):
    _build_root = DATASET_CONFIGS[dataset_dd.value].output_root
    _layer = int(layer_dd.value)
    _path = activation_path(_build_root, domain_dd.value, prompt_dd.value, rollout_dd.value)

    features = None
    features_note = None
    if _path.is_file():
        try:
            features = load_probe_layer(
                _build_root, domain_dd.value, prompt_dd.value, rollout_dd.value, probe_layer=_layer,
            ).float()
        except (FileNotFoundError, ValueError) as _exc:
            features_note = str(_exc)
    else:
        features_note = f"no cached activations for this rollout at layer {_layer} ({_path})"

    probe_sigmoid = None
    probe_logit = None
    if features is not None and probe_dd.value is not None:
        _cfg = read_probe_config(probe_dd.value)
        _hidden_size = _cfg["input_size"] // _cfg["context_window_size"]
        _probe = CachedFeatureProbe(hidden_size=_hidden_size, path=probe_dd.value)
        _probe.eval()
        with torch.no_grad():
            _logits = _probe(features.unsqueeze(0))["probe_logits"].squeeze(0)
        probe_logit = _logits.numpy()
        probe_sigmoid = 1.0 / (1.0 + np.exp(-probe_logit))

    # Per-rollout clustering (not fit across the dataset): the question is whether
    # THIS rollout's own tokens revisit the same region of representation space
    # once it starts looping, so each rollout gets its own KMeans fit.
    cluster_labels = None
    if features is not None:
        _n = features.shape[0]
        _k = min(int(k_slider.value), _n)
        if _k >= 2:
            cluster_labels = KMeans(n_clusters=_k, random_state=0, n_init=5).fit_predict(features.numpy())
    return cluster_labels, features_note, probe_logit, probe_sigmoid


@app.cell
def _(
    FIGURES_DIR,
    cluster_labels,
    color_by_cluster_cb,
    dataset_dd,
    domain_dd,
    entropy,
    features_note,
    metric_radio,
    mo,
    num_tokens,
    onset_quote_position,
    plt,
    probe_logit,
    probe_sigmoid,
    prompt_dd,
    rollout_dd,
    show_onset_quote_cb,
    stop_reason,
    text_selector,
):
    _metric_values = {
        "entropy": entropy,
        "probe_sigmoid": probe_sigmoid,
        "probe_logit": probe_logit,
    }[metric_radio.value]
    _metric_label = {
        "entropy": "entropy (nats)",
        "probe_sigmoid": "probe score (sigmoid)",
        "probe_logit": "probe score (raw logit)",
    }[metric_radio.value]

    _frontier_pos = text_selector.frontier_index if text_selector.frontier_index >= 0 else None
    _band_start = text_selector.band_start if text_selector.band_start >= 0 else None
    _band_end = text_selector.band_end if text_selector.band_end >= 0 else None

    if _metric_values is None:
        rollout_figure = mo.md(
            f"**{metric_radio.value}** is not available for this rollout/layer -- "
            f"{features_note or 'no matching trained probe at this layer for this dataset.'}"
        )
    else:
        _x = list(range(len(_metric_values)))
        _fig, _ax = plt.subplots(figsize=(11, 4.2))

        if color_by_cluster_cb.value and cluster_labels is not None:
            _ax.plot(_x, _metric_values, color="0.8", lw=0.8, zorder=1)
            _cmap = plt.get_cmap("tab10")
            for _cid in sorted(set(cluster_labels)):
                _mask = cluster_labels == _cid
                _ax.scatter(
                    [_i for _i in _x if _mask[_i]],
                    [_metric_values[_i] for _i in _x if _mask[_i]],
                    s=12, color=_cmap(_cid % 10), label=f"cluster {_cid}", zorder=2,
                )
        else:
            _ax.plot(_x, _metric_values, color="#3366cc", lw=1.2)

        if _band_start is not None and _band_end is not None and _band_end > _band_start:
            _ax.axvspan(_band_start, _band_end, color="purple", alpha=0.18, label="selected loop band")
        if _frontier_pos is not None:
            _ax.axvline(_frontier_pos, color="purple", linestyle=":", lw=2, label="frontier")
        if show_onset_quote_cb.value and onset_quote_position is not None:
            _ax.axvline(onset_quote_position, color="#d95f02", linestyle="--", lw=1.2, label="onset_quote (judge)")

        _ax.set_xlabel("token index")
        _ax.set_ylabel(_metric_label)
        _ax.set_title(
            f"{dataset_dd.value} / {domain_dd.value} / {prompt_dd.value} / rollout_idx={rollout_dd.value}\n"
            f"stop_reason={stop_reason}  num_tokens={num_tokens}"
        )
        _ax.grid(alpha=0.3)
        _handles, _labels = _ax.get_legend_handles_labels()
        if _handles:
            _ax.legend(_handles, _labels, loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
        _fig.tight_layout()

        _out_stub = (
            FIGURES_DIR
            / f"{dataset_dd.value}_{domain_dd.value}_{prompt_dd.value}_r{rollout_dd.value}_{metric_radio.value}"
        )
        _fig.savefig(_out_stub.with_suffix(".png"), dpi=150, bbox_inches="tight")
        _fig.savefig(_out_stub.with_suffix(".pdf"), bbox_inches="tight")
        rollout_figure = _fig

    rollout_figure
    return


if __name__ == "__main__":
    app.run()
