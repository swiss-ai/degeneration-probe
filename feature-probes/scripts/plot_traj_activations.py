"""2D projections (PCA, UMAP, t-SNE) of per-token activations.

Reads one or more .npz files produced by extract_traj_activations.py and writes
PNG plots: for each requested layer a row with three columns (PCA / UMAP / t-SNE),
colored by token position with markers distinguishing clean vs degenerate tokens.

If two .npz files are passed (base and LoRA), produces side-by-side figures so the
two variants can be compared at a glance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap


def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {
        "layers": data["layers"].tolist(),
        "degenerating": data["degenerating"],
        "repetition": data["repetition"],
        "token_position": data["token_position"],
        "variant": str(data["variant"]),
        "row_index": int(data["row_index"]),
        "source_dataset": str(data["source_dataset"]),
    }
    out["acts"] = {f"layer_{l}": data[f"layer_{l}"] for l in out["layers"]}
    return out


def project_all(activations: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    n = activations.shape[0]
    pca = PCA(n_components=2, random_state=seed).fit_transform(activations)
    perplexity = max(5, min(30, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    ).fit_transform(activations)
    n_neighbors = max(5, min(15, n - 1))
    umap_model = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=0.1,
    )
    um = umap_model.fit_transform(activations)
    return {"PCA": pca, "UMAP": um, "t-SNE": tsne}


def plot_one_variant(data: dict, out_path: Path, seed: int = 0) -> None:
    layers = data["layers"]
    degen = data["degenerating"]
    pos = data["token_position"]
    n_layers = len(layers)
    methods = ["PCA", "UMAP", "t-SNE"]

    fig, axes = plt.subplots(
        n_layers, 3,
        figsize=(13, 3.6 * n_layers),
        squeeze=False,
    )

    for r, layer in enumerate(layers):
        acts = data["acts"][f"layer_{layer}"]
        projs = project_all(acts, seed=seed)
        for c, method in enumerate(methods):
            ax = axes[r, c]
            xy = projs[method]
            # color by token position; marker distinguishes clean (circle) vs degenerate (x)
            clean_mask = ~degen
            sc1 = ax.scatter(
                xy[clean_mask, 0],
                xy[clean_mask, 1],
                c=pos[clean_mask],
                cmap="viridis",
                s=12,
                alpha=0.75,
                marker="o",
                linewidths=0,
                label="clean",
            )
            ax.scatter(
                xy[degen, 0],
                xy[degen, 1],
                c=pos[degen],
                cmap="viridis",
                s=30,
                alpha=0.95,
                marker="X",
                edgecolors="red",
                linewidths=0.7,
                label="degen",
            )
            ax.set_title(f"layer {layer}  -  {method}", fontsize=10)
            ax.tick_params(labelsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"layer {layer}", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(loc="best", fontsize=7, frameon=True)
        # one shared colorbar per row
        cbar = fig.colorbar(sc1, ax=axes[r, :].tolist(), shrink=0.85, pad=0.01)
        cbar.set_label("token position", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Apertus-8B token-level activations  -  variant={data['variant']}  "
        f"row={data['row_index']}  ({data['source_dataset']})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_compare(datas: list[dict], out_path: Path, seed: int = 0) -> None:
    """Side-by-side comparison: each layer = one row, columns = method x variant."""
    layers = datas[0]["layers"]
    n_layers = len(layers)
    methods = ["PCA", "UMAP", "t-SNE"]
    variants = [d["variant"] for d in datas]
    n_cols = len(methods) * len(variants)

    fig, axes = plt.subplots(
        n_layers, n_cols,
        figsize=(4.0 * n_cols, 3.6 * n_layers),
        squeeze=False,
    )

    last_sc = None
    for r, layer in enumerate(layers):
        # Project each variant independently
        projs_per_variant = []
        for d in datas:
            projs_per_variant.append(project_all(d["acts"][f"layer_{layer}"], seed=seed))
        for v_idx, (d, projs) in enumerate(zip(datas, projs_per_variant)):
            degen = d["degenerating"]
            pos = d["token_position"]
            clean_mask = ~degen
            for m_idx, method in enumerate(methods):
                c = v_idx * len(methods) + m_idx
                ax = axes[r, c]
                xy = projs[method]
                sc1 = ax.scatter(
                    xy[clean_mask, 0],
                    xy[clean_mask, 1],
                    c=pos[clean_mask],
                    cmap="viridis",
                    s=12,
                    alpha=0.75,
                    marker="o",
                    linewidths=0,
                )
                ax.scatter(
                    xy[degen, 0],
                    xy[degen, 1],
                    c=pos[degen],
                    cmap="viridis",
                    s=30,
                    alpha=0.95,
                    marker="X",
                    edgecolors="red",
                    linewidths=0.7,
                )
                ax.set_title(f"layer {layer}  -  {method}  ({d['variant']})", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                last_sc = sc1

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01)
        cbar.set_label("token position", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Apertus-8B token-level activations  -  variants={variants}  "
        f"row={datas[0]['row_index']}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more .npz files from extract_traj_activations.py",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datas = [load_npz(Path(p)) for p in args.inputs]
    for d in datas:
        out_path = out_dir / f"row{d['row_index']}_{d['variant']}.png"
        plot_one_variant(d, out_path, seed=args.seed)

    if len(datas) >= 2:
        out_path = out_dir / f"row{datas[0]['row_index']}_compare.png"
        plot_compare(datas, out_path, seed=args.seed)


if __name__ == "__main__":
    main()
