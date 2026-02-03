# plot.py
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLOR_MAP = {
    "Home": "green",
    "Northwestern": "purple",
    "Wisconsin": "red",
    "Abroad": "blue",
}
DEFAULT_COLOR = "gray"


def parse_axes_arg(s: str):
    return [a.strip().lower() for a in s.split(",") if a.strip()]


def plot_numberline(ax, df_plot: pd.DataFrame, xcol: str, jitter: float, seed: int, title: str, hover: bool):
    x = df_plot[xcol].to_numpy()
    rng = np.random.default_rng(seed)
    y = rng.uniform(-jitter, jitter, size=len(df_plot))

    # baseline
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1, linestyle="--", alpha=0.5)

    # per-context scatter (nice legend grouping)
    artists = []
    artist_to_rows = {}

    for ctx in sorted(df_plot["context"].unique()):
        mask = (df_plot["context"] == ctx).values

        sc = ax.scatter(
            df_plot.loc[mask, xcol].values,
            y[mask],
            label=ctx,
            s=55,
            alpha=0.9,
            edgecolors="none",
            color=COLOR_MAP.get(ctx, DEFAULT_COLOR),
        )
        artists.append(sc)
        artist_to_rows[sc] = df_plot.index[mask].to_numpy()

    # labels
    pad = 0.15 * (np.nanmax(x) - np.nanmin(x) if np.nanmax(x) != np.nanmin(x) else 1.0)
    ax.set_xlim(np.nanmin(x) - pad, np.nanmax(x) + pad)
    ax.set_yticks([])
    ax.set_xlabel(f"{xcol}   (more left  ←   →  more right)")
    ax.set_title(title)

    if hover:
        try:
            import mplcursors
        except ImportError:
            raise RuntimeError("Hover requested, but mplcursors not installed. Run: pip install mplcursors")

        cursor = mplcursors.cursor(artists, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            sc = sel.artist
            i = sel.index
            row_idx = artist_to_rows[sc][i]
            row = df_plot.loc[row_idx]

            traits = str(row.get("traits_text", "")).strip()
            ctx = str(row.get("context", "")).strip()
            yrs = str(row.get("years_bucket", "")).strip()
            rel = str(row.get("relationship", "")).strip()
            score = float(row[xcol])

            sel.annotation.set_text(
                f"{ctx} | {yrs} | {rel}\n{xcol}: {score:.3f}\n{traits}"
            )
            sel.annotation.get_bbox_patch().set_alpha(0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="perception_scored.csv")
    parser.add_argument("--axes", default="spontaneity", help="comma-separated axis names (e.g. spontaneity,risk)")
    parser.add_argument("--metric", default="z", choices=["z", "raw"], help="which metric column to plot per axis")
    parser.add_argument("--show-all", action="store_true", help="include low_confidence rows if present")
    parser.add_argument("--save", default=None, help="save figure to a PNG path instead of showing")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jitter", type=float, default=0.02)
    parser.add_argument("--hover", action="store_true", help="enable hover tooltips (interactive only)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    axes = parse_axes_arg(args.axes)

    # filter confidence
    df_plot = df.copy()
    if (not args.show_all) and ("low_confidence" in df_plot.columns):
        df_plot = df_plot[~df_plot["low_confidence"]].copy()

    if "context" not in df_plot.columns:
        df_plot["context"] = "Unknown"

    # build x columns list and validate
    xcols = []
    for axis_name in axes:
        xcol = f"{axis_name}_{args.metric}"
        if xcol not in df_plot.columns:
            raise ValueError(
                f"Missing column '{xcol}' in {args.csv}. "
                f"Did you score that axis? Try: python main.py --axes {axis_name}"
            )
        df_plot[xcol] = pd.to_numeric(df_plot[xcol], errors="coerce")
        xcols.append(xcol)

    # drop rows that are NaN for all requested plots
    df_plot = df_plot.dropna(subset=xcols, how="any")
    if df_plot.empty:
        raise ValueError("No rows to plot after filtering/NaN removal.")

    # one subplot per axis
    fig_h = max(2.6, 2.2 * len(xcols))
    fig, axs = plt.subplots(nrows=len(xcols), ncols=1, figsize=(12, fig_h), squeeze=False)
    axs = axs[:, 0]

    for i, (axis_name, xcol) in enumerate(zip(axes, xcols)):
        title = f"{axis_name} ({args.metric}) — colored by context"
        if (not args.show_all) and ("low_confidence" in df.columns):
            title += " — high confidence only"
        plot_numberline(
            ax=axs[i],
            df_plot=df_plot,
            xcol=xcol,
            jitter=args.jitter,
            seed=args.seed + i,   # vary per plot so points don't align perfectly across plots
            title=title,
            hover=args.hover,
        )

    # legend once (top)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save:
        plt.savefig(args.save, dpi=220)
        print(f"Saved plot to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
