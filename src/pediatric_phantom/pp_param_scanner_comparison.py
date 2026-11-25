"""
Pediatric Phantom – PARAMETER COMPARISON PCCT vs. EICT (add-on to tasks 1 and 2)
================================================================================
The paper (Fig. 5) only shows the parameter comparison of the PCCT (task 1, "param_PCCT").
The same comparison was also acquired and evaluated on the EICT (SOMATOM X.cite, task 2,
"param_EICT"). This script puts both side by side WITHOUT recomputing anything:

  1) param_EICT_Figure_kVp_Thickness_Kernel_<pair>.svg
       EICT counterpart of Fig. 5 – drawn with the unchanged pp_plots._draw_pairwise_bars /
       _stacked_figure (identical style); the PCCT figure is redrawn the same way for comparison.
  2) param_PCCT_vs_EICT_Figure_CNR_<pair>.svg
       a = tube voltage, b = slice thickness, c = kernel; CNR vs. CTDIvol for BOTH scanners
       (colour = scanner, solid/filled = CNR-maximizing setting, dashed/open = alternative).
       Lines with the true CTDIvol on the x-axis are used because the achieved CTDIvol of the two
       scanners differs slightly per level (e.g. 0.34 vs. 0.37 mGy), so bars cannot be paired.
  3) param_PCCT_vs_EICT_Figure_ratio_<pair>.svg
       CNR ratio CNR-maximizing / alternative setting per scanner (> 1 = setting is better).
  4) CSV with all plotted values: CSV_DIR/plot_values_param_PCCT_vs_EICT.csv

Input: the plot-value CSVs written by pp_plots.render_task (PlotValueExporter):
       CSV_DIR/plot_values_param_PCCT.csv and CSV_DIR/plot_values_param_EICT.csv.
       These contain exactly the values drawn in the existing param_PCCT / param_EICT plots.
Output: PLOTS_DIR/param_PCCT_vs_EICT/ (new folder; existing plots and CSVs are not changed).

Usage (run from inside src/pediatric_phantom):
    python pp_param_scanner_comparison.py                 -> organ pair liver – autochthonous muscle
    python pp_param_scanner_comparison.py liver_spleen    -> another category key from the CSV
"""
from __future__ import annotations
import os
import sys
import csv

from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

import pp_paths as paths
import pp_plots
from pp_plots import _draw_pairwise_bars, _stacked_figure, _new_fig, _save, _legend

OUT_NAME = "param_PCCT_vs_EICT"
SOURCES = {"PCCT": "plot_values_param_PCCT.csv", "EICT": "plot_values_param_EICT.csv"}

# Panel order a, b, c (same as Fig. 5): comparison title in the CSV, CNR-maximizing setting first
PANELS = [
    ("Tube Voltage: 70 kV vs. 120 kV", "70 kV", "120 kV"),
    ("Slice Thickness: 1 mm vs. 3 mm", "3 mm", "1 mm"),
    ("Kernel: Br40 vs. Br64", "Br40", "Br64"),
]
# Scanner styles: same colours as the scanner comparison (Fig. 6: PCCT orange, X.cite blue)
SCANNER_STYLE = {"PCCT": {"color": "tab:orange", "marker": "o", "label": "PCCT (NAEOTOM Alpha)"},
                 "EICT": {"color": "tab:blue", "marker": "s", "label": "EICT (SOMATOM X.cite)"}}
DOSE_TICKS = [0.3, 0.5, 0.75, 1, 2, 3, 4]


# ══════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════
def read_values(scanner, cat_key, metric="CNR"):
    """{comparison title: {group label: [(ctdivol, mean, n), ...]}} for one scanner / organ pair / metric."""
    path = paths.CSV_DIR / SOURCES[scanner]
    if not path.exists():
        raise SystemExit(f"{path} not found -> run pp_plots.py (task {1 if scanner == 'PCCT' else 2}) first.")
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r["metric"] != metric or r["category_key"] != cat_key:
                continue
            out.setdefault(r["comparison"], {}).setdefault(r["group"], []).append(
                (float(r["ctdivol_mGy"]), float(r["mean"]), int(r["n"])))
    for comp in out.values():
        for g in comp:
            comp[g].sort()
    return out


def category_title(cat_key):
    with open(paths.CSV_DIR / SOURCES["PCCT"], newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r["category_key"] == cat_key:
                return r["category"]
    return cat_key


def pairwise_dict(values, comp_title, label1, label2, cat_key):
    """Rebuild the plot-data dict expected by pp_plots._draw_pairwise_bars from the exported values.

    label1/label2 follow the order of the original comparison (Config.COMPARISONS), so that the
    bars and the ratio direction ("ratio_invert") are exactly those of the original figure.
    One value per CTDIvol level is stored ([mean]); _draw_pairwise_bars averages the list -> same bar.
    """
    g = values[comp_title]
    d1 = {c: m for c, m, _ in g.get(label1, [])}
    d2 = {c: m for c, m, _ in g.get(label2, [])}
    keys = sorted(set(d1) & set(d2))
    return {"metric": "cnr", "comp_title": comp_title, "label1": label1, "label2": label2,
            "matches": [(c, c, c) for c in keys],
            "map1": {c: {cat_key: [d1[c]]} for c in keys},
            "map2": {c: {cat_key: [d2[c]]} for c in keys}}


# Original group order per comparison (= order of label1/label2 in pp_config.Config.COMPARISONS)
ORIGINAL_ORDER = {
    "Tube Voltage: 70 kV vs. 120 kV": ("70 kV", "120 kV"),
    "Slice Thickness: 1 mm vs. 3 mm": ("1 mm", "3 mm"),
    "Kernel: Br40 vs. Br64": ("Br40", "Br64"),
}


# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════
def _dose_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.28, 4.6)                                  # identical x-range in all panels
    ax.xaxis.set_major_locator(FixedLocator(DOSE_TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in DOSE_TICKS])
    ax.set_xlabel("CTDIvol (mGy)")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _draw_lines(ax, data, comp_title, best, alt):
    """CNR vs. CTDIvol: one line per scanner and setting (solid/filled = best, dashed/open = alternative)."""
    for scanner, vals in data.items():
        st = SCANNER_STYLE[scanner]
        for grp, ls, filled in ((best, "-", True), (alt, "--", False)):
            pts = vals.get(comp_title, {}).get(grp, [])
            if not pts:
                continue
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(x, y, ls=ls, marker=st["marker"], color=st["color"], ms=5, lw=1.6,
                    mfc=st["color"] if filled else "white", mec=st["color"], zorder=3)
    _dose_axis(ax)
    ax.set_ylabel("CNR")
    ax.set_ylim(bottom=0)
    handles = ([Line2D([], [], color=SCANNER_STYLE[s]["color"], marker=SCANNER_STYLE[s]["marker"], lw=1.6,
                       label=s) for s in data]
               + [Line2D([], [], color="black", ls="-", marker="o", mfc="black", lw=1.4, label=best),
                  Line2D([], [], color="black", ls="--", marker="o", mfc="white", lw=1.4, label=alt)])
    _legend(ax, headroom=1.45, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, handles=handles,
            handlelength=2.2, columnspacing=1.0)


def _draw_ratio(ax, data, comp_title, best, alt):
    """Ratio CNR(best) / CNR(alternative) per scanner at each CTDIvol level of that scanner."""
    for scanner, vals in data.items():
        st = SCANNER_STYLE[scanner]
        g = vals.get(comp_title, {})
        b = {c: m for c, m, _ in g.get(best, [])}
        a = {c: m for c, m, _ in g.get(alt, [])}
        keys = sorted(set(b) & set(a))
        ax.plot(keys, [b[k] / a[k] for k in keys], marker=st["marker"], color=st["color"], ms=5, lw=1.6,
                label=scanner, zorder=3)
    ax.axhline(1.0, color="gray", ls=":", lw=1.2, zorder=1)
    _dose_axis(ax)
    ax.set_ylabel(f"CNR ratio {best} / {alt}")
    ax.set_ylim(bottom=0)
    _legend(ax, headroom=1.15, loc="lower right")            # area below ratio = 1 is always empty


def export_csv(data, cat_key, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["scanner", "comparison", "category_key", "setting", "role", "ctdivol_mGy", "CNR", "n",
                    "ratio_best_alt_at_this_level"])
        for scanner, vals in data.items():
            for comp_title, best, alt in PANELS:
                g = vals.get(comp_title, {})
                b = {c: m for c, m, _ in g.get(best, [])}
                a = {c: m for c, m, _ in g.get(alt, [])}
                for grp, role in ((best, "CNR-maximizing"), (alt, "alternative")):
                    for c, m, n in g.get(grp, []):
                        ratio = b[c] / a[c] if (c in b and c in a and a[c]) else float("nan")
                        w.writerow([scanner, comp_title, cat_key, grp, role, f"{c:.3f}", f"{m:.4f}", n,
                                    f"{ratio:.3f}"])
    print(f"  CSV: {path}")


# ══════════════════════════════════════════════════════════════════
def main(argv):
    cat_key = argv[0] if argv else "liver_muscle"
    cat_title = category_title(cat_key)
    data = {s: read_values(s, cat_key) for s in SOURCES}
    out_dir = paths.PLOTS_DIR / OUT_NAME
    pp_plots._OUT_DIR = str(out_dir)                         # same mechanism as pp_plots.render()
    os.makedirs(out_dir, exist_ok=True)
    print(f"Organ pair: {cat_title} -> {out_dir}")

    # 1) Fig. 5 style per scanner (unchanged drawing function, values from the exported CSV)
    for scanner, vals in data.items():
        missing = [t for t, _, _ in PANELS if t not in vals]
        if missing:
            print(f"  {scanner}: missing comparisons {missing} -> Fig. 5-style figure skipped")
            continue
        draws = [(lambda ax, dd=pairwise_dict(vals, t, *ORIGINAL_ORDER[t], cat_key):
                  _draw_pairwise_bars(ax, dd, cat_key, cat_title)) for t, _, _ in PANELS]
        _stacked_figure(draws, f"param_{scanner}", "Figure_kVp_Thickness_Kernel", cat_title)

    # 2) + 3) both scanners in one figure
    _stacked_figure([(lambda ax, p=p: _draw_lines(ax, data, *p)) for p in PANELS],
                    OUT_NAME, "Figure_CNR", cat_title)
    _stacked_figure([(lambda ax, p=p: _draw_ratio(ax, data, *p)) for p in PANELS],
                    OUT_NAME, "Figure_ratio", cat_title)
    for i, p in enumerate(PANELS):                           # single plots
        fig, ax = _new_fig(); _draw_lines(ax, data, *p); _save(fig, OUT_NAME, "CNR", p[1], "vs", p[2], cat_key)
        fig, ax = _new_fig(); _draw_ratio(ax, data, *p); _save(fig, OUT_NAME, "ratio", p[1], "vs", p[2], cat_key)

    export_csv(data, cat_key, paths.CSV_DIR / f"plot_values_{OUT_NAME}.csv")

    # Console summary: is the CNR-maximizing setting better at every dose level on both scanners?
    print("\nRatio CNR-maximizing / alternative setting (min – max over all dose levels):")
    for comp_title, best, alt in PANELS:
        for scanner, vals in data.items():
            g = vals.get(comp_title, {})
            b = {c: m for c, m, _ in g.get(best, [])}
            a = {c: m for c, m, _ in g.get(alt, [])}
            r = [b[k] / a[k] for k in sorted(set(b) & set(a))]
            if r:
                print(f"  {best:6s} vs {alt:6s} {scanner}: {min(r):.2f} – {max(r):.2f} "
                      f"({len(r)} levels, {'always > 1' if min(r) > 1 else 'NOT always > 1'})")


if __name__ == "__main__":
    main(sys.argv[1:])
