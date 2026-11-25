"""
Technical phantom – CNR vs. dose PER INSERT and PER CONTRAST GROUP (add-on to mode 4)
=====================================================================================
Reviewer comment on the dose test (Table 1, Test No. 3): averaging the CNR over all seven CTP404
inserts (Air, PMP, LDPE, Polystyrene, Acrylic, Delrin, Teflon) mixes values that differ by up to
two orders of magnitude (e.g. Air ≈ 17 vs. Acrylic ≈ 0.4 on PCCT at 1.13 mGy), so the mean is
dominated by the high-contrast inserts and is hard to interpret.

This script therefore shows the SAME CNR values that enter `tp_plotdata.cnr_dose` (mean CNR,
Fig. 3a of the paper) without averaging over all inserts:
  1) one bar plot per insert (PCCT vs. EICT per CTDIvol group, ratio PCCT/EICT annotated),
  2) one bar plot per contrast group (CNR averaged only within the group, see CNR_GROUPS),
  3) multi-panel figures (all inserts; contrast groups; contrast groups + ΥNPS).

Nothing is recomputed from DICOM and no existing result or plot is changed:
  * the per-insert CNR values are read from the dose-series CSV written by tp_compute.py
    (tp_export, columns "CNR_<insert>", one row per dataset),
  * datasets are grouped by CTDIvol with the unchanged `tp_plotdata.dose_groups`
    (same tolerance DOSE_TOLERANCE_MGY as the original mean-CNR plot),
  * bars, ratio brackets and multi-panel layout are drawn with the unchanged helpers of tp_plots.py,
  * ΥNPS (panel c of the group figure) is taken from the newest dose-series .pkl (kind "nps_dose").
As a consistency check, the mean over all inserts is recomputed and compared with the "cnr_dose"
entry of that .pkl (must agree with the published Fig. 3a).

Output (new sub-folder, existing plots are not touched):
    PLOTS_DIR/dose_cnr_by_insert/*.svg
    output_data/technical_phantom/csv/dose_cnr_by_insert_values.csv

Usage (run from inside src/technical_phantom, like the other tp_* scripts):
    python tp_cnr_by_insert.py                  -> newest dose-series CSV + newest dose .pkl
    python tp_cnr_by_insert.py path\\to\\qa_results_xxx.csv
"""
import os
import sys
import csv
import glob
import pickle

import numpy as np
from matplotlib.figure import Figure

import tp_plots
from tp_plots import (_draw_dose_bars, _draw_nps_dose, _stacked_figure, _new_fig, _save,
                      FIG_SIZE, MULTI_PANEL_H, MULTI_MARGINS_IN, PANEL_LABEL_FMT, PANEL_LABEL_SIZE,
                      PANEL_LABEL_X_IN)
from tp_plotdata import dose_groups
from tp_config import INSERT_NAMES, NOMINAL_HU, DATA_DIR, RESULTS_DIR, PLOTS_DIR, DOSE_TOLERANCE_MGY

# ── Settings ──────────────────────────────────────────────────────
# Contrast groups. Split by the nominal CT number of the insert (tp_config.NOMINAL_HU):
# Air (-1000 HU) and Teflon (+990 HU) differ by roughly 1000 HU from the background,
# all other inserts lie between -200 and +340 HU. Edit here if another grouping is preferred.
CNR_GROUPS = {
    "High": ["Air", "Teflon"],
    "Low": ["PMP", "LDPE", "Polystyrene", "Acrylic", "Delrin"],
}
GROUP_LABELS = {"High": "High-contrast inserts", "Low": "Low-contrast inserts"}

# Panel order of the per-insert multi-panel figure: high-contrast inserts first
INSERT_ORDER = CNR_GROUPS["High"] + CNR_GROUPS["Low"]

OUT_SUBDIR = "dose_cnr_by_insert"
CSV_DIR = os.path.join(DATA_DIR, "technical_phantom", "csv")
OUT_CSV = os.path.join(CSV_DIR, "dose_cnr_by_insert_values.csv")


# ══════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════
def _is_dose_csv(path):
    """True if the QA CSV contains CTDIvol + per-insert CNR for PCCT AND EICT at >= 2 dose levels."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh, delimiter=";"))
    except Exception:
        return False
    if not rows or "CNR_Air" not in rows[0] or "CTDIvol (mGy)" not in rows[0]:
        return False
    types = {r.get("Scanner_Type") for r in rows}
    doses = {r.get("CTDIvol (mGy)") for r in rows}
    # the dose series is the only run with PCCT + EID rows at several low CTDIvol levels (< 2 mGy)
    try:
        low = all(float(d) < 2.0 for d in doses)
    except ValueError:
        return False
    return {"PCCT", "EID"} <= types and len(doses) >= 4 and low


def newest_dose_csv():
    """Newest qa_results_*.csv in CSV_DIR that belongs to a dose-series run (see _is_dose_csv)."""
    files = sorted(glob.glob(os.path.join(CSV_DIR, "qa_results_*.csv")), key=os.path.getmtime, reverse=True)
    for f in files:
        if _is_dose_csv(f):
            return f
    return None


def newest_dose_item(kind):
    """(file name, data) of the plot entry `kind` from the newest dose-series .pkl that contains it.

    Not every dose run stores every kind (e.g. a run without NPS has no "nps_dose"), so the
    newest .pkl that actually contains `kind` is used. Returns (None, None) if none does.
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "dose_*.pkl")), key=os.path.getmtime, reverse=True)
    for f in files:
        with open(f, "rb") as fh:
            bundle = pickle.load(fh)
        for it in bundle.get("plots", []):
            if it.get("kind") == kind:
                return os.path.basename(f), it["data"]
    return None, None


def read_results(path):
    """Read the QA CSV into minimal result dicts {"label", "ctdivol_mgy", "scanner_type", "cnr_by_material"}."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    out = []
    for r in rows:
        cnr = {}
        for n in INSERT_NAMES:
            v = r.get(f"CNR_{n}", "")
            if v not in ("", None):
                cnr[n] = float(v)
        if not cnr or r.get("CTDIvol (mGy)") in ("", None):
            continue
        out.append({"label": r["Dataset"], "ctdivol_mgy": float(r["CTDIvol (mGy)"]),
                    "scanner_type": r["Scanner_Type"], "cnr_by_material": cnr})
    return out


# ══════════════════════════════════════════════════════════════════
# DATA (grouped by dose with the unchanged tp_plotdata.dose_groups)
# ══════════════════════════════════════════════════════════════════
def cnr_dose_for(results, materials):
    """dose_groups dict of the CNR averaged over `materials` per dataset (one material -> its own CNR)."""
    entries = []
    for r in results:
        vals = [r["cnr_by_material"][m] for m in materials if m in r["cnr_by_material"]]
        if vals:
            entries.append((r["ctdivol_mgy"], r["scanner_type"], float(np.mean(vals))))
    return dose_groups(entries, DOSE_TOLERANCE_MGY) if entries else None


def _ratio(p, e):
    return p / e if (np.isfinite(p) and np.isfinite(e) and e != 0) else float("nan")


def export_csv(per_insert, per_group, all_mean, path):
    """One row per (quantity, dose group): PCCT, EICT and the ratio PCCT/EICT."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["quantity", "materials", "ctdivol_group_mGy", "CNR_PCCT", "CNR_EICT", "ratio_PCCT_EICT"])
        blocks = ([(f"insert:{m}", m, g) for m, g in per_insert.items()]
                  + [(f"group:{k}", ", ".join(CNR_GROUPS[k]), g) for k, g in per_group.items()]
                  + [("all_inserts_mean (original Fig. 3a)", ", ".join(INSERT_NAMES), all_mean)])
        for name, mats, g in blocks:
            for d, p, e in zip(g["doses"], g["PCCT"], g["EID"]):
                w.writerow([name, mats, f"{d:.3f}", f"{p:.3f}", f"{e:.3f}", f"{_ratio(p, e):.2f}"])
    print(f"  CSV: {path}")


# ══════════════════════════════════════════════════════════════════
# PLOTS (drawing via the unchanged tp_plots helpers)
# ══════════════════════════════════════════════════════════════════
def _draw_insert(ax, g, material):
    _draw_dose_bars(ax, g, f"CNR ({material})", "", f"CNR {material} vs. Dose")


def _draw_group(ax, g, key):
    _draw_dose_bars(ax, g, f"Mean CNR ({key.lower()} contrast)", "",
                    f"{GROUP_LABELS[key]}: {', '.join(CNR_GROUPS[key])}")


def _grid_figure(draws, ncols, *name_parts):
    """Multi-panel grid (row-major, letters a, b, c ...) with exactly the same cell size, margins and
    letter placement as tp_plots._stacked_figure (which only offers 1, 2, 3 or a vertical stack)."""
    n = len(draws)
    nrows = int(np.ceil(n / ncols))
    W_cell, H_cell, m = FIG_SIZE[0], MULTI_PANEL_H, MULTI_MARGINS_IN
    W, H = W_cell * ncols, H_cell * nrows
    fig = Figure(figsize=(W, H))
    ax_w, ax_h = W_cell - m["left"] - m["right"], H_cell - m["bottom"] - m["top"]
    for i, draw in enumerate(draws):
        r, c = divmod(i, ncols)
        ox, oy = c * W_cell, r * H_cell
        ax = fig.add_axes([(ox + m["left"]) / W, (H - (oy + H_cell) + m["bottom"]) / H, ax_w / W, ax_h / H])
        draw(ax); ax.set_title("")
        fig.text((ox + PANEL_LABEL_X_IN) / W, 1 - (oy + 0.03) / H, PANEL_LABEL_FMT.format(chr(97 + i)),
                 ha="left", va="top", fontweight="bold", fontsize=PANEL_LABEL_SIZE)
    _save(fig, *name_parts, size=(W, H), strip_titles=False)


def plot_all(per_insert, per_group, g_nps):
    tp_plots._OUT_DIR = os.path.join(PLOTS_DIR, OUT_SUBDIR)   # same mechanism as tp_plots.render()
    print(f"\nPlotting -> {tp_plots._OUT_DIR}")
    # 1) single plots
    for m, g in per_insert.items():
        fig, ax = _new_fig(); _draw_insert(ax, g, m); _save(fig, "dose", "CNR", m)
    for k, g in per_group.items():
        fig, ax = _new_fig(); _draw_group(ax, g, k); _save(fig, "dose", "CNR_group", k)
    # 2) multi-panel: all inserts (2 columns)
    mats = [m for m in INSERT_ORDER if m in per_insert]
    _grid_figure([(lambda ax, m=m: _draw_insert(ax, per_insert[m], m)) for m in mats], 2,
                 "dose", "Figure_CNR_by_insert")
    # 3) multi-panel: a = high contrast, b = low contrast
    keys = [k for k in CNR_GROUPS if k in per_group]
    _stacked_figure([(lambda ax, k=k: _draw_group(ax, per_group[k], k)) for k in keys],
                    "dose", "Figure_CNR_groups")
    # 4) drop-in replacement for the published Fig. 3a/b: a = high, b = low, c = ΥNPS
    if g_nps is not None and len(keys) == 2:
        _stacked_figure([(lambda ax, k=k: _draw_group(ax, per_group[k], k)) for k in keys]
                        + [lambda ax: _draw_nps_dose(ax, g_nps)],
                        "dose", "Figure_CNR_groups_UpsilonNPS")


# ══════════════════════════════════════════════════════════════════
def main(argv):
    path = argv[0] if argv else newest_dose_csv()
    if not path or not os.path.exists(path):
        raise SystemExit("No dose-series QA CSV found -> run tp_compute.py (mode 4) first or pass the CSV path.")
    print(f"Dose-series CSV: {path}")
    results = read_results(path)
    for r in results:
        print(f"  {r['label']:10s} {r['scanner_type']:4s} {r['ctdivol_mgy']:.2f} mGy")

    per_insert = {m: cnr_dose_for(results, [m]) for m in INSERT_NAMES}
    per_insert = {m: g for m, g in per_insert.items() if g}
    per_group = {k: cnr_dose_for(results, mats) for k, mats in CNR_GROUPS.items()}
    per_group = {k: g for k, g in per_group.items() if g}
    all_mean = cnr_dose_for(results, INSERT_NAMES)           # = original mean CNR (Fig. 3a)

    # Consistency check against the original dose-series results (.pkl written by tp_compute.py)
    pkl_cnr, d_cnr = newest_dose_item("cnr_dose")
    if d_cnr:
        g_orig = d_cnr["g"]
        diff = max(abs(a - b) for s in ("PCCT", "EID") for a, b in zip(g_orig[s], all_mean[s]))
        print(f"\nCheck vs. {pkl_cnr}: max |mean CNR (CSV) - mean CNR (.pkl)| = {diff:.4f}"
              + ("  -> identical" if diff < 1e-3 else "  -> WARNING: CSV and .pkl differ!"))
    pkl_nps, d_nps = newest_dose_item("nps_dose")
    g_nps = d_nps["g"] if d_nps else None
    if g_nps:
        print(f"ΥNPS taken from {pkl_nps}: PCCT {[round(v, 1) for v in g_nps['PCCT']]}, "
              f"EICT {[round(v, 1) for v in g_nps['EID']]}")

    print("\nCNR PCCT / EICT per dose group (mGy):")
    for name, g in list(per_insert.items()) + [(f"[{k}]", g) for k, g in per_group.items()] + [("[all, mean]", all_mean)]:
        txt = "  ".join(f"{d:.2f}: {p:6.2f}/{e:5.2f} (x{_ratio(p, e):.2f})" for d, p, e in zip(g["doses"], g["PCCT"], g["EID"]))
        print(f"  {name:13s} {txt}")

    export_csv(per_insert, per_group, all_mean, OUT_CSV)
    plot_all(per_insert, per_group, g_nps)


if __name__ == "__main__":
    main(sys.argv[1:])
