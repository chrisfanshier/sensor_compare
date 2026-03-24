"""
Prototype: core-barrel geometry at trip vs at start-of-penetration.

Loads a corrected sensor CSV, computes piston position and seafloor,
finds start_penetration (where core tip reaches seafloor), and draws
two side-by-side column groups comparing the geometry.

Left group  — referenced to WS @ trip:
  ┌─────────────┐  WS @ trip
  │   1.5 m     │  offset
  ├─────────────┤  WS @ trip + 1.5
  │ core_length │  barrel
  ├─────────────┤  WS @ trip + 1.5 + core_length  (core tip @ trip)
  │ freefall    │
  └─────────────┘  seafloor

Right group — referenced to WS @ start_penetration:
  ┌─────────────┐  WS @ start_pen
  │   1.5 m     │  offset
  ├─────────────┤  WS @ start_pen + 1.5
  │ core_length │  barrel
  ├─────────────┤  seafloor  (core tip touches seafloor here)
  + piston_alt  │  piston altitude @ start_pen → seafloor

Usage:
  python plot_penetration_prototype.py [corrected_csv]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────
FT_TO_M = 1.0 / 3.28

# Parameters for 14JC (same as the calculation report)
CORE_LENGTH_FT = 50.0
TRIGGER_CORE_LENGTH_FT = 10.0
TRIGGER_PEN_M = 1.5
SCOPE_FT = 22.5              # scope for this cast (feet)
OFFSET_CONSTANT = 1.25       # piston offset constant (metres)
WEIGHT_LENGTH = 1.5          # length of the weight (metres)


# ── Load & compute ────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load corrected sensor CSV, skipping '#' comment lines."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('datetime'):
            header_idx = i
            break
    df = pd.read_csv(path, skiprows=header_idx, parse_dates=['datetime'])
    return df


def find_columns(df: pd.DataFrame):
    """Identify weight-stand, release, and trigger columns by name pattern."""
    cols = df.columns.tolist()
    ws = [c for c in cols if 'Weight Stand' in c and 'Depth' in c]
    rel = [c for c in cols if 'Release' in c and 'Depth' in c]
    trig = [c for c in cols if 'Trigger' in c and 'Depth' in c]
    return ws[0] if ws else None, rel[0] if rel else None, trig[0] if trig else None


def find_trip_index(path: str) -> int | None:
    """Extract trip index from header comment."""
    import re
    text = Path(path).read_text(encoding='utf-8')
    m = re.search(r'Trip detected at:.*\(index:\s*(\d+)\)', text)
    return int(m.group(1)) if m else None


def compute_piston(ws, rel, scope_ft, core_length_ft, start_core_idx,
                   offset=OFFSET_CONSTANT):
    """Compute piston position array (same formula as piston_position.py)."""
    scope_m = scope_ft * FT_TO_M
    barrel_m = core_length_ft * FT_TO_M
    piston = np.empty_like(ws)
    piston[:start_core_idx] = ws[:start_core_idx] + offset + barrel_m
    piston[start_core_idx:] = rel[start_core_idx:] + scope_m + barrel_m + offset
    return piston


def detect_start_core(ws, rel, scope_ft, trip_idx):
    """First index >= trip_idx where |release - ws| > scope."""
    scope_m = scope_ft * FT_TO_M
    sep = np.abs(rel[trip_idx:] - ws[trip_idx:])
    candidates = np.where(sep > scope_m)[0]
    if len(candidates) == 0:
        return len(ws) - 1
    return trip_idx + int(candidates[0])


# ── Main ───────────────────────────────────────────────────────────────

def main():
    default = "SKQ202512S-14JC_corrected_20260224_065002.csv"
    path = sys.argv[1] if len(sys.argv) > 1 else default

    print(f"Loading {path}...")
    df = load_csv(path)
    ws_col, rel_col, trig_col = find_columns(df)
    print(f"  WS:  {ws_col}")
    print(f"  Rel: {rel_col}")
    print(f"  Trig: {trig_col}")

    ws = df[ws_col].interpolate().ffill().bfill().values
    rel = df[rel_col].interpolate().ffill().bfill().values
    trig = df[trig_col].interpolate().ffill().bfill().values if trig_col else None

    trip_idx = find_trip_index(path)
    if trip_idx is None:
        print("ERROR: no trip index found in header")
        return
    print(f"  Trip index: {trip_idx}")

    # Detect start_core
    sc_idx = detect_start_core(ws, rel, SCOPE_FT, trip_idx)
    print(f"  Start core index: {sc_idx}")

    # Compute piston
    piston = compute_piston(ws, rel, SCOPE_FT, CORE_LENGTH_FT, sc_idx)

    # Derived constants
    core_length_m = CORE_LENGTH_FT * FT_TO_M
    tc_length_m = TRIGGER_CORE_LENGTH_FT * FT_TO_M

    # Key depth values
    ws_at_trip = float(ws[trip_idx])
    rel_at_trip = float(rel[trip_idx])

    # Seafloor
    seafloor = float(trig[trip_idx]) + tc_length_m - TRIGGER_PEN_M
    print(f"  Seafloor: {seafloor:.3f} m")

    # Core tip depth at trip = ws_at_trip + weight_length + core_length
    core_tip_at_trip = ws_at_trip + WEIGHT_LENGTH + core_length_m
    freefall_est = abs(core_tip_at_trip - seafloor)
    print(f"  WS @ trip: {ws_at_trip:.3f} m")
    print(f"  Core tip @ trip: {core_tip_at_trip:.3f} m")
    print(f"  Freefall estimate: {freefall_est:.3f} m")

    # ── Find start_penetration ──
    # start_penetration = timestamp where core tip reaches seafloor
    # core_tip(t) = ws(t) + weight_length + core_length_m
    core_tip_series = ws + WEIGHT_LENGTH + core_length_m
    # First index >= trip where core_tip >= seafloor
    after_trip = core_tip_series[trip_idx:]
    pen_candidates = np.where(after_trip >= seafloor)[0]
    if len(pen_candidates) == 0:
        print("WARNING: core tip never reaches seafloor")
        start_pen_idx = None
    else:
        start_pen_idx = trip_idx + int(pen_candidates[0])

    if start_pen_idx is not None:
        ws_at_sp = float(ws[start_pen_idx])
        piston_at_sp = float(piston[start_pen_idx])
        piston_alt_sp = seafloor - piston_at_sp
        dt_sp = df['datetime'].iloc[start_pen_idx]
        print(f"  Start penetration index: {start_pen_idx}")
        print(f"  Start penetration time: {dt_sp}")
        print(f"  WS @ start_pen: {ws_at_sp:.3f} m")
        print(f"  Piston @ start_pen: {piston_at_sp:.3f} m")
        print(f"  Piston altitude @ start_pen: {piston_alt_sp:.3f} m")
    else:
        ws_at_sp = None

    # ── Plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))

    bar_w = 0.8
    gap = 3.0       # x gap between groups

    # Group 1 (left): @ trip
    x_trip = 0.0
    # Group 2 (right): @ start_penetration
    x_pen = x_trip + gap

    def draw_box(ax, x, top, height, colour, label, alpha=0.5):
        """Draw a vertical box (bar) with end-caps and centre label."""
        bot = top + height
        ax.bar(x, height, bottom=top, width=bar_w,
               color=colour, alpha=alpha, edgecolor=colour, linewidth=1.5)
        # End-caps
        cap = bar_w * 0.65
        for y in (top, bot):
            ax.plot([x - cap / 2, x + cap / 2], [y, y],
                    color=colour, linewidth=2)
        # Label
        mid = top + height / 2
        ax.text(x, mid, f"{label}\n{height:.3f} m",
                ha='center', va='center', fontsize=7.5, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.25', fc=colour, alpha=0.85))

    # ── GROUP 1: @ trip ────────────────────────────────────────────────
    draw_box(ax, x_trip, ws_at_trip, WEIGHT_LENGTH, "#3498db", "Weight")
    draw_box(ax, x_trip, ws_at_trip + WEIGHT_LENGTH, core_length_m,
             "#2ecc71", "Core Barrel")
    draw_box(ax, x_trip, ws_at_trip + WEIGHT_LENGTH + core_length_m,
             freefall_est, "#e74c3c", "Freefall\nEstimate")

    ax.text(x_trip, ws_at_trip - 1.5,
            "@ Trip", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#2c3e50')

    # ── GROUP 2: @ start_penetration ───────────────────────────────────
    if ws_at_sp is not None:
        draw_box(ax, x_pen, ws_at_sp, WEIGHT_LENGTH, "#3498db", "Weight")
        draw_box(ax, x_pen, ws_at_sp + WEIGHT_LENGTH, core_length_m,
                 "#2ecc71", "Core Barrel")
        draw_box(ax, x_pen, piston_at_sp, piston_alt_sp,
                 "#9b59b6", "Piston\nAltitude")

        ax.text(x_pen, ws_at_sp - 1.5,
                "@ Start Penetration", ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#2c3e50')

    # ── Reference lines ───────────────────────────────────────────────
    x_lo = x_trip - 1.5
    x_hi = (x_pen + 1.5) if ws_at_sp else (x_trip + 1.5)

    ax.axhline(seafloor, color='#c0392b', linestyle='--', linewidth=1.5,
               alpha=0.8)
    ax.text(x_hi + 0.3, seafloor, f"Seafloor\n{seafloor:.1f} m",
            va='center', fontsize=8, color='#c0392b', fontstyle='italic')

    ax.axhline(ws_at_trip, color='#7f8c8d', linestyle=':', linewidth=1,
               alpha=0.6)
    ax.text(x_hi + 0.3, ws_at_trip, f"WS @ trip\n{ws_at_trip:.1f} m",
            va='center', fontsize=7.5, color='#7f8c8d', fontstyle='italic')

    if ws_at_sp is not None:
        ax.axhline(ws_at_sp, color='#8e44ad', linestyle=':', linewidth=1,
                   alpha=0.6)
        ax.text(x_hi + 0.3, ws_at_sp,
                f"WS @ start pen\n{ws_at_sp:.1f} m",
                va='center', fontsize=7.5, color='#8e44ad',
                fontstyle='italic')

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_xlim(x_lo, x_hi + 4)
    ax.invert_yaxis()
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.set_title('Core-Barrel Geometry: Trip vs Start of Penetration',
                 fontsize=13, fontweight='bold')
    ax.set_xticks([])
    ax.grid(axis='y', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
