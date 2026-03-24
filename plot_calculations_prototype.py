"""
Prototype visualisation of coring-analysis calculated fields.

Parses an exported calculation report (.txt) and produces a y-axis
(depth) diagram showing each metric as a depth range (vertical bar
with end-caps and a label).

Run standalone:
  python plot_calculations_prototype.py
  python plot_calculations_prototype.py path/to/report.txt
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Parse the export report ────────────────────────────────────────────

def parse_report(path: str) -> dict:
    """Extract calculated values, inputs, and source info from a report."""
    text = Path(path).read_text(encoding='utf-8')
    data = {}

    def _grab(label: str) -> float | None:
        m = re.search(rf"{re.escape(label)}\s+([\d.+-]+)\s*m", text)
        return float(m.group(1)) if m else None

    data['recoil_max']      = _grab("Recoil Max:")
    data['fall_dist']       = _grab("Fall Distance:")
    data['suck_in']         = _grab("Suck-in:")
    data['recoil_start']    = _grab("Recoil at Start Core:")
    data['freefall_start']  = _grab("Freefall at Start Core:")
    data['piston_suck']     = _grab("Piston Suck:")
    data['seafloor']        = _grab("Seafloor Depth:")
    data['piston_alt']      = _grab("Piston Altitude:")
    data['pen_deficit']     = _grab("Penetration Deficit:")
    data['freefall_est']    = _grab("Freefall Estimate:")

    # Metadata for title
    m = re.search(r"Core:\s+(.+)", text)
    data['core'] = m.group(1).strip() if m else "Unknown"

    # Trigger core length (metres)
    m = re.search(r"Trigger core length:\s+[\d.]+\s*ft\s+\(([\d.]+)\s*m\)", text)
    data['tc_length_m'] = float(m.group(1)) if m else None

    # Core length (metres)
    m = re.search(r"Core length:\s+[\d.]+\s*ft\s+\(([\d.]+)\s*m\)", text)
    data['core_length_m'] = float(m.group(1)) if m else None

    # Trigger penetration
    m = re.search(r"Trigger penetration:\s+([\d.]+)\s*m", text)
    data['trigger_pen'] = float(m.group(1)) if m else 0.0

    return data


# ── Reconstruct anchor depths from the calculated distances ───────────

def build_ranges(d: dict):
    """
    We don't have the raw sensor depths in the report, but seafloor is
    an absolute depth.  We can back-derive the key anchor points:

        seafloor            = known
        ws_max              = seafloor - pen_deficit
        ws_at_trip          = ws_max - fall_dist - suck_in
                              (because fall_dist = |ws[trip] - ws[trip+5s]|
                               and suck_in = |ws[trip+5s] - ws_max|)
        ws_at_5s            = ws_at_trip + fall_dist
        ws_at_sc            = ws_at_trip + freefall_start
        rel_at_trip         = ws_at_trip  (approximation; offset is small)
        rel_at_sc           = rel_at_trip + recoil_start
        piston_at_sc        = seafloor - piston_alt
    """
    sf = d['seafloor']
    if sf is None:
        # Can't anchor without seafloor – place bars relative to zero
        sf = 0.0

    pen_deficit   = d['pen_deficit']   or 0.0
    fall_dist     = d['fall_dist']     or 0.0
    suck_in       = d['suck_in']       or 0.0
    freefall_st   = d['freefall_start'] or 0.0
    recoil_max    = d['recoil_max']    or 0.0
    recoil_start  = d['recoil_start']  or 0.0
    piston_suck   = d['piston_suck']   or 0.0
    piston_alt    = d['piston_alt']    or 0.0
    freefall_est  = d['freefall_est']  or 0.0

    ws_max     = sf - pen_deficit
    ws_at_5s   = ws_max - suck_in
    ws_at_trip = ws_at_5s - fall_dist
    ws_at_sc   = ws_at_trip + freefall_st
    rel_at_trip = ws_at_trip          # close enough for visualisation
    rel_at_sc   = rel_at_trip + recoil_start
    piston_at_sc = sf - piston_alt

    core_length_m = d['core_length_m'] or 0.0

    ranges = []

    def _add(label, top, val, colour):
        if val and val > 0:
            ranges.append((label, top, top + val, colour, val))

    _add("Recoil\nMax",            rel_at_trip,  recoil_max,   "#e74c3c")
    _add("Fall\nDist",             ws_at_trip,   fall_dist,    "#3498db")
    _add("Suck-in",                ws_at_5s,     suck_in,      "#2ecc71")
    _add("Recoil @\nStart Core",   rel_at_trip,  recoil_start, "#e67e22")
    _add("Freefall @\nStart Core", ws_at_trip,   freefall_st,  "#9b59b6")
    _add("Piston\nSuck",           piston_at_sc - piston_suck,
         piston_suck, "#1abc9c")
    _add("Pen\nDeficit",           ws_max,       pen_deficit,  "#f39c12")
    _add("Piston\nAltitude",       piston_at_sc, piston_alt,   "#34495e")
    _add("Freefall\nEstimate",     ws_at_trip + 1.5,
         freefall_est, "#8e44ad")

    ref_lines = [
        ("Seafloor",   sf,           "#c0392b", "--"),
        ("WS @ trip",  ws_at_trip,   "#7f8c8d", ":"),
        ("WS max",     ws_max,       "#27ae60", ":"),
    ]

    return ranges, ref_lines


# ── Plot ───────────────────────────────────────────────────────────────

def plot_fields(ranges, ref_lines, title: str):
    fig, ax = plt.subplots(figsize=(10, 8))

    bar_width = 0.5
    x_positions = np.arange(len(ranges)) * 1.2

    for i, (label, top, bot, colour, val) in enumerate(ranges):
        x = x_positions[i]
        height = bot - top

        # Vertical bar
        ax.bar(x, height, bottom=top, width=bar_width,
               color=colour, alpha=0.55, edgecolor=colour, linewidth=1.2)

        # End-cap ticks
        cap_hw = bar_width * 0.6
        for y in (top, bot):
            ax.plot([x - cap_hw / 2, x + cap_hw / 2], [y, y],
                    color=colour, linewidth=2)

        # Value annotation (centred on bar)
        mid = (top + bot) / 2
        ax.text(x, mid, f"{val:.3f} m",
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc=colour, alpha=0.85))

    # Reference lines
    x_lo = x_positions[0] - 1
    x_hi = x_positions[-1] + 1
    for label, depth, colour, ls in ref_lines:
        ax.axhline(depth, color=colour, linestyle=ls, linewidth=1, alpha=0.7)
        ax.text(x_hi + 0.3, depth, f"{label}\n{depth:.1f} m",
                va='center', fontsize=7.5, color=colour, fontstyle='italic')

    # Axes formatting
    ax.set_xlim(x_lo, x_hi + 4)
    ax.invert_yaxis()
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([r[0] for r in ranges], fontsize=7.5)
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    default = "SKQ202512S-14JC_calculations_20260319_103733.txt"
    path = sys.argv[1] if len(sys.argv) > 1 else default
    data = parse_report(path)

    ranges, ref_lines = build_ranges(data)
    title = f"Coring Analysis — {data['core']}"
    plot_fields(ranges, ref_lines, title)
