"""
Debug piston_suck calculation for 29JC.

Loads the raw export file, computes piston position, finds trip index,
then shows exactly what timestamps the current algorithm flags for
Point A (ws_max_idx) and Point B (release crossing), as well as what
the proposed new method would flag.
"""
import numpy as np
import pandas as pd
from pathlib import Path

FT_TO_M = 1.0 / 3.28

FILE = (
    r"C:\Users\chris\Documents\sediment_app\combined_by_sensor"
    r"\SKQ202512S_sensor_export"
    r"\sensor_export_SKQ202512S_SKQ202512S-29JC_20260219_081453.csv"
)

# ── Load file ──────────────────────────────────────────────────────────
lines = Path(FILE).read_text(encoding="utf-8").splitlines()
meta = {}
for line in lines:
    if not line.startswith("#"):
        break
    for key in ("Scope", "Core length", "Trigger core length"):
        if line.startswith(f"# {key}:"):
            meta[key] = float(line.split(":")[1].strip())

hdr = next(i for i, l in enumerate(lines) if l.strip().startswith("datetime"))
df = pd.read_csv(FILE, skiprows=hdr, parse_dates=["datetime"])
print(f"Rows: {len(df):,}  |  Cols: {len(df.columns)}")
print(f"Metadata: {meta}")

scope_ft           = meta.get("Scope", 20.0)
core_length_ft     = meta.get("Core length", 70.0)
trig_length_ft     = meta.get("Trigger core length", 10.0)
OFFSET_CONSTANT    = 1.25
TRIGGER_PEN        = 1.5   # m

scope_m   = scope_ft   * FT_TO_M
barrel_m  = core_length_ft * FT_TO_M
tc_m      = trig_length_ft * FT_TO_M

ws_col   = [c for c in df.columns if "Weight Stand" in c and "Depth" in c][0]
rel_col  = [c for c in df.columns if "Release" in c and "Depth" in c][0]
trig_col = [c for c in df.columns if "Trigger" in c and "Depth" in c][0]

print(f"\nWS col  : {ws_col}")
print(f"REL col : {rel_col}")
print(f"TRIG col: {trig_col}")

ws  = df[ws_col].interpolate().ffill().bfill().values
rel = df[rel_col].interpolate().ffill().bfill().values
trig = df[trig_col].interpolate().ffill().bfill().values
ts  = df["datetime"].values  # numpy datetime64

# ── Find trip index automatically ─────────────────────────────────────
# Trip = point of maximum WS descent rate (largest single-step increase)
ws_diff = np.diff(ws)
# Look in the middle third of the record where the core is expected
n = len(ws)
search_start = n // 4
search_end   = 3 * n // 4
trip_idx = search_start + int(np.nanargmax(ws_diff[search_start:search_end]))
print(f"\nAuto-detected trip_idx : {trip_idx}")
print(f"  WS  @ trip           : {ws[trip_idx]:.3f} m")
print(f"  REL @ trip           : {rel[trip_idx]:.3f} m")
print(f"  Time @ trip          : {ts[trip_idx]}")

# ── Seafloor ───────────────────────────────────────────────────────────
sf = float(trig[trip_idx]) + tc_m - TRIGGER_PEN
print(f"\nSeafloor               : {sf:.3f} m")

# ── Start-core detection ───────────────────────────────────────────────
sep = np.abs(rel[trip_idx:] - ws[trip_idx:])
cands = np.where(sep > scope_m)[0]
sc_idx = trip_idx + int(cands[0]) if len(cands) else len(ws) - 1
print(f"\nStart-core idx         : {sc_idx}")
print(f"  Time @ start-core    : {ts[sc_idx]}")
print(f"  WS  @ start-core     : {ws[sc_idx]:.3f} m")
print(f"  REL @ start-core     : {rel[sc_idx]:.3f} m")

# ── Piston position ────────────────────────────────────────────────────
piston = np.empty_like(ws)
piston[:sc_idx] = ws[:sc_idx] + OFFSET_CONSTANT + barrel_m
piston[sc_idx:] = rel[sc_idx:] + scope_m + barrel_m + OFFSET_CONSTANT

# ── CURRENT algorithm for piston_suck ─────────────────────────────────
print("\n" + "="*60)
print("CURRENT ALGORITHM")
print("="*60)

ws_max_idx = int(np.nanargmax(ws))
print(f"Point A — ws_max_idx   : {ws_max_idx}")
print(f"  Time                 : {ts[ws_max_idx]}")
print(f"  WS depth             : {ws[ws_max_idx]:.3f} m")
print(f"  Piston depth         : {piston[ws_max_idx]:.3f} m")

release_at_trip = rel[trip_idx]
# 5 s after trip, capped at 10 min — using actual timestamps
ts_epoch = df["datetime"].astype("int64") / 1e9
search_start = int(np.searchsorted(ts_epoch.values, ts_epoch.iloc[trip_idx] + 5.0))
search_end   = int(np.searchsorted(ts_epoch.values, ts_epoch.iloc[trip_idx] + 600.0))
search_end   = min(search_end, len(rel) - 1)
diffs = np.abs(rel[search_start:search_end] - release_at_trip)
crossing_candidates = np.where(diffs <= np.nanmin(diffs) * 1.0 + 0.05)[0]
if len(crossing_candidates) > 0:
    cross_idx = search_start + int(crossing_candidates[0])
    piston_suck_current = abs(float(piston[ws_max_idx] - piston[cross_idx]))
    print(f"\nPoint B — cross_idx    : {cross_idx}")
    print(f"  Time                 : {ts[cross_idx]}")
    print(f"  REL depth            : {rel[cross_idx]:.3f} m  (ref={release_at_trip:.3f})")
    print(f"  Piston depth         : {piston[cross_idx]:.3f} m")
    print(f"\npiston_suck (current)  : {piston_suck_current:.3f} m")
else:
    print("Point B: NOT FOUND (no crossing)")
    cross_idx = None

# ── PROPOSED algorithm ─────────────────────────────────────────────────
# Point A: first point AFTER trip where release returns to release[trip_idx]
#          (wire taut again → release is meaningful piston indicator)
#          Search only in a window after trip (e.g. first 30 min)
# Point B: ws_max_idx (WS starts moving upward)
print("\n" + "="*60)
print("PROPOSED ALGORITHM")
print("="*60)

# Define search window: trip_idx+1 .. trip_idx + 30 min worth of samples
sample_rate_hz = 32  # RBR sensors at 32 Hz
window_30min = int(30 * 60 * sample_rate_hz)
search_end_new = min(trip_idx + window_30min, len(rel))

# Point A: first crossing from BELOW back up to release[trip_idx]
# After trip the release goes slack/deeper, then comes back taut.
# We want: first idx after trip where rel[i] <= release_at_trip
# (ascending in depth = going shallower once taut)
ref_depth = release_at_trip
search_slice = rel[trip_idx+1 : search_end_new]
# Find where release is shallower than (or equal to) trip depth
taut_candidates = np.where(search_slice <= ref_depth)[0]
if len(taut_candidates) > 0:
    new_ptA_idx = trip_idx + 1 + int(taut_candidates[0])
else:
    # Fallback: closest approach
    new_ptA_idx = trip_idx + 1 + int(np.argmin(np.abs(search_slice - ref_depth)))

print(f"Point A — wire-taut    : {new_ptA_idx}  (ref depth={ref_depth:.3f} m)")
print(f"  Time                 : {ts[new_ptA_idx]}")
print(f"  REL depth            : {rel[new_ptA_idx]:.3f} m")
print(f"  Piston depth         : {piston[new_ptA_idx]:.3f} m")

# Point B: ws_max_idx (WS at maximum depth before ascending)
new_ptB_idx = ws_max_idx
print(f"\nPoint B — WS max depth : {new_ptB_idx}")
print(f"  Time                 : {ts[new_ptB_idx]}")
print(f"  WS depth             : {ws[new_ptB_idx]:.3f} m")
print(f"  Piston depth         : {piston[new_ptB_idx]:.3f} m")

piston_suck_new = abs(float(piston[new_ptA_idx] - piston[new_ptB_idx]))
print(f"\npiston_suck (proposed) : {piston_suck_new:.3f} m")

# ── Print table of key values around trip for context ─────────────────
print("\n" + "="*60)
print(f"Data sample around trip (idx {trip_idx-2} .. {trip_idx+10})")
print("="*60)
print(f"{'idx':>8}  {'time':>26}  {'WS':>8}  {'REL':>8}  {'PISTON':>8}")
for i in range(max(0, trip_idx-2), min(len(ws), trip_idx+11)):
    print(f"{i:>8}  {str(ts[i]):>26}  {ws[i]:>8.3f}  {rel[i]:>8.3f}  {piston[i]:>8.3f}")

print("\n" + "="*60)
print(f"Data sample around ws_max_idx ({ws_max_idx})")
print("="*60)
print(f"{'idx':>8}  {'time':>26}  {'WS':>8}  {'REL':>8}  {'PISTON':>8}")
for i in range(max(0, ws_max_idx-3), min(len(ws), ws_max_idx+8)):
    print(f"{i:>8}  {str(ts[i]):>26}  {ws[i]:>8.3f}  {rel[i]:>8.3f}  {piston[i]:>8.3f}")

print("\n" + "="*60)
print(f"Data sample around new Point A (wire-taut idx {new_ptA_idx})")
print("="*60)
print(f"{'idx':>8}  {'time':>26}  {'WS':>8}  {'REL':>8}  {'PISTON':>8}")
for i in range(max(0, new_ptA_idx-3), min(len(ws), new_ptA_idx+8)):
    print(f"{i:>8}  {str(ts[i]):>26}  {ws[i]:>8.3f}  {rel[i]:>8.3f}  {piston[i]:>8.3f}")
