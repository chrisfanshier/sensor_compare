"""Quick debug: verify piston values at start_penetration."""
import numpy as np
import pandas as pd
from pathlib import Path

FT_TO_M = 1.0 / 3.28
CORE_LENGTH_FT = 50.0
TRIGGER_CORE_LENGTH_FT = 10.0
TRIGGER_PEN_M = 1.5
SCOPE_FT = 22.5
OFFSET_CONSTANT = 1.25
WEIGHT_LENGTH = 1.5

path = "SKQ202512S-14JC_corrected_20260224_065002.csv"
lines = Path(path).read_text(encoding="utf-8").splitlines()
hdr = next(i for i, l in enumerate(lines) if l.strip().startswith("datetime"))
df = pd.read_csv(path, skiprows=hdr, parse_dates=["datetime"])

ws_col = [c for c in df.columns if "Weight Stand" in c][0]
rel_col = [c for c in df.columns if "Release" in c][0]
trig_col = [c for c in df.columns if "Trigger" in c][0]

ws = df[ws_col].interpolate().ffill().bfill().values
rel = df[rel_col].interpolate().ffill().bfill().values
trig = df[trig_col].interpolate().ffill().bfill().values

trip_idx = 311871
scope_m = SCOPE_FT * FT_TO_M
barrel_m = CORE_LENGTH_FT * FT_TO_M
tc_m = TRIGGER_CORE_LENGTH_FT * FT_TO_M

# Start core detection
sep = np.abs(rel[trip_idx:] - ws[trip_idx:])
cands = np.where(sep > scope_m)[0]
sc_idx = trip_idx + int(cands[0]) if len(cands) else len(ws) - 1

# Piston position
piston = np.empty_like(ws)
piston[:sc_idx] = ws[:sc_idx] + OFFSET_CONSTANT + barrel_m
piston[sc_idx:] = rel[sc_idx:] + scope_m + barrel_m + OFFSET_CONSTANT

# Seafloor
sf = float(trig[trip_idx]) + tc_m - TRIGGER_PEN_M

# Core tip series
core_tip = ws + WEIGHT_LENGTH + barrel_m

# Start penetration (last index before core_tip crosses seafloor)
after = core_tip[trip_idx:]
pc = np.where(after >= sf)[0]
first_cross = trip_idx + int(pc[0])
sp_idx = max(first_cross - 1, trip_idx)

print(f"Scope: {SCOPE_FT} ft = {scope_m:.3f} m")
print(f"Barrel: {CORE_LENGTH_FT} ft = {barrel_m:.3f} m")
print(f"Start core idx: {sc_idx} (trip+{sc_idx - trip_idx})")
print(f"Seafloor: {sf:.3f} m")
print()
print(f"WS @ trip: {ws[trip_idx]:.3f}")
print(f"Rel @ trip: {rel[trip_idx]:.3f}")
print(f"Piston @ trip: {piston[trip_idx]:.3f}")
print()
print(f"Start pen idx: {sp_idx} (trip+{sp_idx - trip_idx})")
dt_col = "datetime"
print(f"Time @ start_pen: {df[dt_col].iloc[sp_idx]}")
print(f"WS @ start_pen: {ws[sp_idx]:.3f}")
print(f"Rel @ start_pen: {rel[sp_idx]:.3f}")
print(f"Piston @ start_pen: {piston[sp_idx]:.3f}")
print(f"Piston alt @ start_pen: {sf - piston[sp_idx]:.3f}")
print(f"Core tip @ start_pen: {core_tip[sp_idx]:.3f}")
print(f"Core tip - seafloor: {core_tip[sp_idx] - sf:.3f}")
print()
# Also show a few samples around start_pen
print("Samples around start_penetration:")
for off in [-2, -1, 0, 1, 2]:
    idx = sp_idx + off
    print(f"  idx {idx} (trip+{idx-trip_idx}): "
          f"ws={ws[idx]:.3f}, rel={rel[idx]:.3f}, "
          f"piston={piston[idx]:.3f}, core_tip={core_tip[idx]:.3f}")
