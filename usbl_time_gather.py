import pandas as pd                      # tabular data handling
from pathlib import Path                 # filesystem-safe paths

data_dir = Path("/home/server/project/marssam/marssam_data_files/usbl_data")   # directory containing USBL files

rows = []                                # one summary record per file
skipped = []                              # records of skipped files

for f in sorted(data_dir.glob("PSONLLD_*.txt")):   # loop over all USBL files

    date_str = f.stem.split("_")[1]      # extract YYYYMMDD from filename

    df = pd.read_csv(
        f,
        header=None,
        names=[
            "ID",
            "utc_time",
            "beacon_name",
            "fix_status",
            "latitude_deg",
            "longitude_deg",
            "depth_m",
            "northing_error_m",
            "easting_error_m",
            "vertical_error_m",
            "quality",
            "reserved_1",
            "reserved_2",
            "reserved_3",
            "checksum",
        ],
        dtype={"utc_time": str},  # preserve leading zeros in UTC field
    )

    # prepare raw tokens
    utc_raw = df["utc_time"].astype(str).str.strip()

    # quick vectorized attempt to find failures
    combined_all = date_str + " " + utc_raw
    dt_try = pd.to_datetime(
        combined_all,
        format="%Y%m%d %H%M%S.%f",
        utc=True,
        errors="coerce",
    )

    total_rows = len(dt_try)
    valid_count = dt_try.notna().sum()

    # If the file has no valid timestamps, skip it and record why
    if valid_count == 0:
        print(f"[{f.name}] SKIPPING: no valid timestamps ({total_rows} rows)")
        skipped.append({"file_name": f.name, "reason": "no_valid_timestamps", "rows": total_rows})
        continue

    # If some rows failed, print a short diagnostic and proceed using the valid rows
    if valid_count < total_rows:
        print(f"[{f.name}] Partial parse: {valid_count}/{total_rows} valid timestamps (NaT for others).")

    # proceed using the vectorized result (NaT where parsing failed)
    datetime_utc = dt_try

    rows.append({
        "file_name": f.name,             # original filename
        "start_datetime": datetime_utc.min(),  # earliest timestamp in file (ignores NaT)
        "end_datetime": datetime_utc.max(),    # latest timestamp in file (ignores NaT)
        "valid_timestamps": int(valid_count),
        "total_rows": int(total_rows),
    })

summary_df = pd.DataFrame(rows)           # build summary table

summary_df.to_csv(
    "psonnlld_file_times.csv",            # output CSV file
    index=False                           # no row index in output
)

# record skipped files for later inspection
if skipped:
    skipped_df = pd.DataFrame(skipped)
    skipped_df.to_csv("psonnlld_skipped_files.csv", index=False)
    print(f"Skipped {len(skipped)} files; details written to psonnlld_skipped_files.csv")
else:
    print("No files skipped.")