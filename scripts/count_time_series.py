import polars as pl
import glob
import numpy as np

files = sorted(glob.glob('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/top_secret/MC_v2/22042/0000000-0000999/*.parquet'))
# remove files from list with 'meta' in name
files = [file for file in files if 'meta' not in file]

# create a dictionary with filenames as keys and lengths as values
total_length_dict = {
    file: pl.scan_parquet(file)
    .group_by(["event_id", "sensor_id"])
    .agg([
        pl.len(),
        pl.col("time"),
        pl.col("charge"),
        pl.col("auxiliary"),
    ])
    .collect()
    .height
    for file in files
}

# Save the dictionary using numpy
np.save('./data/22042_0_999_time_series_lengths.npy', total_length_dict)