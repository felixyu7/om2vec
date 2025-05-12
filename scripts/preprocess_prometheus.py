#!/usr/bin/env python3
# preprocess_prometheus.py – folder-aware, optional chunking by row count

import argparse
import pathlib
import sys
from typing import List, Dict, Deque
from collections import deque # Import deque

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ───────────────── target schema ─────────────────
TARGET_SCHEMA = pa.schema( # Renamed for clarity, TARGET is a bit generic
    [
        ("event_id", pa.uint32()),
        ("string_id", pa.int64()),
        ("sensor_id", pa.int64()),
        ("sensor_pos_x", pa.float64()),
        ("sensor_pos_y", pa.float64()),
        ("sensor_pos_z", pa.float64()),
        ("nhits", pa.int32()),
        ("hits_t", pa.list_(pa.float64())),
        ("hits_id_idx", pa.list_(pa.int64())),
    ]
)

# ───────────────── explode a single event ─────────────────
def explode_event(p: Dict, eid: int) -> List[Dict]:
    # Assuming p's values are already list-like (e.g., from to_pylist() or as_py())
    s_id_arr, p_id_arr = map(np.asarray, (p["string_id"], p["sensor_id"]))
    pos_x_arr, pos_y_arr, pos_z_arr = map(
        np.asarray,
        (p["sensor_pos_x"], p["sensor_pos_y"], p["sensor_pos_z"]),
    )
    t_arr = np.asarray(p["t"])
    idx_arr = np.asarray(p["id_idx"])

    # np.stack creates a new array, ensure types are compatible for stacking if not already numeric
    # If s_id_arr and p_id_arr can have different dtypes that are not auto-promoted,
    # consider converting them to a common type or handling it carefully.
    # For int64, this should be fine.
    stacked_ids = np.stack([s_id_arr, p_id_arr], axis=1)
    
    # Using return_inverse=True is crucial for the logic below
    unique_sensor_pairs, inverse_indices = np.unique(
        stacked_ids, axis=0, return_inverse=True
    )
    
    rows: List[Dict] = []
    for k, (sid, pid) in enumerate(unique_sensor_pairs):
        mask = inverse_indices == k
        # Get the first valid index for position data based on the mask
        # This assumes that all entries for a given sensor pair have the same position.
        first_occurrence_idx = np.where(mask)[0][0]

        rows.append(
            dict(
                event_id=eid,
                string_id=int(sid), # Ensure conversion to Python int
                sensor_id=int(pid), # Ensure conversion to Python int
                sensor_pos_x=float(pos_x_arr[first_occurrence_idx]),
                sensor_pos_y=float(pos_y_arr[first_occurrence_idx]),
                sensor_pos_z=float(pos_z_arr[first_occurrence_idx]),
                nhits=int(mask.sum()), # Number of hits for this sensor
                hits_t=t_arr[mask].tolist(),
                hits_id_idx=idx_arr[mask].tolist(),
            )
        )
    return rows

# ───────────────── flush helper for chunked mode ─────────────────
def flush_rows(
    buf: Deque[Dict], out_dir: pathlib.Path, part_idx: int, rows_per_file: int
) -> int:
    """Write full chunks of size rows_per_file; return updated part_idx."""
    while len(buf) >= rows_per_file:
        chunk_list: List[Dict] = []
        for _ in range(rows_per_file):
            chunk_list.append(buf.popleft()) # Efficiently remove from left
        
        out_path = out_dir / f"part_{part_idx:05d}.parquet"
        # Ensure TARGET_SCHEMA is used
        table_to_write = pa.Table.from_pylist(chunk_list, schema=TARGET_SCHEMA)
        pq.write_table(table_to_write, out_path, compression="zstd")
        print(f"✓ wrote {len(chunk_list):,} rows → {out_path.name}")
        part_idx += 1
    return part_idx

# ───────────────── process ONE parquet (adds to shared buffer) ─────────────────
def process_one_file(src: pathlib.Path, buf: Deque[Dict]) -> int: # Changed buf type
    """Processes a single Parquet file, adds exploded events to buffer, returns total event count."""
    pf = pq.ParquetFile(src)
    # Assuming 'event_id' should be unique across all events from all files if processed sequentially
    # The original script reset eid = 0 for each file.
    # If eid needs to be globally unique and sequential across files in a directory run,
    # it should be passed in and returned, or handled by the caller.
    # For now, maintaining original behavior: eid is local to file processing.
    event_id_counter = 0
    
    # Specify the column we are interested in
    # This makes the code more robust than relying on column index 0.
    columns_to_read = ["photons"]

    for rg_idx in range(pf.num_row_groups):
        # Read only the 'photons' column from the row group
        row_group_table = pf.read_row_group(rg_idx, columns=columns_to_read)
        
        # Get the 'photons' column as an Arrow Array
        photons_struct_array = row_group_table.column("photons")
        
        # Convert the whole StructArray to a list of Python dicts efficiently
        # Each dict in this list represents an event.
        events_data_list = photons_struct_array.to_pylist()
        
        for event_dict in events_data_list:
            if event_dict is not None: # StructArray can have nulls
                 # Ensure `event_dict` contains all keys expected by `explode_event`
                required_keys = {"string_id", "sensor_id", "sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", "id_idx"}
                if not required_keys.issubset(event_dict.keys()):
                    print(f"Warning: Event data missing required keys at event_id {event_id_counter} in {src.name}. Skipping event. Data: {event_dict}", file=sys.stderr)
                    # Optionally, handle this case more gracefully, e.g., by filling missing keys with defaults or raising an error
                else:
                    buf.extend(explode_event(event_dict, event_id_counter))
            event_id_counter += 1
    return event_id_counter # Return the number of events processed in this file


# ───────────────── CLI ─────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert event-level parquet(s) to sensor-level parquet(s)."
    )
    ap.add_argument(
        "input",
        type=pathlib.Path,
        help="single parquet OR directory containing parquets",
    )
    ap.add_argument(
        "output",
        type=pathlib.Path,
        help="output file (normal mode) OR directory (chunked mode)",
    )
    ap.add_argument(
        "--rows-per-file",
        type=int,
        default=0,
        metavar="N",
        help="chunk output into files with N rows each; "
        "if 0 (default) each input file → one output file (if input is dir) "
        "or single output file (if input is file)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: input path {args.input} not found.")

    chunked_output = args.rows_per_file > 0

    if chunked_output:
        if not args.output.is_dir(): # If it exists but not a dir, error
            if args.output.exists():
                sys.exit(f"ERROR: Chunked output destination {args.output} exists but is not a directory.")
            args.output.mkdir(parents=True, exist_ok=True)
        elif not args.output.exists(): # Should be caught by above, but defensive
             args.output.mkdir(parents=True, exist_ok=True)


    if args.input.is_file():
        if chunked_output:
            # This behavior can be debated. One might want to chunk a single large input file.
            # Sticking to original script's restriction.
            sys.exit(
                "ERROR: when --rows-per-file is set, input should be a folder."
                " (Original script restriction)"
            )
        
        # Simple one-file conversion (no chunking)
        # Use deque for consistency, though for single file it might not matter much
        rows_buffer: Deque[Dict] = deque()
        process_one_file(args.input, rows_buffer)
        
        if not rows_buffer:
            print(f"No data processed from {args.input.name}. Output file {args.output.name} will not be created.", file=sys.stderr)
            return

        # Ensure output directory exists if output is a path like dir/file.parquet
        if args.output.parent and not args.output.parent.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(
            pa.Table.from_pylist(list(rows_buffer), schema=TARGET_SCHEMA), # Convert deque to list for from_pylist
            args.output,
            compression="zstd",
        )
        print(f"✓ {args.input.name} ({len(rows_buffer):,} rows) → {args.output.name}")

    elif args.input.is_dir():
        files = sorted(args.input.glob("*.parquet"))
        if not files:
            sys.exit(f"ERROR: input directory {args.input} contains no .parquet files.")

        if chunked_output:
            if not args.output.is_dir(): # Should have been created or exited already
                 sys.exit(f"ERROR: Output directory {args.output} for chunked files not found or not a directory.")

            buffer_for_chunks: Deque[Dict] = deque()
            current_part_idx = 0
            total_rows_processed = 0
            for f_path in files:
                print(f"Processing {f_path.name}...")
                # process_one_file now appends to buffer_for_chunks
                # eid is managed within process_one_file per file basis as per original.
                # If global eid is needed, it must be threaded through.
                num_events_in_file = process_one_file(f_path, buffer_for_chunks)
                print(f"  {f_path.name} contributed {num_events_in_file} events, buffer now {len(buffer_for_chunks)} rows.")
                current_part_idx = flush_rows(
                    buffer_for_chunks, args.output, current_part_idx, args.rows_per_file
                )
            
            # Handle any leftover rows in the buffer
            if buffer_for_chunks:
                out_path = args.output / f"part_{current_part_idx:05d}.parquet"
                pq.write_table(
                    pa.Table.from_pylist(list(buffer_for_chunks), schema=TARGET_SCHEMA), # Convert deque to list
                    out_path,
                    compression="zstd",
                )
                print(f"✓ wrote {len(buffer_for_chunks):,} rows (leftover) → {out_path.name}")
                total_rows_processed += len(buffer_for_chunks)
        else: # Not chunked_output, process directory into multiple output files
            # legacy per-file behaviour
            args.output.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
            total_rows_processed_all_files = 0
            for f_path in files:
                rows_for_single_output: Deque[Dict] = deque()
                print(f"Processing {f_path.name} for individual output...")
                process_one_file(f_path, rows_for_single_output)
                
                if not rows_for_single_output:
                    print(f"  No data processed from {f_path.name}. Output file will not be created.", file=sys.stderr)
                    continue

                # Output name based on input stem
                out_file_path = args.output / (f_path.stem + "_sensors.parquet")
                pq.write_table(
                    pa.Table.from_pylist(list(rows_for_single_output), schema=TARGET_SCHEMA), # Convert to list
                    out_file_path,
                    compression="zstd",
                )
                print(f"✓ {f_path.name} ({len(rows_for_single_output):,} rows) → {out_file_path.name}")
                total_rows_processed_all_files += len(rows_for_single_output)
            print(f"Total rows processed across all files: {total_rows_processed_all_files:,}")
    else:
        # This case should ideally be caught by the initial args.input.exists() check
        # if it's not a file or directory (e.g. broken symlink after check)
        sys.exit(f"ERROR: input path {args.input} is not a file or directory.")

if __name__ == "__main__":
    main()