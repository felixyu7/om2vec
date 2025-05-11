#!/usr/bin/env python3
# preprocess_prometheus.py   ← third-time-lucky version

import argparse, pathlib
from typing import List, Dict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ────────────────── target schema (sensor-level rows) ──────────────────
TARGET = pa.schema(
    [
        ("event_id",     pa.uint32()),
        ("string_id",    pa.int64()),
        ("sensor_id",    pa.int64()),
        ("sensor_pos_x", pa.float64()),
        ("sensor_pos_y", pa.float64()),
        ("sensor_pos_z", pa.float64()),
        ("nhits",        pa.int32()),
        ("hits_t",       pa.list_(pa.float64())),
        ("hits_id_idx",  pa.list_(pa.int64())),
    ]
)

# ────────────────── helper: explode one event ──────────────────
def explode_event(photons: Dict, event_id: int) -> List[Dict]:
    """Turn one event's photon dict → list of per-sensor dicts."""
    s_id  = np.asarray(photons["string_id"])
    p_id  = np.asarray(photons["sensor_id"])
    pos_x = np.asarray(photons["sensor_pos_x"])
    pos_y = np.asarray(photons["sensor_pos_y"])
    pos_z = np.asarray(photons["sensor_pos_z"])
    t     = np.asarray(photons["t"])
    idx   = np.asarray(photons["id_idx"])

    pairs, inv = np.unique(np.stack([s_id, p_id], 1), axis=0, return_inverse=True)

    out: List[Dict] = []
    for k, (sid, pid) in enumerate(pairs):
        m = inv == k
        out.append(
            dict(
                event_id     = event_id,
                string_id    = int(sid),
                sensor_id    = int(pid),
                sensor_pos_x = float(pos_x[m][0]),
                sensor_pos_y = float(pos_y[m][0]),
                sensor_pos_z = float(pos_z[m][0]),
                nhits        = int(m.sum()),
                hits_t       = t[m].tolist(),
                hits_id_idx  = idx[m].tolist(),
            )
        )
    return out

# ────────────────── main streaming loop ──────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-arrange event-level parquet so each row "
                    "corresponds to one sensor in one event."
    )
    ap.add_argument("input",  type=pathlib.Path)
    ap.add_argument("output", type=pathlib.Path)
    ap.add_argument("--batch", type=int, default=200,
                    help="flush after N sensor rows (controls memory)")
    args = ap.parse_args()

    pf      = pq.ParquetFile(args.input)
    writer  = pq.ParquetWriter(args.output, TARGET, compression="zstd")
    buf: List[Dict] = []
    ev_id   = 0

    for rg in range(pf.num_row_groups):
        # ask only for the photons struct column
        tbl = pf.read_row_group(rg, columns=["photons"])

        photons_col = tbl.column(0)             # only column in the table
        for row_idx in range(tbl.num_rows):
            photons = photons_col[row_idx].as_py()     # → plain dict of lists
            buf.extend(explode_event(photons, ev_id))
            ev_id += 1

            if len(buf) >= args.batch:
                writer.write_table(pa.Table.from_pylist(buf, schema=TARGET))
                buf.clear()

    if buf:
        writer.write_table(pa.Table.from_pylist(buf, schema=TARGET))

    writer.close()
    print(f"Sensor-level parquet written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
