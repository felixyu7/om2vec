import numpy as np
import awkward as ak
import polars
import glob, sys, time
from collections import defaultdict

def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        dest="input_dir_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--chunk_size",
        dest="chunk_size",
        type=int,
        required=False,
        default=250000
    )
    parser.add_argument(
        "--num_bins",
        dest="num_bins",
        type=int,
        required=False,
        default=6400
    )
    parser.add_argument(
        "--max_time",
        dest="max_time",
        type=int,
        required=False,
        default=6400
    )
    args = parser.parse_args()
    return args

if __name__=="__main__":

    args = initialize_args()

    files = sorted(glob.glob(args.input_dir_path + '*.parquet'))
    output_dir = args.output_dir_path

    data_labels = []
    chunk_size = args.chunk_size
    file_id = 0
    for file in files:
        print("Processing file: {}".format(file))
        data = ak.from_parquet(file)
        for event in data:
            if len(event.photons.t) == 0:
                continue
            stime = time.time()
            pos_t = np.array([event.photons.sensor_pos_x.to_numpy(),
                        event.photons.sensor_pos_y.to_numpy(),
                        event.photons.sensor_pos_z.to_numpy(),
                        event.photons.string_id.to_numpy(),
                        event.photons.sensor_id.to_numpy(),
                        event.photons.t.to_numpy() - event.photons.t.to_numpy().min()]).T
            unique_coords_dict = defaultdict(list)
            for i, coord in enumerate(pos_t[:, :5]):
                unique_coords_dict[tuple(coord)].append(i)
            for coord, indices in unique_coords_dict.items():
                mask = np.zeros(pos_t.shape[0], dtype=bool)
                mask[indices] = True
                
                dom_times = pos_t[:,-1][mask]
                
                # bin hits on individual sensors
                num_bins = args.num_bins
                max_time = args.max_time
                bin_edges = np.linspace(0, max_time, num_bins + 1, endpoint=True)
                
                dom_times = dom_times
                
                # do not consider hits with time > max_time
                max_time_mask = (dom_times < max_time)
                
                binned_times = np.digitize(dom_times[max_time_mask], bin_edges, right=True)
                binned_time_counts = np.histogram(binned_times, bins=bin_edges)[0]
                
                # put hits with time > max_time in the last bin
                binned_time_counts[-1] += np.sum(~max_time_mask)
                
                dom_data = {'binned_time_counts': binned_time_counts.astype(np.int32).tolist(), 'pos': list(coord)}
                data_labels.append(dom_data)
                
                # whenever data_labels hits chunk_size, save to parquet and reset
                if len(data_labels) == chunk_size:
                    ak.to_parquet(data_labels, output_dir + 'chunk_' + str(file_id) + '.parquet')
                    del data_labels
                    data_labels = []
                    file_id += 1
                    print('chunk done!')

    ak.to_parquet(data_labels, output_dir + 'chunk_' + str(file_id) + '.parquet')