import numpy as np
import awkward as ak
import glob
import time
from collections import defaultdict
import sys
import random

from icecube import dataio, dataclasses

def load_i3_data(filename):
    # Create empty lists to store the data
    total_hits = []
    labels = []
    geo = np.load('/n/home10/felixyu/nt_mlreco/scratch/geo_dict.npy', allow_pickle=True).item()
    # Open the data file
    with dataio.I3File(filename) as f:
        # Loop over all frames in the file
        while f.more():
            frame = f.pop_physics()
            pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'InIceDSTPulses')
            om_keys = pulse_map.keys()
            pulses = pulse_map.values()
            pulse_times = [[obj.time for obj in inner_list] for inner_list in pulses]
            pulse_charges = [[obj.charge for obj in inner_list] for inner_list in pulses]

            om_pos = []
            for omkey in om_keys:
                om_pos.append(geo[omkey])
            om_pos = np.array(om_pos)

            # convert nested list to flat array
            pulse_times_arr = np.concatenate(pulse_times)
            pulse_charges_arr = np.concatenate(pulse_charges)

            # repeat arr1 for each element in lst
            om_pos = np.repeat(om_pos, [len(l) for l in pulse_times], axis=0)

            # concatenate arr1_repeated and arr2
            hits = np.concatenate((om_pos, pulse_times_arr.reshape((-1, 1)), pulse_charges_arr.reshape((-1, 1))), axis=1)
            total_hits.append(hits)

    return total_hits

dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Lab/IceCube_MC/22079/0000000-0000999/'
output_dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/felixyu/IceCube_MC_Pulses/22079/0000000-0000999/'
# indices = [25, 26, 27, 37, 50, 53, 54, 55, 56, 57, 58, 59]
files_list = sorted(glob.glob(dir + '*.i3.zst'))
file = files_list[int(sys.argv[1])]

print("Processing file: {}".format(file))
import pdb; pdb.set_trace()
data_labels = []
chunk_size = 250000
file_id = 0
for event in data:
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
        first_dom_hit = dom_times.min()
        
        # bin hits on individual sensors
        num_bins = 3000
        max_time = 3000
        bin_edges = np.linspace(0, max_time, num_bins + 1, endpoint=True)
        
        dom_times = dom_times - first_dom_hit # shift by first hit time
        
        # do not consider hits with time > max_time
        max_time_mask = (dom_times < max_time)
        
        binned_times = np.digitize(dom_times[max_time_mask], bin_edges, right=True)
        binned_time_counts = np.histogram(binned_times, bins=bin_edges)[0]
        
        # put hits with time > max_time in the last bin
        binned_time_counts[-1] += np.sum(~max_time_mask)
        
        dom_data = {'first_hit_time': first_dom_hit, 'binned_time_counts': binned_time_counts.astype(np.int32).tolist()}
        data_labels.append(dom_data)
        
        # whenever data_labels hits chunk_size, save to parquet and reset
        if len(data_labels) == chunk_size:
            ak.to_parquet(data_labels, output_dir + str(int(sys.argv[1])) + '_chunk_' + str(file_id) + '.parquet')
            del data_labels
            data_labels = []
            file_id += 1
            print('chunk done!')

ak.to_parquet(data_labels, output_dir + str(int(sys.argv[1])) + '_chunk_' + str(file_id) + '.parquet')