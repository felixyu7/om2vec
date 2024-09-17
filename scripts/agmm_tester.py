import numpy as np
from scipy.optimize import minimize
import awkward as ak
from sklearn.cluster import KMeans
import awkward as ak
import glob
import time
from collections import defaultdict
import sys

from scipy.stats import kstest
from scipy import interpolate
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import skew

def asymmetric_gaussian(x, mu, sigma, r):
    N = 2 / (np.sqrt(2 * np.pi) * sigma * (r + 1))
    return np.where(x <= mu,
                    N * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)),
                    N * np.exp(-(x - mu) ** 2 / (2 * (sigma * r) ** 2)))

def mixture_density(x, mus, sigmas, rs, weights, n_components):
    density = np.zeros_like(x)
    for i in range(n_components):
        density += weights[i] * asymmetric_gaussian(x, mus[i], sigmas[i], rs[i])
    return density

def initialize_parameters(data, n_components):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_components).fit(data.reshape(-1, 1))
    
    # Initialize parameters
    mus = kmeans.cluster_centers_.flatten()
    sigmas = np.zeros(n_components)
    weights = np.zeros(n_components)
    rs = np.ones(n_components)
    
    # Estimate standard deviations, weights, and skewness ('r' values)
    labels = kmeans.labels_
    for i in range(n_components):
        component_data = data[labels == i]
        sigmas[i] = np.std(component_data)
        weights[i] = len(component_data) / len(data)
        component_skewness = skew(component_data)
        # Mapping skewness to 'r' values
        if component_skewness > 0:
            rs[i] = 1 + (np.abs(component_skewness)[0] * 100)  # Example mapping
        elif component_skewness < 0:
            rs[i] = 1 / (1 + (np.abs(component_skewness)[0] * 100))  # Example mapping
        else:
            rs[i] = 1
    return mus, sigmas, rs, weights

def log_likelihood(params, x, counts, n_components):
    """
    Compute the log likelihood of the data given the mixture model parameters.
    """
    log_likelihood = 0
    for i in range(n_components):
        mu, sigma, r, weight = params[i*4:(i+1)*4]
        N = 2 / (np.sqrt(2 * np.pi) * sigma * (r + 1))
        gaussian_left = N * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        gaussian_right = N * np.exp(-(x - mu) ** 2 / (2 * (sigma * r) ** 2))
        gaussian = np.where(x <= mu, gaussian_left, gaussian_right)
        log_likelihood += weight * gaussian
    weighted_log_likelihood = counts * np.log(log_likelihood + 1e-8)
    return (-np.sum(weighted_log_likelihood)) / np.sum(counts)

def fit_mixture_asymmetric_gaussians(x, n_components):
    """
    Fit a mixture of asymmetric Gaussians to the data.
    """
    # Initial guesses for parameters: [mu, sigma, r, weight] for each component
    mus, sigmas, rs, weights = initialize_parameters(x, n_components)
    params_initial = np.random.rand(n_components * 4)
    params_initial[::4] = mus
    params_initial[1::4] = sigmas
    params_initial[2::4] = rs
    params_initial[3::4] = weights

    # Constraints: weights sum to 1
    constraints = ({
        'type': 'eq',
        'fun': lambda params: np.sum(params[3::4]) - 1
    })

    # Bounds for each parameter
    bounds = [(None, None), (0.01, None), (0.01, None), (0.0, 1.0)] * n_components

    # truncate to nanosecond for efficiency
    trunc_x = np.trunc(x)
    trunc_x, counts = np.unique(trunc_x, return_counts=True)
    
    result = minimize(log_likelihood, params_initial, args=(trunc_x, counts, n_components), constraints=constraints, bounds=bounds,
                      method='SLSQP')
    fitted_params = result.x

    # Extract the parameters for each component
    mus = fitted_params[::4]
    sigmas = fitted_params[1::4]
    rs = fitted_params[2::4]
    weights = fitted_params[3::4]

    return mus, sigmas, rs, weights

dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Lab/Prometheus_MC/IceCube_HE/EMinus/'
files_list = sorted(glob.glob(dir + 'Generation*.parquet'))
pos_offset = np.array([0, 0, 2000])

file = files_list[int(sys.argv[1])]

print("Processing file: {}".format(file))
data = ak.from_parquet(file)
num_components = 16
results = {'num_hits': [], 'js_div': [], 'time': []}
for event in data:
    if len(event.photons.t) == 0:
        continue
    pos_t = np.array([event.photons.sensor_pos_x.to_numpy(),
                event.photons.sensor_pos_y.to_numpy(),
                event.photons.sensor_pos_z.to_numpy(),
                event.photons.string_id.to_numpy(),
                event.photons.sensor_id.to_numpy(),
                event.photons.t.to_numpy() - event.photons.t.to_numpy().min()]).T
    unique_coords_dict = defaultdict(list)
    for i, coord in enumerate(pos_t[:, :5]):
        unique_coords_dict[tuple(coord)].append(i)
    event_labels = {'string_sensor_pos': [], 'pos': [], 'num_hits': [], 'means': [], 'variances': [], 'rs': [], 'weights': []}
    for coord, indices in unique_coords_dict.items():
        stime = time.time()
        event_labels['pos'].append((np.array(coord)[:3] + pos_offset).tolist())
        event_labels['string_sensor_pos'].append((np.array(coord)[3:5]).tolist())
        mask = np.zeros(pos_t.shape[0], dtype=bool)
        mask[indices] = True
        mask_sum = mask.sum()
        event_labels['num_hits'].append(mask_sum)
        time_series = pos_t[:,-1][mask].reshape(-1, 1)
        
        if np.unique(pos_t[:,-1][mask]).shape[0] < num_components:
            mus = np.zeros(num_components)
            sigmas = np.ones(num_components)
            rs = np.ones(num_components)
            weights = np.zeros(num_components)
            for i in range(mask_sum):
                mus[i] = pos_t[:,-1][mask][i]
                weights[i] = 1. / mask_sum
        else:
            mus, sigmas, rs, weights = fit_mixture_asymmetric_gaussians(time_series, num_components)
            
        results['time'].append(time.time() - stime)
        # # truncate to nanosecond for efficiency
        # trunc_x = np.trunc(time_series)
        # trunc_x, counts = np.unique(trunc_x, return_counts=True)
        
        # result = fit_mixture_model(trunc_x.reshape(-1, 1), counts.reshape(-1, 1) / counts.max(), num_components, log_seed=False)
        # fitted_params = result.x

        # # Extract the parameters for each component
        # mus = fitted_params[::4]
        # sigmas = fitted_params[1::4]
        # rs = fitted_params[2::4]
        # weights = fitted_params[3::4]
        
        x_range = np.linspace(0, 6400, 6400)
        y_mixture = mixture_density(x_range, mus, sigmas, rs, weights, num_components)
        pdf = y_mixture / (y_mixture.sum(axis=-1) + 1e-8)

        # num_bins = 6400
        # max_time = 6400
        # bin_edges = np.linspace(0, max_time, num_bins + 1, endpoint=True)
        
        # dom_times = pos_t[:,-1][mask]
        
        # # do not consider hits with time > max_time
        # max_time_mask = (dom_times < max_time)
        
        # binned_times = np.digitize(dom_times[max_time_mask], bin_edges, right=True)
        # binned_time_counts = np.histogram(binned_times, bins=bin_edges)[0]
        
        # # put hits with time > max_time in the last bin
        # binned_time_counts[-1] += np.sum(~max_time_mask)
        
        # results['inputs'].append(binned_time_counts)
        # results['output_pdf'].append(pdf)
    
        # if len(results['inputs']) == 100:
        #     np.save('./results/agmm_outputs_EMinus_{}.npy'.format(sys.argv[1]), results)
            # exit()
    
        # # compute the poisson nll for the time series and pdf
        if np.unique(pos_t[:,-1][mask]).shape[0] < num_components:
            js_div = 0.
        else:
            # compute js div
            num_bins = 6400
            max_time = 6400
            bin_edges = np.linspace(0, max_time, num_bins + 1, endpoint=True)
            
            dom_times = pos_t[:,-1][mask]
            
            # do not consider hits with time > max_time
            max_time_mask = (dom_times < max_time)
            
            binned_times = np.digitize(dom_times[max_time_mask], bin_edges, right=True)
            binned_time_counts = np.histogram(binned_times, bins=bin_edges)[0]
            
            # put hits with time > max_time in the last bin
            binned_time_counts[-1] += np.sum(~max_time_mask)
            
            # normalize binned_time_counts to 1
            counts_prob = binned_time_counts / binned_time_counts.sum()
            js_div = jensenshannon(counts_prob, pdf)
            if np.isnan(js_div):
                js_div = 1.
            if np.isinf(js_div):
                js_div = 1.
            
        results['num_hits'].append(mask_sum)
        results['js_div'].append(js_div)
        
np.save('./results/agmm_16/agmm_metrics_EMinus_{}.npy'.format(sys.argv[1]), results) 