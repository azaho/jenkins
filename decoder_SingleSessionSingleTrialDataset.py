import torch
from torch.utils.data import Dataset
import numpy as np
from pynwb import NWBHDF5IO

import os


class SingleSessionSingleTrialDataset(Dataset):
    def __init__(self, trial_data, hand_data, hand_timestamps, unit_spike_times, trial_id, bin_size=0.02, n_context_bins=50):
        """
        Dataset for a single trial from a single session, providing binned spike counts and hand velocity labels

        Args:
            hand_data (np.ndarray): Hand data
            hand_timestamps (np.ndarray): Hand timestamps
            trial_id (int): Trial ID to extract data from
        """
        self.unit_spike_times = unit_spike_times
        self.n_neurons = len(self.unit_spike_times)

        # Get trial start and stop times
        self.trial_start = trial_data['start_time'][trial_id]
        self.trial_stop = trial_data['stop_time'][trial_id]

        # Get data only for this trial
        trial_mask = (hand_timestamps >= self.trial_start) & (
            hand_timestamps <= self.trial_stop)
        self.hand_data = hand_data[trial_mask]
        self.hand_timestamps = hand_timestamps[trial_mask]

        # cut the trial short if it goes beyond the hand data
        self.trial_stop = min(self.trial_stop, self.hand_timestamps[-1])
        self.num_bins = int((self.trial_stop - self.trial_start) / bin_size)
        self.n_context_bins = n_context_bins

        # Create spike count bins
        bin_edges = np.linspace(
            self.trial_start, self.trial_stop, self.num_bins + 1)
        self.spike_counts = np.zeros((self.n_neurons, self.num_bins))

        # Fill spike count matrix
        for neuron_idx in range(self.n_neurons):
            spike_times = self.unit_spike_times[neuron_idx]
            spike_mask = (self.trial_start <= spike_times) & (
                spike_times <= self.trial_stop)
            filtered_spikes = spike_times[spike_mask]
            counts, _ = np.histogram(filtered_spikes, bins=bin_edges)
            self.spike_counts[neuron_idx] = counts

        # Create hand position matrix (2 x num_bins)
        self.hand_matrix = np.zeros((2, self.num_bins))
        # For each bin, calculate mean hand position of all samples in that bin
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for bin_idx in range(self.num_bins):
            # Find all timestamps that fall within this bin
            bin_mask = (self.hand_timestamps >= bin_edges[bin_idx]) & (
                self.hand_timestamps < bin_edges[bin_idx + 1])
            if np.any(bin_mask):
                # Take mean of all positions in the bin
                self.hand_matrix[:, bin_idx] = np.mean(
                    self.hand_data[bin_mask], axis=0)
            else:
                print(f"No samples in bin {bin_idx} for trial {trial_id}")
                # If no samples in bin, use nearest neighbor
                closest_idx = np.argmin(
                    np.abs(self.hand_timestamps - bin_centers[bin_idx]))
                self.hand_matrix[:, bin_idx] = self.hand_data[closest_idx]

        # Convert matrices to torch tensors
        self.hand_matrix = torch.FloatTensor(
            self.hand_matrix) / 200  # normalize
        self.spike_counts = torch.FloatTensor(self.spike_counts)

    def __len__(self):
        return self.num_bins - self.n_context_bins

    def __getitem__(self, idx):
        return self.spike_counts[:, idx:idx + self.n_context_bins], self.hand_matrix[:, idx + self.n_context_bins - 1]-self.hand_matrix[:, idx + self.n_context_bins - 2]


if __name__ == "__main__":
    dataset_path = "000070"
    nwb_file_path = os.path.join(
        dataset_path, "sub-Jenkins", "sub-Jenkins_ses-20090916_behavior+ecephys.nwb")
    io = NWBHDF5IO(nwb_file_path, 'r')
    nwb_file = io.read()
    hand_data = nwb_file.processing['behavior'].data_interfaces['Position']['Hand'].data[:]
    hand_timestamps = nwb_file.processing['behavior'].data_interfaces['Position']['Hand'].timestamps[:]
    trial_data = nwb_file.intervals['trials'][0]

    unit_spike_times = [nwb_file.units[unit_id]['spike_times'].iloc[0][:]
                        for unit_id in range(len(nwb_file.units))]

    dataset = SingleSessionSingleTrialDataset(
        trial_data, hand_data, hand_timestamps, unit_spike_times, 0, bin_size=0.02, n_context_bins=50)
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1])
