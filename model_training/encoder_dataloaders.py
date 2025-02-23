import os
import torch
from encoder_SingleSessionSingleTrialDataset import SingleSessionSingleTrialDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
from pynwb import NWBHDF5IO


def get_datasets(bin_size=0.02, n_future_vel_bins=20, n_trials=20, session_string="sub-Jenkins/sub-Jenkins_ses-20090916_behavior+ecephys.nwb"):
    dataset_path = "000070"
    nwb_file_path = os.path.join(dataset_path, session_string)
    io = NWBHDF5IO(nwb_file_path, 'r')
    nwb_file = io.read()
    hand_data = nwb_file.processing['behavior'].data_interfaces['Position']['Hand'].data[:]
    hand_timestamps = nwb_file.processing['behavior'].data_interfaces['Position']['Hand'].timestamps[:]
    trial_data = nwb_file.intervals['trials']

    unit_spike_times = [nwb_file.units[unit_id]['spike_times'].iloc[0][:]
                        for unit_id in range(len(nwb_file.units))]
    n_neurons = len(unit_spike_times)

    datasets = [SingleSessionSingleTrialDataset(
        trial_data, hand_data, hand_timestamps, unit_spike_times, trial_id, bin_size=bin_size, n_future_vel_bins=n_future_vel_bins) for trial_id in range(n_trials)]

    return datasets, n_neurons


def get_dataloaders(datasets, bin_size=0.02, verbose=True, batch_size=100):
    dataset = torch.utils.data.ConcatDataset(datasets)
    if verbose:
        print(f"Datasets have {len(dataset)} samples")

    def custom_collate(batch):
        # Sort batch by sequence length in descending order
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)

        # Create list of tensors with different first dimensions
        velocities = [item[0] for item in batch]
        spikes = [item[1] for item in batch]
        spikes_future = [item[2] for item in batch]

        # Create nested tensor
        spikes = torch.nested.nested_tensor(spikes)
        velocities = torch.nested.nested_tensor(velocities)
        spikes_future = torch.nested.nested_tensor(spikes_future)

        return velocities, spikes, spikes_future

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(0.9 * total_size)  # 90% for training
    test_size = total_size - train_size  # Remaining 10% for testing
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    # Use custom collate function to handle variable sequence lengths
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=custom_collate)

    max_trial_length = get_max_trial_length(
        dataset, bin_size, min_max_trial_length_seconds=4)

    return train_loader, test_loader, test_dataset, max_trial_length


def get_max_trial_length(dataset, bin_size, min_max_trial_length_seconds=20):
    max_trial_length = max(dataset[i][0].shape[0] for i in range(len(dataset)))
    max_trial_length = max(max_trial_length, int(
        min_max_trial_length_seconds * 1 / bin_size))
    return max_trial_length


if __name__ == "__main__":
    datasets, n_neurons = get_datasets()
    print(n_neurons)
