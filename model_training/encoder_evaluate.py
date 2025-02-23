import torch
import os
import matplotlib.pyplot as plt


def visualize_with_real_data(model, test_loader, n_neurons, n_fr_bins, device, prefix, temperature=1.0):
    with torch.no_grad():
        # Get a single batch from test loader
        batch_velocities, batch_spikes, batch_spikes_future = next(
            iter(test_loader))
        batch_spikes = batch_spikes.to_padded_tensor(0).to(device)
        batch_velocities = batch_velocities.to_padded_tensor(0).to(device)

        # Get model predictions
        # Shape: (batch, n_context_bins-1, n_neurons, n_fr_bins)
        outputs = model(batch_spikes, batch_velocities)

        # Apply temperature scaling
        scaled_outputs = outputs / temperature
        test_pred_probs = torch.softmax(
            scaled_outputs, dim=3)  # Get probabilities

        # Get both argmax and sampled predictions
        pred_classes = torch.argmax(outputs, dim=3).cpu().numpy()
        test_pred_sample = torch.multinomial(
            test_pred_probs.reshape(-1, n_fr_bins), 1)
        test_pred_sample = test_pred_sample.reshape(
            outputs.shape[0], outputs.shape[1], outputs.shape[2]).cpu().numpy()

    # Create figure with subplots and shared x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(
        16, 12), height_ratios=[1, 1, 1], sharex=True)

    # Plot heatmap of true data
    max_show_timesteps = 1000
    # Get ground truth from next timesteps
    true_data = batch_spikes[:, 1:, :n_neurons].cpu().numpy()
    im0 = ax1.imshow(true_data.reshape(-1, n_neurons)
                     [:max_show_timesteps, :].T, aspect='auto', cmap='viridis')
    ax1.set_title('True Neural Activity')
    ax1.set_ylabel('Neuron')
    plt.colorbar(im0, ax=ax1, orientation='horizontal',
                 pad=0.1, aspect=40, label='Spike Counts')

    # Plot heatmap of argmax predictions
    im1 = ax2.imshow(pred_classes.reshape(-1, n_neurons)
                     [:max_show_timesteps, :].T, aspect='auto', cmap='viridis')
    ax2.set_title('Argmax Predicted Neural Activity')
    ax2.set_ylabel('Neuron')
    plt.colorbar(im1, ax=ax2, orientation='horizontal',
                 pad=0.1, aspect=40, label='Spike Counts')

    # Plot heatmap of sampled predictions
    im2 = ax3.imshow(test_pred_sample.reshape(-1, n_neurons)
                     [:max_show_timesteps, :].T, aspect='auto', cmap='viridis')
    ax3.set_title('Sampled Predicted Neural Activity')
    ax3.set_ylabel('Neuron')
    ax3.set_xlabel('Time Step')
    plt.colorbar(im2, ax=ax3, orientation='horizontal',
                 pad=0.1, aspect=40, label='Spike Counts')

    # Create model_data directory if it doesn't exist
    os.makedirs('model_data', exist_ok=True)
    plt.savefig('model_data/'+prefix+'_real_data.pdf', bbox_inches='tight')


def visualize_with_its_own_data(model, test_dataset, n_neurons, n_fr_bins, device, prefix, temperature=1.0):
    model.eval()

    # Initialize lists to store results
    all_spikes = []
    all_modified_spikes = []

    # Loop through all trials in test dataset
    for trial_idx in range(len(test_dataset)):
        if trial_idx > 20:
            break  # only do 20 trials
        # Get data for this trial
        velocities, spikes, spikes_future = test_dataset[trial_idx]
        spikes = spikes.to(device).unsqueeze(0)
        velocities = velocities.to(device).unsqueeze(0)

        # Create a copy that we'll modify
        n_context_bins = 30
        modified_spikes = spikes.clone()
        modified_spikes[:, n_context_bins:, :] = 0

        for i in range(n_context_bins, modified_spikes.size(1)):
            # Get model predictions
            outputs = model(modified_spikes[:, :i, :],
                            velocities[:, :i, :])

            # Sample from probability distribution with temperature
            logits = outputs / temperature  # Apply temperature scaling
            pred_probs = torch.softmax(logits, dim=3)  # Get probabilities
            pred_sample = torch.multinomial(
                pred_probs.reshape(-1, n_fr_bins), 1)  # Sample from probabilities
            pred_sample = pred_sample.reshape(
                outputs.shape[0], outputs.shape[1], outputs.shape[2])

            # Update next timestep with predictions
            modified_spikes[:, i, :] = pred_sample[:, -1, :]

        # Store results for this trial
        all_spikes.append(spikes.squeeze(0))
        all_modified_spikes.append(modified_spikes.squeeze(0))

    # Concatenate results from all trials
    all_spikes = torch.cat(all_spikes, dim=0)
    all_modified_spikes = torch.cat(all_modified_spikes, dim=0)

    # Create figure with subplots and shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(
        16, 8), height_ratios=[1.2, 1], sharex=True)

    im = ax1.imshow(all_spikes.cpu().numpy().T, aspect='auto', cmap='viridis')
    ax1.set_title('Neural Activity (Averaged Across Context Window)')
    ax1.set_ylabel('Neuron')
    # Add smaller colorbar at the top
    cbar = plt.colorbar(im, ax=ax1, orientation='horizontal',
                        pad=0.1, aspect=40, label='Spike Counts')

    # Add vertical lines for trial boundaries and context windows
    trial_lengths = [len(test_dataset[i][0])
                     for i in range(min(21, len(test_dataset)))]
    current_pos = 0
    for length in trial_lengths:
        # Add trial boundary line
        ax1.axvline(x=current_pos, color='white', linestyle='-', alpha=0.5)
        ax2.axvline(x=current_pos, color='white', linestyle='-', alpha=0.5)

        # Add context window line
        context_pos = current_pos + n_context_bins
        ax1.axvline(x=context_pos, color='black', linestyle='-', alpha=0.5)
        ax2.axvline(x=context_pos, color='black', linestyle='-', alpha=0.5)

        current_pos += length

    im = ax2.imshow(all_modified_spikes.cpu().numpy().T,
                    aspect='auto', cmap='viridis')
    ax2.set_title('Sampled Predicted Neural Activity')
    ax2.set_ylabel('Neuron')

    plt.savefig('model_data/'+prefix+'_own_data.pdf', bbox_inches='tight')
