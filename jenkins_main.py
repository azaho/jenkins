import time
import pygame
from torch import nn
from encoder_dataloaders import get_max_trial_length
from encoder_models import TransformerModel
from dynamixel_sdk import PortHandler, PacketHandler
from ikpy.link import OriginLink, URDFLink
from ikpy.chain import Chain
import torch
from encoder_SingleSessionSingleTrialDataset import SingleSessionSingleTrialDataset
import numpy as np
from pynwb import NWBHDF5IO
import os
from robot_setup import leader_setup, follower_setup


dataset_path = "000070"
nwb_file_path = os.path.join(
    dataset_path, "sub-Jenkins", "sub-Jenkins_ses-20090916_behavior+ecephys.nwb")
io = NWBHDF5IO(nwb_file_path, 'r')
nwb_file = io.read()
hand_data = nwb_file.processing['behavior'].data_interfaces['Position']['Hand'].data[:]
hand_timestamps = nwb_file.processing['behavior'].data_interfaces['Position']['Hand'].timestamps[:]
trial_data = nwb_file.intervals['trials']

unit_spike_times = [nwb_file.units[unit_id]['spike_times'].iloc[0][:]
                    for unit_id in range(len(nwb_file.units))]
n_neurons = len(unit_spike_times)
n_future_vel_bins = 40

trials_start_from = int(2000 * 0.95)
n_trials = int(2000 * 0.001)
datasets = [SingleSessionSingleTrialDataset(
    trial_data, hand_data, hand_timestamps, unit_spike_times, trial_id, bin_size=0.02, n_future_vel_bins=n_future_vel_bins) for trial_id in range(trials_start_from, trials_start_from + n_trials)]
dataset = torch.utils.data.ConcatDataset(datasets)
print(f"Dataset from {n_trials} trials has {len(dataset)} samples")


current_angles_leader = leader_setup.read_current_angles()
fk_leader = leader_setup.forward_kinematics(current_angles_leader)
print("Leader current angles (rad):", current_angles_leader)
print("Leader FK:\n", fk_leader)

# Example usage for follower:
current_angles_follower = follower_setup.read_current_angles()
fk_follower = follower_setup.forward_kinematics(current_angles_follower)
print("Follower current angles (rad):", current_angles_follower)
print("Follower FK:\n", fk_follower)


n_fr_bins = 9
d_model = 256
latent_dim = None
model_type = "transformer"  # transformer, lstm


n_trials = 200
n_epochs = 400
lr = 0.0005
weight_decay = 0.0

n_fr_bins = 9
bin_size = 0.02


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# get_max_trial_length(dataset, bin_size, min_max_trial_length_seconds=4)
max_trial_length = 200

input_size = (n_neurons) + 2 * n_future_vel_bins
hidden_size = d_model
model = TransformerModel(input_size, hidden_size,
                         n_neurons, n_fr_bins, max_trial_length).to(device)
checkpoint = torch.load(f'encoder_transformer.pt', map_location=device)
model.load_state_dict(checkpoint)

model.eval()

# Define forward model (taken from decoder_visualize.ipynb)
forward_model_input_size = n_neurons * 50
forward_model = nn.Sequential(
    nn.Linear(forward_model_input_size, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
).to(device)

# Load the trained model
forward_model.load_state_dict(torch.load('decoder_mlp.pth'))
forward_model.eval()  # Set the model to evaluation mode

# how many bins from some random trial to give the model before the model prediction starts (to give some context for the model)
n_context_bins = 50
current_bin = n_context_bins
max_n_bins = 5000

# # Get data from first trial
test_dataset = dataset
future_velocities, spikes, _ = test_dataset[1]
# Pad spikes and future_velocities to max_n_bins
spikes_padded = torch.zeros((max_n_bins, n_neurons), device=device)
spikes_padded[:spikes.shape[0]] = spikes
spikes = spikes_padded

future_velocities_padded = torch.zeros(
    (max_n_bins, 2, n_future_vel_bins), device=device)
future_velocities_padded[:future_velocities.shape[0]] = future_velocities
future_velocities = future_velocities_padded
# remove all spikes from the future that the model will predict
spikes[n_context_bins:] = 0

pred_velocities = np.zeros((len(spikes), 2))

position_multiplier = 5
read_positions = np.zeros((len(spikes), 2))
current_angles = leader_setup.read_current_angles()
current_fk = leader_setup.forward_kinematics(current_angles)
current_pos = current_fk[:3, 3]
read_positions[:] = current_pos[1:3]*position_multiplier

# remove all spikes from the future that the model will predict
spikes[n_context_bins:] = 0
time.sleep(3)
# Initialize Pygame
pygame.init()

# Set up display
fullscreen = True
if fullscreen:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    WIDTH, HEIGHT = screen.get_size()
    WIDTH -= 10  # XXX making it work on the mac screen
else:
    # Set up display
    WIDTH = 1200
    HEIGHT = 800  # Increased height to accommodate velocity plots
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Neural Spike Train and Velocity Visualization")

# # Colors
BLACK = (0, 0, 0)
GRAY = (140, 140, 140)  # For grid lines
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 40)  # Darker gray for velocity lines

# X offset for plots
X_OFFSET = 45


window_size = 150
bin_step = 1  # Number of bins to advance each frame (1 bin = 20ms)

# Calculate scaling factors
spike_plot_height = HEIGHT // 5 * 4
neuron_height = spike_plot_height // n_neurons
time_bin_width = WIDTH // window_size
plot_height = HEIGHT // 10  # Reduced height for each velocity plot


def normalize_for_plot(value, height):
    # Normalize values to fit in plot height
    return height // 2 + (value * height // 40)


# Create font for labels
font = pygame.font.SysFont('arial', 24)

running = True
clock = pygame.time.Clock()

# Pre-create surface for spike data
spike_surface = pygame.Surface((WIDTH, spike_plot_height))
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Add escape key to exit
                    running = False

        current_angles = leader_setup.read_current_angles()
        current_fk = leader_setup.forward_kinematics(current_angles)

        current_pos = current_fk[:3, 3]

        read_positions[current_bin] = current_pos[1:3]*position_multiplier
        # Clear screen
        screen.fill(BLACK)
        spike_surface.fill(BLACK)

        current_velocity = read_positions[current_bin] - \
            read_positions[current_bin-1]
        current_velocity = (read_positions[current_bin] -
                            read_positions[current_bin-2])/2
        future_velocities[current_bin, :, :-
                          1] = future_velocities[current_bin-1, :, 1:]
        future_velocities[current_bin, :, -
                          1] = torch.tensor(current_velocity, device=device)

        for future_bin_id in range(n_future_vel_bins):
            future_velocities[current_bin+future_bin_id, :, -1 -
                              future_bin_id] = torch.tensor(current_velocity, device=device)
        future_velocities[current_bin+n_future_vel_bins, :,
                          0] = torch.tensor(current_velocity, device=device)

        # Get model predictions for next timestep
        with torch.no_grad():
            give_context_from = max(current_bin-max_trial_length, 0)
            give_context_to = current_bin
            outputs = model(spikes[give_context_from:give_context_to].unsqueeze(
                # shape: [1, n_timesteps, n_neurons, n_fr_bins]
                0), future_velocities[give_context_from:give_context_to].unsqueeze(0))
            # shape: [1, n_timesteps, n_neurons, n_fr_bins]
            pred_probs = torch.softmax(outputs, dim=3)
            pred_sample = torch.multinomial(
                pred_probs.reshape(-1, n_fr_bins), 1)
            # shape: [1, n_timesteps, n_neurons]
            pred_sample = pred_sample.reshape(
                outputs.shape[0], outputs.shape[1], outputs.shape[2])
            # shape: [n_neurons, ] -- this is the prediction for the spikes in the future timestep
            last_sample = pred_sample[:, -1, :].squeeze(0)

            # spikes[give_context_from:give_context_to]zed
            spikes[current_bin] = last_sample

        show_spikes_from = max(
            current_bin+n_future_vel_bins-window_size, n_context_bins)
        show_spikes_to = min(current_bin+n_future_vel_bins, len(spikes))
        # Draw spike trains first using numpy operations
        # Get the current window of spike data
        # shape: [n_neurons, n_timesteps]
        spikes_numpy = spikes.cpu().numpy().T
        spike_data_normalized = spikes_numpy**2 / 64
        window_data = spike_data_normalized[:, show_spikes_from:show_spikes_to]

        # Convert to pixel values (0-255)
        pixel_values = np.minimum(
            window_data * 255 * 1.5, 255).astype(np.uint8)

        # Create a surface from the numpy array
        for neuron in range(n_neurons):
            row_data = pixel_values[neuron]
            for t, intensity in enumerate(row_data):
                if intensity > 0:  # Only draw if there's activity
                    pygame.draw.rect(spike_surface, (intensity, intensity, intensity),
                                     (X_OFFSET + t * time_bin_width, neuron * neuron_height,
                                     time_bin_width, neuron_height))

        # Draw the spike surface to the screen
        screen.blit(spike_surface, (0, 0))

        # Draw grid lines and channel numbers on top
        for i in range(0, spike_plot_height, neuron_height * n_neurons):  # Draw every 200 channels
            pygame.draw.line(screen, GRAY, (X_OFFSET, i), (WIDTH, i), 1)
            # Draw channel number
            label = font.render(str(i // neuron_height), True, WHITE)
            # Rotate the label surface
            rotated_label = pygame.transform.rotate(label, 90)
            screen.blit(rotated_label, (10, i))

        # Draw velocity plots
        true_velocities_numpy = future_velocities.cpu().numpy()[:, :, 0] * 200
        y_pred = forward_model(
            spikes[current_bin-50:current_bin].T.reshape(-1).unsqueeze(0)).detach().cpu().numpy()
        pred_velocities[current_bin] = y_pred.flatten() * 200

        y_offset = spike_plot_height - 10  # Start below spike plot

        # Draw X velocity plot
        pygame.draw.line(screen, DARK_GRAY, (X_OFFSET, y_offset +
                         plot_height//2), (WIDTH, y_offset + plot_height//2), 1)

        show_vel_from = max(current_bin+n_future_vel_bins -
                            window_size, n_context_bins)
        show_vel_to = min(current_bin+n_future_vel_bins+1, len(spikes))
        # Pre-calculate positions for velocity plots
        t_range = np.arange(show_vel_to-show_vel_from-1)
        max_pred_show_t = max(len(t_range) - n_future_vel_bins, 0)
        x_coords = X_OFFSET + t_range * time_bin_width
        x_coords_next = X_OFFSET + (t_range + 1) * time_bin_width

        # X velocity
        # get the first X velocity from the future (aka current/next velocity)
        true_vel_x = y_offset + \
            normalize_for_plot(
                true_velocities_numpy[show_vel_from:show_vel_to-1, 0], plot_height)
        true_vel_x_next = y_offset + \
            normalize_for_plot(
                true_velocities_numpy[show_vel_from+1:show_vel_to, 0], plot_height)
        pred_vel_x = y_offset + \
            normalize_for_plot(
                pred_velocities[show_vel_from:show_vel_to-1, 0], plot_height)
        pred_vel_x_next = y_offset + \
            normalize_for_plot(
                pred_velocities[show_vel_from+1:show_vel_to, 0], plot_height)

        # Draw lines in batches
        for i in range(len(x_coords)):
            pygame.draw.line(screen, GRAY,
                             (int(x_coords[i]), int(true_vel_x[i])),
                             (int(x_coords_next[i]), int(true_vel_x_next[i])), 2)
            if i >= max_pred_show_t:
                continue
            pygame.draw.line(screen, WHITE,
                             (int(x_coords[i]), int(pred_vel_x[i])),
                             (int(x_coords_next[i]), int(pred_vel_x_next[i])), 2)

        # Draw Y velocity plot
        y_offset += plot_height + 10
        pygame.draw.line(screen, DARK_GRAY, (X_OFFSET, y_offset +
                         plot_height//2), (WIDTH, y_offset + plot_height//2), 1)

        # Y velocity
        true_vel_y = y_offset + \
            normalize_for_plot(
                true_velocities_numpy[show_vel_from:show_vel_to-1, 1], plot_height)
        true_vel_y_next = y_offset + \
            normalize_for_plot(
                true_velocities_numpy[show_vel_from+1:show_vel_to, 1], plot_height)
        pred_vel_y = y_offset + \
            normalize_for_plot(
                pred_velocities[show_vel_from:show_vel_to-1, 1], plot_height)
        pred_vel_y_next = y_offset + \
            normalize_for_plot(
                pred_velocities[show_vel_from+1:show_vel_to, 1], plot_height)

        # Draw lines in batches
        for i in range(len(x_coords)):
            pygame.draw.line(screen, GRAY,
                             (int(x_coords[i]), int(true_vel_y[i])),
                             (int(x_coords_next[i]), int(true_vel_y_next[i])), 2)
            if i >= max_pred_show_t:
                continue
            pygame.draw.line(screen, WHITE,
                             (int(x_coords[i]), int(pred_vel_y[i])),
                             (int(x_coords_next[i]), int(pred_vel_y_next[i])), 2)

        # Draw axis labels
        time_label = font.render("Time", True, WHITE)
        channels_label = font.render("Channels", True, WHITE)
        x_vel_label = font.render(
            "velocity X (prediction: WHITE)", True, WHITE)
        y_vel_label = font.render(
            "velocity Y (prediction: WHITE)", True, WHITE)

        screen.blit(time_label, (WIDTH // 2 - 30, HEIGHT - 30))
        screen.blit(x_vel_label, (WIDTH // 2 - 150, spike_plot_height - 10))
        screen.blit(y_vel_label, (WIDTH // 2 - 150,
                    spike_plot_height + plot_height - 15))

        # Rotate and draw y-axis label
        channels_surface = pygame.Surface((200, 30))
        channels_surface.fill(BLACK)
        channels_surface.blit(channels_label, (50, 0))
        channels_surface = pygame.transform.rotate(channels_surface, 90)
        screen.blit(channels_surface, (10, spike_plot_height // 2 - 100))

        # Update display
        pygame.display.flip()

        # Move window by one bin (20ms) each frame
        current_bin += bin_step
        if current_bin == max_n_bins:
            break

        # leader movement
        vx, vy = pred_velocities[current_bin][0],  pred_velocities[current_bin][1]

        if current_bin % 2 == 0:
            lambda_decay = 0.9
            integrated_v = np.zeros(2)
            for i in range(0, current_bin):
                integrated_v = lambda_decay * integrated_v + pred_velocities[i]
            integrated_v = integrated_v / 70
            integrated_v = integrated_v.clip(-1, 1)
            integrated_v = integrated_v * 15
            # print(integrated_v)
            x, y = integrated_v[0], integrated_v[1]

            current_angles = follower_setup.read_current_angles()
            follower_setup.move_to_target_from_current(
                18, -x, y+25, current_angles=current_angles)

        # Control frame rate to 50 FPS (20ms per frame)
        clock.tick(100)
except Exception as e:
    print(f"Error occurred: {e}")

pygame.quit()
