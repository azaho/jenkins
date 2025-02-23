import torch
import torch.nn as nn
import math


class SpikeEncoder(nn.Module):
    def __init__(self, n_neurons, n_context_bins, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_neurons, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, n_context_bins, n_neurons)
        return self.encoder(x)


class SpikeDecoder(nn.Module):
    def __init__(self, n_neurons, n_context_bins, n_fr_bins, latent_dim=16):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_context_bins = n_context_bins
        self.n_fr_bins = n_fr_bins
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, n_neurons * n_fr_bins)
        )

    def forward(self, x):
        # x shape: (batch_size, latent_dim)
        x = self.decoder(x)
        return x.reshape(-1, self.n_neurons * self.n_fr_bins)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(2*max_len) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0)]


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, n_neurons, n_fr_bins, max_trial_length, latent_dim=None):
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dim is None:
            self.spike_encoder = None
            self.spike_decoder = None
        else:
            self.spike_encoder = SpikeEncoder(n_neurons, n_fr_bins, latent_dim)
            self.spike_decoder = SpikeDecoder(n_neurons, n_fr_bins, latent_dim)

        self.input_projection = nn.Linear(
            input_size, d_model) if latent_dim is None else self.spike_encoder
        self.pos_encoder = PositionalEncoding(d_model, max_trial_length)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model if latent_dim is None else latent_dim,
                dim_feedforward=4*d_model if latent_dim is None else 4*latent_dim,
                nhead=8,
                batch_first=True,
                norm_first=True,
                dropout=0.2
            ),
            num_layers=4,
            enable_nested_tensor=False
        )
        self.output_projection = nn.Linear(
            d_model, n_neurons * n_fr_bins) if latent_dim is None else self.spike_decoder
        self.unflatten = nn.Unflatten(2, (n_neurons, n_fr_bins))
        self.register_buffer('causal_mask',
                             nn.Transformer.generate_square_subsequent_mask(max_trial_length))

    def forward(self, spikes, velocities):
        batch_size, n_context_bins, n_features = spikes.shape
        x = torch.cat((spikes, velocities.reshape(
            batch_size, n_context_bins, -1)), dim=2)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(
            x, mask=self.causal_mask[:n_context_bins, :n_context_bins])
        x = self.output_projection(x)
        x = self.unflatten(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_neurons, n_fr_bins, max_trial_length=None, latent_dim=None, num_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dim is None:
            self.spike_encoder = None
            self.spike_decoder = None
        else:
            self.spike_encoder = SpikeEncoder(n_neurons, n_fr_bins, latent_dim)
            self.spike_decoder = SpikeDecoder(n_neurons, n_fr_bins, latent_dim)

        # Input embedding layer
        self.input_projection = nn.Linear(
            input_size, hidden_size) if latent_dim is None else self.spike_encoder

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        self.output_projection = nn.Linear(
            hidden_size, n_neurons * n_fr_bins) if latent_dim is None else self.spike_decoder
        self.unflatten = nn.Unflatten(2, (n_neurons, n_fr_bins))

    def forward(self, spikes, velocities):
        batch_size, n_context_bins, n_features = spikes.shape
        x = torch.cat((spikes, velocities.reshape(
            batch_size, n_context_bins, -1)), dim=2)
        x = self.input_projection(x)
        x, _ = self.lstm(x)
        x = self.output_projection(x)
        x = self.unflatten(x)
        return x
