import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1)) # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class ConvEncoder(nn.Module):
    def __init__(self,
                 input_shape,
                 latent_size,
                 conv_layers=3,
                 filter_sizes=[9,9,10],
                 channels=[9,9,11],
                 fc_width=512):
        super().__init__()

        final_dense_width = (input_shape[-1] - (filter_sizes[0] - 1) - (filter_sizes[1] - 1) - (filter_sizes[2] - 1)) * channels[-1]
        self.conv1 = nn.Conv1d(input_shape[0], channels[0], filter_sizes[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], filter_sizes[1])
        self.conv3 = nn.Conv1d(channels[1], channels[2], filter_sizes[2])
        self.dense = nn.Linear(final_dense_width, fc_width)
        self.bn = nn.BatchNorm1d(fc_width)
        self.z_means = nn.Linear(fc_width, latent_size)
        self.z_vars = nn.Linear(fc_width, latent_size)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous().view(x.size(0), -1)
        x = self.bn(F.relu(self.dense(x)))
        mu, logvar = self.z_means(x), self.z_vars(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu * eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class GRUDecoder(nn.Module):
    def __init__(self,
                 input_shape,
                 latent_size,
                 gru_layers=3,
                 gru_size=488,
                 drop_prob=0.2):
        super().__init__()

        self.repeat = input_shape[1]
        self.hidden_dim = gru_size
        self.n_layers = gru_layers
        self.gru = nn.GRU(latent_size, gru_size, gru_layers, dropout=drop_prob)
        self.decode = nn.Linear(gru_size, input_shape[0])
        self.bn = nn.BatchNorm1d(input_shape[0])
        self.log_sm = nn.LogSoftmax(1)

    def forward(self, x, h):
        x = x.unsqueeze(0).repeat(self.repeat, 1, 1)
        x, h = self.gru(x, h)
        h = h.detach()
        x = self.decode(x)
        x = self.bn(x.permute(1, 2, 0))
        # x = self.log_sm(x)
        # x = F.softmax(x, dim=1)
        return x, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new_zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

class GenerativeVAE(nn.Module):
    def __init__(self,
                 input_shape,
                 latent_size):
        super().__init__()

        self.encoder = ConvEncoder(input_shape, latent_size)
        self.decoder = GRUDecoder(input_shape, latent_size)

    def forward(self, x, h):
        z, mu, logvar = self.encoder(x)
        x_decode, h = self.decoder(z, h)
        return x_decode, mu, logvar, h
