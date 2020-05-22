# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class GRUEncoder(nn.Module):
    def __init__(self,
                 input_shape,
                 latent_size,
                 gru_layers=3,
                 gru_size=488,
                 drop_prob=0.,
                 bi_direc=False):
        super().__init__()

        self.num_char = input_shape[0]
        self.max_len = input_shape[1]
        self.hidden_dim = gru_size
        self.n_layers = gru_layers
        if bi_direc:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.gru = nn.GRU(input_shape[0], gru_size, gru_layers, dropout=drop_prob, bidirectional=bi_direc)
        self.conv1 = nn.Conv1d(self.num_directions*gru_size, self.num_directions*gru_size // 2, 9)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(self.num_directions*gru_size // 2, gru_size // 4, 9)
        self.conv3 = nn.Conv1d(gru_size // 4, 60, 9)
        self.z_means = nn.Linear(120, latent_size)
        self.z_var = nn.Linear(120, latent_size)

    def encode(self, x):
        x = x.permute(2, 0, 1)
        x, h = self.gru(x)
        h = h.detach()
        x = x.permute(1, 2, 0)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous().view(x.size(0), -1)
        mu, logvar = self.z_means(x), self.z_var(x)
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
                 gru_in_dim=48,
                 gru_layers=3,
                 gru_size=488,
                 drop_prob=0,
                 bi_direc=False):
        super().__init__()

        # self.repeat = input_shape[1]
        self.num_char = input_shape[0]
        self.max_len = input_shape[1]
        self.embed_len = gru_in_dim #
        self.hidden_dim = gru_size
        self.n_layers = gru_layers
        if bi_direc:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.dense = nn.Linear(latent_size, self.max_len*gru_in_dim) #
        self.gru = nn.GRU(gru_in_dim, gru_size, gru_layers, dropout=drop_prob, bidirectional=bi_direc) #
        self.decode = nn.Linear(self.num_directions*gru_size, self.num_char)
        self.bn = nn.BatchNorm1d(self.num_char)

    def forward(self, x):
        # x = x.unsqueeze(0).repeat(self.repeat, 1, 1)
        x = F.relu(self.dense(x)) #
        x = x.unsqueeze(1).view(x.size(0), self.embed_len, self.max_len).permute(2, 0, 1) #
        x, h = self.gru(x)
        h = h.detach()
        x = self.decode(x)
        x = self.bn(x.permute(1, 2, 0))
        return x

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new_zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

class ConvGRU(nn.Module):
    def __init__(self,
                 input_shape,
                 latent_size):
        super().__init__()

        self.encoder = ConvEncoder(input_shape, latent_size)
        self.decoder = GRUDecoder(input_shape, latent_size)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_decode = self.decoder(z)
        return x_decode, mu, logvar

class GRUGRU(nn.Module):
    def __init__(self,
                 input_shape,
                 latent_size,
                 enc_bi=False,
                 dec_bi=False):
        super().__init__()

        self.encoder = GRUEncoder(input_shape, latent_size, bi_direc=enc_bi)
        self.decoder = GRUDecoder(input_shape, latent_size, bi_direc=dec_bi)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_decode = self.decoder(z)
        return x_decode, mu, logvar
