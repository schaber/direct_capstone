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

class ConvEncoder_v2(nn.Module):
    def __init__(self,
                 max_len,
                 embed_dim,
                 latent_size,
                 conv_layers=3,
                 filter_sizes=[9,9,10],
                 channels=[9,9,11],
                 fc_width=168):
        super().__init__()

        final_dense_width = (max_len - (filter_sizes[0] - 1) - (filter_sizes[1] - 1) - (filter_sizes[2] - 1)) * channels[-1]
        self.conv1 = nn.Conv1d(embed_dim, channels[0], filter_sizes[0])
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
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class GRUDecoder_v2(nn.Module):
    def __init__(self,
                 max_len,
                 embed_dim,
                 vocab_size,
                 latent_size,
                 gru_layers=3,
                 gru_size=488,
                 drop_prob=0.2):
        super().__init__()

        self.hidden_dim = gru_size
        self.n_layers = gru_layers
        self.voc_size = vocab_size
        self.seq_len = max_len
        self.l2h = nn.Linear(latent_size, gru_size * gru_layers)
        self.gru = nn.GRU(embed_dim, gru_size, gru_layers, dropout=drop_prob, batch_first=True)
        self.decode = nn.Linear(gru_size, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.log_sm = nn.LogSoftmax(1)

    def forward(self, x, z):
        h = F.relu(self.l2h(z))
        h = h.view(-1, self.n_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()
        x = x.permute(0, 2, 1).contiguous()

        x, _ = self.gru(x, h)
        b, seq_len, hsize = x.size()
        x = x.contiguous().view(-1, hsize)
        x = self.decode(x)
        x = x.view(b, self.voc_size, seq_len)
        x = self.bn(x)
        return x

    def inference(self, z, embedding, params, use_gpu):
        max_len = params['MAX_LENGTH']
        batch_size = z.size(0)
        h = F.relu(self.l2h(z))
        h = h.view(-1, self.n_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()

        input_seq = torch.Tensor(batch_size).fill_(params['PAD_NUM']).unsqueeze(1).long()
        logits_t = torch.FloatTensor()
        if use_gpu:
            input_seq = input_seq.cuda()
            logits_t = logits_t.cuda()
        for t in range(max_len):
            input_embedding = embedding(input_seq)
            x, h = self.gru(input_embedding, h)
            logits = self.decode(x)
            logits_t = torch.cat((logits_t, logits), dim=1)
            input_seq = torch.argmax(logits, dim=-1)
        logits_t = logits_t.view(batch_size, self.voc_size, self.seq_len).contiguous()
        return logits_t


class GenerativeVAE_v2(nn.Module):
    def __init__(self,
                 max_len,
                 vocab_size,
                 embed_dim,
                 latent_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = ConvEncoder_v2(max_len, embed_dim, latent_size)
        self.decoder = GRUDecoder_v2(max_len, embed_dim, vocab_size, latent_size)

    def forward(self, x):
        x = x[:,:-1]
        x = self.embed(x)
        x = x.permute(0, 2, 1).contiguous()
        z, mu, logvar = self.encoder(x)
        x_decode = self.decoder(x, z)
        return x_decode, mu, logvar

    def inference(self, x, params, use_gpu):
        x = x[:,:-1]
        x = self.embed(x)
        x = x.permute(0, 2, 1).contiguous()
        z, mu, logvar = self.encoder(x)
        x_decode = self.decoder.inference(z, self.embed, params, use_gpu)
        return x_decode, mu, logvar
