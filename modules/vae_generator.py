import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import util.util as uu
from util.pred_blocks import GenerativeVAE
from util.losses import vae_loss

class PlastVAEGen():
    def __init__(self, params, verbose=False):
        self.verbose = verbose
        self.params = {}
        for p, v in params.items():
            self.params[p] = v
        if 'LATENT_SIZE' in self.params.keys():
            self.latent_size = self.params['LATENT_SIZE']
        else:
            self.latent_size = 292
        self.history = {'train_loss': [],
                        'val_loss': []}
        self.best_loss = np.inf
        self.n_epochs = 0
        self.current_state = {'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'input_shape': None,
                              'latent_size': None,
                              'history': self.history}
        self.best_state = {'epoch': self.n_epochs,
                           'model_state_dict': None,
                           'optimizer_state_dict': None,
                           'best_loss': self.best_loss,
                           'input_shape': None,
                           'latent_size': None,
                           'history': self.history}
        self.trained = False

    def save(self, state, fn, path='checkpoints'):
        os.makedirs(path, exist_ok=True)
        if os.path.splitext(fn)[1] == '':
            save_fn += '.ckpt'
        torch.save(state, os.path.join(path, fn))

    def load(self, checkpoint_path):
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        for key in self.current_state.keys():
            self.current_state[key] = loaded_checkpoint[key]

        self.history = self.current_state['history']
        self.n_epochs = self.current_state['epoch']
        self.best_loss = self.current_state['best_loss']
        self.network = GenerativeVAE(self.current_state['input_shape'], self.current_state['latent_size'])
        self.network.load_state_dict(self.current_state['model_state_dict'])
        self.trained = True

    def initiate(self, data):
        """
        This function not only loads data, but also builds the network
        architecture. This must be done after the data is loaded because
        the number of input channels is dependent on the max length of smiles
        strings and number of unique characters.
        """
        # Setting up parameters
        self.all_smiles = data[:,0]
        self.all_lls = data[:,1]
        self.data_length = max(map(len, data[:,0]))
        if 'MAX_LENGTH' in self.params.keys():
            self.max_length = self.params['MAX_LENGTH']
        else:
            self.max_length = int(self.data_length * 1.5)
        if 'TRAIN_SPLIT' in self.params.keys():
            self.train_split = self.params['TRAIN_SPLIT']
        else:
            self.train_split = 0.8

        # One-hot encoding smiles below the max length
        self.usable_data = [(ll, sm) for ll, sm in zip(self.all_lls, self.all_smiles) if len(sm) < self.max_length]
        self.usable_lls = np.array([x[0] for x in self.usable_data])
        self.usable_smiles = [x[1] for x in self.usable_data]
        self.char_dict, self.ord_dict = uu.get_smiles_vocab(self.usable_smiles)
        self.num_char = len(self.char_dict)
        self.pad_num = self.char_dict['_']
        self.encoded = torch.empty((len(self.usable_smiles), self.num_char, self.max_length))
        for i, sm in enumerate(self.usable_smiles):
            self.encoded[i,:,:] = torch.tensor(uu.encode_smiles(sm, self.max_length, self.char_dict))
        self.input_shape = (self.num_char, self.max_length)

        # Data preparation
        self.n_samples = self.encoded.shape[0]
        self.n_train = int(self.n_samples * self.train_split)
        self.n_test = self.n_samples - self.n_train
        rand_idxs = np.random.choice(np.arange(self.n_samples), size=self.n_samples)
        train_idxs = rand_idxs[:self.n_train]
        val_idxs = rand_idxs[self.n_train:]

        self.X_train = self.encoded[train_idxs,:,:]
        self.X_val = self.encoded[val_idxs,:,:]
        self.y_train = self.usable_lls[train_idxs]
        self.y_val = self.usable_lls[val_idxs]

        # Build network
        if self.trained:
            assert self.input_shape == self.current_state['input_shape'], "ERROR - Shape of data different than that used to train loaded model"
            assert self.latent_size == self.current_state['latent_size'], "ERROR - Latent space of trained model unequal to input parameter"
        else:
            self.network = GenerativeVAE(self.input_shape, self.latent_size)

        # Update state dictionaries
        self.current_state['input_shape'] = self.input_shape
        self.best_state['input_shape'] = self.input_shape
        self.current_state['latent_size'] = self.latent_size
        self.best_state['latent_size'] = self.latent_size

    def train(self, save_last=True, save_best=True, log=True):
        """
        Function to train model with loaded data
        """
        # Setting up parameters
        if 'BATCH_SIZE' in self.params.keys():
            self.batch_size = self.params['BATCH_SIZE']
        else:
            self.batch_size = 10
        if 'LEARNING_RATE' in self.params.keys():
            self.lr = self.params['LEARNING_RATE']
        else:
            self.lr = 1e-4
        if 'N_EPOCHS' in self.params.keys():
            epochs = self.params['N_EPOCHS']
        else:
            epochs = 100
        torch.backends.cudnn.benchmark = True
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.network.cuda()

        # Create data iterables
        train_loader = torch.utils.data.DataLoader(self.X_train,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=False)
        val_loader = torch.utils.data.DataLoader(self.X_val,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 pin_memory=False)


        if self.trained:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
            self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Set up logger
        if log:
            log_file = open('log.txt', 'a')
            log_file.write('epoch,batch_idx,data_type,tot_loss,bce_loss,kld_loss\n')
            log_file.close()

        # Epoch Looper
        for epoch in range(epochs):
            if self.verbose:
                epoch_counter = '[{}/{}]'.format(epoch+1, epochs)
                progress_bar = '['+'-'*50+']'
                sys.stdout.write("\r\033[K"+epoch_counter+' '+progress_bar)

            # Train Loop
            self.network.train()
            losses = []
            for batch_idx, data in enumerate(train_loader):
                if use_gpu:
                    data = data.cuda()

                x = torch.autograd.Variable(data)
                x_decode, mu, logvar = self.network(x)
                loss, bce, kld = vae_loss(x, x_decode, mu, logvar, self.max_length)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                losses.append(loss.item())
                if log:
                    log_file = open('log.txt', 'a')
                    log_file.write('{},{},{},{},{},{}\n'.format(epoch,batch_idx,'train',loss.item(),bce.item(),kld.item()))
                    log_file.close()
                # print('{},{},{},{},{},{}\n'.format(epoch,batch_idx,'train',loss.item(),bce.item(),kld.item()))

            train_loss = np.mean(losses)
            self.history['train_loss'].append(train_loss)
            self.n_epochs += 1

            # Val Loop
            self.network.eval()
            losses = []
            for batch_idx, data in enumerate(val_loader):
                if use_gpu:
                    data = data.cuda()

                x = torch.autograd.Variable(data)
                x_decode, mu, logvar = self.network(x)
                loss, bce, kld = vae_loss(x, x_decode, mu, logvar, self.max_length)
                losses.append(loss.item())
                if log:
                    log_file = open('log.txt', 'a')
                    log_file.write('{},{},{},{},{},{}\n'.format(epoch,batch_idx,'test',loss.item(),bce.item(),kld.item()))
                    log_file.close()
                # print('{},{},{},{},{},{}\n'.format(epoch,batch_idx,'test',loss.item(),bce.item(),kld.item()))
            val_loss = np.mean(losses)
            self.history['val_loss'].append(val_loss)
            print('Epoch - {}  Train Loss - {}  Val Loss - {}'.format(self.n_epochs,
                                                                      round(train_loss, 2),
                                                                      round(val_loss, 2)))

            if save_best:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_state['epoch'] = self.n_epochs
                    self.best_state['model_state_dict'] = self.network.state_dict()
                    self.best_state['optimizer_state_dict'] = self.optimizer.state_dict()
                    self.best_state['best_loss'] = self.best_loss
                    self.best_state['history'] = self.history
                    self.save(self.best_state, 'best.ckpt')
            self.current_state['epoch'] = self.n_epochs
            self.current_state['model_state_dict'] = self.network.state_dict()
            self.current_state['optimizer_state_dict'] = self.optimizer.state_dict()
            self.current_state['best_loss'] = self.best_loss
            self.current_state['history'] = self.history
            self.best_state['history'] = self.history
        if save_last:
            self.save(self.current_state, 'latest.ckpt')
            self.save(self.best_state, 'best.ckpt')

    def predict(self, data):
        """
        Predicts output given a set of input data (model must already be trained)
        """
        self.network.eval()
        x = torch.autograd.Variable(torch.from_numpy(data))
        x_decode, mu, logvar = self.network(x)
        return x_decode.cpu().detach().numpy()
