import os
import sys
import shutil
import imageio
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import util.util as uu
from util.pred_blocks import GenerativeVAE, GenerativeVAE_v2
from util.losses import vae_bce_loss, vae_ce_loss

class PlastVAEGen():
    def __init__(self, params={}, verbose=False):
        self.verbose = verbose
        self.params = {}
        for p, v in params.items():
            self.params[p] = v
        if 'LATENT_SIZE' in self.params.keys():
            self.latent_size = self.params['LATENT_SIZE']
        else:
            self.latent_size = 512
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
                              'history': self.history,
                              'params': self.params}
        self.best_state = {'epoch': self.n_epochs,
                           'model_state_dict': None,
                           'optimizer_state_dict': None,
                           'best_loss': self.best_loss,
                           'input_shape': None,
                           'latent_size': None,
                           'history': self.history,
                           'params': self.params}
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
        self.params = self.current_state['params']
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
        self.params['DATA_LENGTH'] = max(map(len, data[:,0]))
        if 'MAX_LENGTH' not in self.params.keys():
            self.params['MAX_LENGTH'] = int(self.params['DATA_LENGTH'] * 1.5)
        if 'TRAIN_SPLIT' not in self.params.keys():
            self.params['TRAIN_SPLIT'] = 0.8

        # One-hot encoding smiles below the max length
        self.usable_data = [(ll, sm) for ll, sm in zip(self.all_lls, self.all_smiles) if len(sm) < self.params['MAX_LENGTH']]
        self.usable_lls = np.array([x[0] for x in self.usable_data])
        self.usable_smiles = [x[1] for x in self.usable_data]
        self.params['CHAR_DICT'], self.params['ORD_DICT'] = uu.get_smiles_vocab(self.usable_smiles)
        self.params['NUM_CHAR'] = len(self.params['CHAR_DICT'])
        self.params['PAD_NUM'] = self.params['CHAR_DICT']['_']
        self.encoded = torch.empty((len(self.usable_smiles), self.params['NUM_CHAR'], self.params['MAX_LENGTH']))
        for i, sm in enumerate(self.usable_smiles):
            self.encoded[i,:,:] = torch.tensor(uu.encode_smiles(sm, self.params['MAX_LENGTH'], self.params['CHAR_DICT']))
        self.input_shape = (self.params['NUM_CHAR'], self.params['MAX_LENGTH'])

        # Data preparation
        self.params['N_SAMPLES'] = self.encoded.shape[0]
        self.params['N_TRAIN'] = int(self.params['N_SAMPLES'] * self.params['TRAIN_SPLIT'])
        self.params['N_TEST'] = self.params['N_SAMPLES'] - self.params['N_TRAIN']
        self.rand_idxs = np.random.choice(np.arange(self.params['N_SAMPLES']), size=self.params['N_SAMPLES'])
        self.params['TRAIN_IDXS'] = self.rand_idxs[:self.params['N_TRAIN']]
        self.params['VAL_IDXS'] = self.rand_idxs[self.params['N_TRAIN']:]

        self.X_train = self.encoded[self.params['TRAIN_IDXS'],:,:]
        self.X_val = self.encoded[self.params['VAL_IDXS'],:,:]
        self.y_train = self.usable_lls[self.params['TRAIN_IDXS']]
        self.y_val = self.usable_lls[self.params['VAL_IDXS']]

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

    def trained_initiate(self, data):
        """
        Function analogous to `self.initiate` except some parameters do not need to be
        re-initialized because model has already been trained for some number of
        epochs (you must use the same data that model was initially trained on)
        """
        # Setting up parameters
        self.all_smiles = data[:,0]
        self.all_lls = data[:,1]

        # One-hot encoding smiles below the max length
        self.usable_data = [(ll, sm) for ll, sm in zip(self.all_lls, self.all_smiles) if len(sm) < self.params['MAX_LENGTH']]
        self.usable_lls = np.array([x[0] for x in self.usable_data])
        self.usable_smiles = [x[1] for x in self.usable_data]
        self.encoded = torch.empty((len(self.usable_smiles), self.params['NUM_CHAR'], self.params['MAX_LENGTH']))
        for i, sm in enumerate(self.usable_smiles):
            self.encoded[i,:,:] = torch.tensor(uu.encode_smiles(sm, self.params['MAX_LENGTH'], self.params['CHAR_DICT']))
        self.input_shape = (self.params['NUM_CHAR'], self.params['MAX_LENGTH'])

        # Data preparation
        self.X_train = self.encoded[self.params['TRAIN_IDXS'],:,:]
        self.X_val = self.encoded[self.params['VAL_IDXS'],:,:]
        self.y_train = self.usable_lls[self.params['TRAIN_IDXS']]
        self.y_val = self.usable_lls[self.params['VAL_IDXS']]

    def train(self, data, save_last=True, save_best=True, log=True, make_grad_gif=False):
        """
        Function to train model with loaded data
        """
        # Setting up parameters
        if 'BATCH_SIZE' in self.params.keys():
            self.batch_size = self.params['BATCH_SIZE']
        else:
            self.batch_size = 1000
        if 'LEARNING_RATE' in self.params.keys():
            self.lr = self.params['LEARNING_RATE']
        else:
            self.lr = 1e-4
        if 'N_EPOCHS' in self.params.keys():
            epochs = self.params['N_EPOCHS']
        else:
            epochs = 100

        if not self.trained:
            self.initiate(data)
        elif self.trained:
            self.trained_initiate(data)

        torch.backends.cudnn.benchmark = True
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.network.cuda()
        if make_grad_gif:
            os.mkdir('gif')
            images = []
            frame = 0

        # Save constant params to state dicts
        if save_best:
            self.best_state['params'] = self.params
        if save_last:
            self.current_state['params'] = self.params

        # Create data iterables
        train_loader = torch.utils.data.DataLoader(self.X_train,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=False,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(self.X_val,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 pin_memory=False,
                                                 drop_last=True)


        if self.trained:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
            self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Set up logger
        if log and not self.trained:
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
            h = self.network.decoder.init_hidden(self.params['BATCH_SIZE']).data
            losses = []
            for batch_idx, data in enumerate(train_loader):
                self.network.zero_grad()
                if use_gpu:
                    data = data.cuda()

                x = torch.autograd.Variable(data)
                x_decode, mu, logvar, h = self.network(x, h.data)
                loss, bce, kld = vae_bce_loss(x, x_decode, mu, logvar, self.params['MAX_LENGTH'])
                if batch_idx < 1:
                    self.sample = x
                    self.out = x_decode
                    self.sample_loss = loss.item()
                    self.mu = mu
                    self.logvar = logvar
                loss.backward()
                if make_grad_gif:
                    plt = uu.plot_grad_flow(self.network.named_parameters())
                    plt.title('Epoch {}  Frame {}'.format(epoch, frame))
                    fn = 'gif/{}.png'.format(frame)
                    plt.savefig(fn)
                    plt.close()
                    images.append(imageio.imread(fn))
                    frame += 1
                self.optimizer.step()

                losses.append(loss.item())
                if log:
                    log_file = open('log.txt', 'a')
                    log_file.write('{},{},{},{},{},{}\n'.format(self.n_epochs,batch_idx,'train',loss.item(),bce.item(),kld.item()))
                    log_file.close()
                # print('{},{},{},{},{},{}\n'.format(epoch,batch_idx,'train',loss.item(),bce.item(),kld.item()))
            train_loss = np.mean(losses)
            self.history['train_loss'].append(train_loss)

            # Val Loop
            self.network.eval()
            h = self.network.decoder.init_hidden(self.params['BATCH_SIZE']).data
            losses = []
            for batch_idx, data in enumerate(val_loader):
                if use_gpu:
                    data = data.cuda()

                x = torch.autograd.Variable(data)
                x_decode, mu, logvar, h = self.network(x, h.data)
                loss, bce, kld = vae_bce_loss(x, x_decode, mu, logvar, self.params['MAX_LENGTH'])
                losses.append(loss.item())
                if log:
                    log_file = open('log.txt', 'a')
                    log_file.write('{},{},{},{},{},{}\n'.format(self.n_epochs,batch_idx,'test',loss.item(),bce.item(),kld.item()))
                    log_file.close()
                # print('{},{},{},{},{},{}\n'.format(epoch,batch_idx,'test',loss.item(),bce.item(),kld.item()))
            val_loss = np.mean(losses)
            self.history['val_loss'].append(val_loss)
            print('Epoch - {}  Train Loss - {}  Val Loss - {}'.format(self.n_epochs,
                                                                      round(train_loss, 2),
                                                                      round(val_loss, 2)))
            self.n_epochs += 1

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
        self.trained = True
        if make_grad_gif:
            imageio.mimsave('grads4.gif', images)
            shutil.rmtree('gif')

    def predict(self, data):
        """
        Predicts output given a set of input data (model must already be trained)
        """
        self.network.eval()
        h = self.network.decoder.init_hidden(data.shape[0])
        x = torch.autograd.Variable(torch.from_numpy(data))
        x_decode, mu, logvar, h = self.network(x, h.data)
        x_decode = F.softmax(x_decode, dim=1)
        return x_decode.cpu().detach().numpy()


class PlastVAEGen_v2():
    def __init__(self, params={}, name=None, verbose=False):
        self.verbose = verbose
        self.params = {}
        self.name = name
        for p, v in params.items():
            self.params[p] = v
        if 'LATENT_SIZE' in self.params.keys():
            self.latent_size = self.params['LATENT_SIZE']
        else:
            self.latent_size = 56
        if 'KL_BETA' not in self.params.keys():
            self.params['KL_BETA'] = 1.0
        self.history = {'train_loss': [],
                        'val_loss': []}
        self.best_loss = np.inf
        self.n_epochs = 0
        self.current_state = {'name': self.name,
                              'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'latent_size': None,
                              'history': self.history,
                              'params': self.params}
        self.best_state = {'name': self.name,
                           'epoch': self.n_epochs,
                           'model_state_dict': None,
                           'optimizer_state_dict': None,
                           'best_loss': self.best_loss,
                           'latent_size': None,
                           'history': self.history,
                           'params': self.params}
        self.trained = False

    def save(self, state, fn, path='checkpoints'):
        os.makedirs(path, exist_ok=True)
        if os.path.splitext(fn)[1] == '':
            if self.name is not None:
                fn += '_' + self.name
            fn += '.ckpt'
        else:
            if self.name is not None:
                fn, ext = fn.split('.')
                fn += '_' + self.name
                fn = '.'.join(fn, ext)
        torch.save(state, os.path.join(path, fn))

    def load(self, checkpoint_path):
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        for key in self.current_state.keys():
            self.current_state[key] = loaded_checkpoint[key]

        self.name = self.current_state['name']
        self.history = self.current_state['history']
        self.n_epochs = self.current_state['epoch']
        self.best_loss = self.current_state['best_loss']
        self.params = self.current_state['params']
        self.network = GenerativeVAE_v2(self.current_state['input_shape'], self.current_state['latent_size'])
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
        self.params['DATA_LENGTH'] = max(map(len, data[:,0]))
        if 'MAX_LENGTH' not in self.params.keys():
            self.params['MAX_LENGTH'] = int(self.params['DATA_LENGTH'] * 1.5)
        if 'TRAIN_SPLIT' not in self.params.keys():
            self.params['TRAIN_SPLIT'] = 0.8
        if 'EMBED_DIM' not in self.params.keys():
            self.params['EMBED_DIM'] = 48

        # Vectorizing smiles below max length
        self.usable_data = [(ll, sm) for ll, sm in zip(self.all_lls, self.all_smiles) if len(sm) < self.params['MAX_LENGTH']]
        self.usable_lls = np.array([x[0] for x in self.usable_data])
        self.usable_smiles = [x[1] for x in self.usable_data]
        self.params['CHAR_DICT'], self.params['ORD_DICT'] = uu.get_smiles_vocab(self.usable_smiles)
        self.params['NUM_CHAR'] = len(self.params['CHAR_DICT'])
        self.params['PAD_NUM'] = self.params['CHAR_DICT']['_']
        self.encoded = torch.empty((len(self.usable_smiles), self.params['MAX_LENGTH'] + 1), dtype=torch.long)
        for i, sm in enumerate(self.usable_smiles):
            self.encoded[i,:] = torch.tensor(uu.encode_smiles(sm, self.params['MAX_LENGTH'], self.params['CHAR_DICT'])).long()

        # Data preparation
        self.params['N_SAMPLES'] = self.encoded.shape[0]
        self.params['N_TRAIN'] = int(self.params['N_SAMPLES'] * self.params['TRAIN_SPLIT'])
        self.params['N_TEST'] = self.params['N_SAMPLES'] - self.params['N_TRAIN']
        self.rand_idxs = np.random.choice(np.arange(self.params['N_SAMPLES']), size=self.params['N_SAMPLES'])
        self.params['TRAIN_IDXS'] = self.rand_idxs[:self.params['N_TRAIN']]
        self.params['VAL_IDXS'] = self.rand_idxs[self.params['N_TRAIN']:]

        self.X_train = self.encoded[self.params['TRAIN_IDXS'],:]
        self.X_val = self.encoded[self.params['VAL_IDXS'],:]
        self.y_train = self.usable_lls[self.params['TRAIN_IDXS']]
        self.y_val = self.usable_lls[self.params['VAL_IDXS']]

        # Build network
        if self.trained:
            assert self.latent_size == self.current_state['latent_size'], "ERROR - Latent space of trained model unequal to input parameter"
        else:
            self.network = GenerativeVAE_v2(self.params['MAX_LENGTH'], self.params['NUM_CHAR'], self.params['EMBED_DIM'], self.latent_size)

        # Update state dictionaries
        self.current_state['latent_size'] = self.latent_size
        self.best_state['latent_size'] = self.latent_size

    def trained_initiate(self, data):
        """
        Function analogous to `self.initiate` except some parameters do not need to be
        re-initialized because model has already been trained for some number of
        epochs (you must use the same data that model was initially trained on)
        """
        # Setting up parameters
        self.all_smiles = data[:,0]
        self.all_lls = data[:,1]

        # One-hot encoding smiles below the max length
        self.usable_data = [(ll, sm) for ll, sm in zip(self.all_lls, self.all_smiles) if len(sm) < self.params['MAX_LENGTH']]
        self.usable_lls = np.array([x[0] for x in self.usable_data])
        self.usable_smiles = [x[1] for x in self.usable_data]
        self.encoded = torch.empty((len(self.usable_smiles), self.params['MAX_LENGTH'] + 1), dtype=torch.long)
        for i, sm in enumerate(self.usable_smiles):
            self.encoded[i,:] = torch.tensor(uu.encode_smiles(sm, self.params['MAX_LENGTH'], self.params['CHAR_DICT'])).long()

        # Data preparation
        self.X_train = self.encoded[self.params['TRAIN_IDXS'],:,:]
        self.X_val = self.encoded[self.params['VAL_IDXS'],:,:]
        self.y_train = self.usable_lls[self.params['TRAIN_IDXS']]
        self.y_val = self.usable_lls[self.params['VAL_IDXS']]

    def train(self, data, save_last=True, save_best=True, log=True, make_grad_gif=False):
        """
        Function to train model with loaded data
        """
        # Setting up parameters
        if 'BATCH_SIZE' in self.params.keys():
            self.batch_size = self.params['BATCH_SIZE']
        else:
            self.batch_size = 1000
        if 'LEARNING_RATE' in self.params.keys():
            self.lr = self.params['LEARNING_RATE']
        else:
            self.lr = 1e-4
        if 'N_EPOCHS' in self.params.keys():
            epochs = self.params['N_EPOCHS']
        else:
            epochs = 100

        if not self.trained:
            self.initiate(data)
        elif self.trained:
            self.trained_initiate(data)

        torch.backends.cudnn.benchmark = True
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.network.cuda()
        if make_grad_gif:
            os.mkdir('gif')
            images = []
            frame = 0

        # Save constant params to state dicts
        if save_best:
            self.best_state['params'] = self.params
        if save_last:
            self.current_state['params'] = self.params

        # Create data iterables
        train_loader = torch.utils.data.DataLoader(self.X_train,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=False,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(self.X_val,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 pin_memory=False,
                                                 drop_last=True)


        if self.trained:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
            self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Set up logger
        if log and not self.trained:
            os.makedirs('trials', exist_ok=True)
            if self.name is not None:
                log_file = open('trials/log{}.txt'.format('_'+self.name), 'a')
            else:
                log_file = open('trials/log.txt', 'a')
            log_file.write('epoch,batch_idx,data_type,tot_loss,bce_loss,kld_loss,naive_loss\n')
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
                self.network.zero_grad()
                if self.use_gpu:
                    data = data.cuda()

                x = torch.autograd.Variable(data)
                x_decode, mu, logvar, x_naive_decode = self.network(x, infer=True, self.params, self.use_gpu)
                loss, bce, kld = vae_ce_loss(x, x_decode, mu, logvar, self.params['MAX_LENGTH'], beta=self.params['KL_BETA'])
                _, naive_loss, _ = vae_ce_loss(x, x_naive_decode, mu, logvar, self.params['MAX_LENGTH', beta=self.params['KL_BETA']])
                loss.backward()
                if make_grad_gif:
                    plt = uu.plot_grad_flow(self.network.named_parameters())
                    plt.title('Epoch {}  Frame {}'.format(epoch, frame))
                    fn = 'gif/{}.png'.format(frame)
                    plt.savefig(fn)
                    plt.close()
                    images.append(imageio.imread(fn))
                    frame += 1
                self.optimizer.step()

                losses.append(loss.item())
                if log:
                    if self.name is not None:
                        log_file = open('trials/log{}.txt'.format('_'+self.name), 'a')
                    else:
                        log_file = open('trials/log.txt', 'a')
                    log_file.write('{},{},{},{},{},{},{}\n'.format(self.n_epochs,batch_idx,'train',loss.item(),bce.item(),kld.item(),naive_loss.item()))
                    log_file.close()
                # print('{},{},{},{},{},{}\n'.format(epoch,batch_idx,'train',loss.item(),bce.item(),kld.item()))
            train_loss = np.mean(losses)
            self.history['train_loss'].append(train_loss)

            # Val Loop
            self.network.eval()
            losses = []
            for batch_idx, data in enumerate(val_loader):
                if self.use_gpu:
                    data = data.cuda()

                x = torch.autograd.Variable(data)
                x_decode, mu, logvar, x_naive_decode = self.network(x, infer=True, self.params, self.use_gpu)
                loss, bce, kld = vae_ce_loss(x, x_decode, mu, logvar, self.params['MAX_LENGTH'], beta=self.params['KL_BETA'])
                _, naive_loss, _ = vae_ce_loss(x, x_naive_decode, mu, logvar, self.params['MAX_LENGTH', beta=self.params['KL_BETA']])
                losses.append(loss.item())
                if log:
                    if self.name is not None:
                        log_file = open('trials/log{}.txt'.format('_'+self.name), 'a')
                    else:
                        log_file = open('trials/log.txt', 'a')
                    log_file.write('{},{},{},{},{},{},{}\n'.format(self.n_epochs,batch_idx,'train',loss.item(),bce.item(),kld.item(),naive_loss.item()))
                    log_file.close()
                # print('{},{},{},{},{},{}\n'.format(epoch,batch_idx,'test',loss.item(),bce.item(),kld.item()))
            val_loss = np.mean(losses)
            self.history['val_loss'].append(val_loss)
            print('Epoch - {}  Train Loss - {}  Val Loss - {}'.format(self.n_epochs,
                                                                      round(train_loss, 2),
                                                                      round(val_loss, 2)))
            self.n_epochs += 1

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
            self.save(self.current_state, 'latest')
            self.save(self.best_state, 'best')
        self.trained = True
        if make_grad_gif:
            imageio.mimsave('grads4.gif', images)
            shutil.rmtree('gif')

    def teacher_predict(self, data):
        """
        Optimistic prediction of output given a set of input data (model must already be trained)
        """
        self.network.eval()
        h = self.network.decoder.init_hidden(data.shape[0])
        x = torch.autograd.Variable(torch.from_numpy(data))
        x_decode, mu, logvar = self.network(x)
        x_decode = F.softmax(x_decode, dim=1)
        return x_decode.cpu().detach().numpy()
