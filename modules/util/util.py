import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, SGDClassifier
from sklearn import model_selection
import pubchempy as pcp

# SMILES Helper Functions
def get_smiles_dicts(smiles):
    i = 0
    char_dict, ord_dict = {'^':i}, {i:'^'}
    for smile in smiles:
        for c in smile:
            if c not in char_dict.keys():
                i += 1
                char_dict[c] = i
                ord_dict[i] = c
    char_dict['_'] = i+1
    ord_dict[i+1] = '_'
    return char_dict, ord_dict

### KDE Funcs
def calc_kde(xis, lo, hi, h=0.17):
    kde = []
    xs = np.linspace(lo, hi, 101)
    for x in xs:
        kde_sum = 0
        for xi in xis:
            x_prime = (x - xi) / h
            kde_sum += 1 / (1 * math.sqrt(2*math.pi))*math.exp(-(1/2)*((x_prime - 0) / 1)**2)
        kde_sum /= len(xis)*h
        kde.append(kde_sum)
    return kde, xs

def get_kde_value(v, kde, xs):
    for i, x in enumerate(xs):
        if v < x:
            idx = i
            break
    val = kde[i]
    return val

def eval_acc(vs, boundary):
    hits = 0
    for v in vs:
        if v > boundary:
            hits += 1
        else:
            pass
    acc = hits / len(vs)
    return acc

def calc_2D_kde(x_vec, x_range, y_range, gridsize=101):
    # Using Scott's rule of thumb
    d = 2
    std_i = np.std(x_vec[:,0])
    std_j = np.std(x_vec[:,1])
    n = x_vec.shape[0]
    H_i = (n**(-1 / (d + 4))*std_i)**2
    H_j = (n**(-1 / (d + 4))*std_j)**2
    H = np.array([[H_i, 0], [0, H_j]])

    xs = np.linspace(x_range[0], x_range[1], gridsize)
    ys = np.linspace(y_range[0], y_range[1], gridsize)
    kde = np.zeros((gridsize, gridsize))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            kde_sum = 0
            for k in range(x_vec.shape[0]):
                xi = x_vec[k,:]
                x_prime = np.array([x, y]) - xi
                val = (2*math.pi)**(-d/2)*(np.linalg.det(H))**(-1/2)*math.exp(-(1/2)*x_prime.T@np.linalg.inv(H)@x_prime)
                kde_sum += val
            kde_sum /= n
            kde[i,j] = kde_sum
    return kde.T, xs, ys

def get_2D_kde_value(v, kde, xs, ys):
    idxs = []
    for i, x in enumerate(xs):
        if v[0] < x:
            idxs.append(i)
            break
    for j, y in enumerate(ys):
        if v[1] < y:
            idxs.append(j)
            break
    val = kde.T[idxs[0], idxs[1]]
    return val


### Analysis Helper Funcs
def pca_data_split(plasticizers, other, n, test_train_split=0.8, cols='default'):
    # important data columns
    if isinstance(cols, str):
        cols = np.array(plasticizers.columns[1:])
    else:
        cols = cols

    # Plasticizers
    pl_smiles = plasticizers['SMILES'].to_numpy()
    pl_data = plasticizers[cols].to_numpy()

    rand_idxs = np.random.choice(np.arange(len(pl_data)), size=len(pl_data), replace=False)
    n_train = int(test_train_split*len(pl_data))
    train_idxs = rand_idxs[:n_train]
    test_idxs = rand_idxs[n_train:]
    pl_tr = pl_data[train_idxs,:]
    pl_smiles_tr = pl_smiles[train_idxs]
    pl_te = pl_data[test_idxs,:]
    pl_smiles_te = pl_smiles[test_idxs]

    # Comparison Molecules
    oth_all = other.sample(n=n)
    oth_smiles = oth_all['SMILES'].to_numpy()
    oth_data = oth_all[cols].to_numpy()

    rand_idxs = np.random.choice(np.arange(len(oth_data)), size=len(oth_data), replace=False)
    train_idxs = rand_idxs[:n_train]
    test_idxs = rand_idxs[n_train:]
    oth_tr = oth_data[train_idxs,:]
    oth_te = oth_data[test_idxs,:]
    oth_smiles_tr = oth_smiles[train_idxs]
    oth_smiles_te = oth_smiles[test_idxs]

    return [pl_tr, pl_te, pl_smiles_tr, pl_smiles_te], [oth_tr, oth_te, oth_smiles_tr, oth_smiles_te]

def run_pca(pl_sets, other_sets, n_components=2, scaling_factor=0.2):
    n_pl_tr = len(pl_sets[0])
    n_pl_te = len(pl_sets[1])
    train_data = np.concatenate([pl_sets[0], other_sets[0]], axis=0)
    test_data = np.concatenate([pl_sets[1], other_sets[1]], axis=0)

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    train_data[:,145:] *= scaling_factor
    test_data[:,145:] *= scaling_factor

    pca = PCA(n_components=n_components)
    train_ics = pca.fit_transform(train_data)
    test_ics = pca.transform(test_data)

    pl_ics_tr = train_ics[:n_pl_tr,:]
    oth_ics_tr = train_ics[n_pl_tr:,:]
    pl_ics_te = test_ics[:n_pl_te,:]
    oth_ics_te = test_ics[n_pl_te:,:]

    return [pl_ics_tr, pl_ics_te], [oth_ics_tr, oth_ics_te], pca, scaler

# Polarity parameter calculations
def calc_num_polar(row):
    num_polar = 0
    polar_groups = ['fr_C_O', 'fr_Al_OH', 'fr_phos_acid', 'fr_para_hydroxylation', \
                    'fr_sulfonamd', 'fr_halogen', 'fr_Ar_OH']
    n_groups = ['fr_ArN', 'fr_Ar_N', 'fr_Ar_NH', 'fr_HOCCN', 'fr_NH0', 'fr_NH1', \
                'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', \
                'fr_Nhpyrrole', 'fr_amide', 'fr_quatN','fr_urea']

    for group in polar_groups:
        num_polar += row[group]

    num_polar += max([0, row['fr_ether'] - row['fr_C_O']])
    for l in row['SMILES']:
        if l == '+':
            num_polar += 1
    n_vals = []
    for group in n_groups:
        n_vals.append(row[group])
    num_polar += max(n_vals)
    return num_polar

def calc_carbon(row):
    num_carbon = 0
    for l in row['SMILES']:
        if l == 'C':
            num_carbon += 1

    num_carbon -= row['fr_C_O']
    num_carbon -= row['fr_benzene']*6
    return num_carbon

def calc_polarity(row):
    if row['Num Carbon'] != 0:
        polarity = (row['MolWt']*(row['Num Polar'] / row['Num Carbon'])) / 1000
    else:
        polarity = 0
    return polarity
