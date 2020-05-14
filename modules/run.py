import torch
import pandas as pd
import numpy as np
from util.util import *
from vae_generator import PlastVAEGen

pl_ll = pd.read_pickle('../database/pl_likelihoods_v1.pkl')
org_ll = pd.read_pickle('../database/org_likelihoods_v1.pkl')

all_data = pd.concat([pl_ll, org_ll]).to_numpy()
params = {'MAX_LENGTH': 120}

pvg = PlastVAEGen(params=params)
pvg.initiate(all_data)
pvg.train()
