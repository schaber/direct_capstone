import torch
import pandas as pd
import numpy as np
from modules.util.util import *
from modules.vae_generator import PlastVAEGen

pl_ll = pd.read_pickle('database/pl_likelihoods_v1.pkl')
org_ll = pd.read_pickle('database/org_likelihoods_v1.pkl')

all_data = pd.concat([pl_ll, org_ll]).to_numpy()
params = {'MAX_LENGTH': 120}

pvg = PlastVAEGen(params=params)
pvg.initiate(test_data)
pvg.train()
