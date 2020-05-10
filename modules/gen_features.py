"""
This module contains function to get the rdkit properties of molecules when provided with the smile string of the molecules. \
The module also adds the Bitvector object from the molecular fingerprints into the dataframe.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit import ML
from rdkit.ML.Descriptors import MoleculeDescriptors

def gen_features(smiles):
    """
    The function returns the dataframe for the molecules containing the rdkit properties of the molecules.

    Parameters:
        smiles: List of smile strings of different molecules.

    Returns:
        dataframe: Pandas dataframe with the rdkit feature properties.
    """
    descript_list = []
    for desc in Chem.Descriptors.descList:
        descript_list.append(desc[0])
    descript_calc = ML.Descriptors.MoleculeDescriptors.MolecularDescriptorCalculator(descript_list)
    descript_names = descript_calc.GetDescriptorNames()

    rdkit_features = np.zeros((len(smiles), len(descript_list)))
    for i, smile in enumerate(smiles):
        if i % 10 == 0:
            print("Progress: "+str(round(i/len(smiles)*100, 3))+'%')
        m = Chem.MolFromSmiles(smile)
        rdkit_features[i,:] = descript_calc.CalcDescriptors(m)

    mols = [Chem.MolFromSmiles(x) for x in smiles]
    fingerprints = [Chem.RDKFingerprint(x) for x in mols]

#     w2=[1,2,3]
#     rdkit_features['c'] = w2

    rdkit_features = pd.DataFrame(rdkit_features, columns=descript_names)
    rdkit_features['SMILES'] = smiles
    wr = [DataStructs.cDataStructs.BitVectToText(s) for s in fingerprints]
    rdkit_features['vecs'] = wr
#     rdkit_features = [rdkit_features[['SMILES']], rdkit_features.iloc[:, :-1]]
#     rdkit_features = pd.concat(rdkit_features, axis = 1)
    return rdkit_features
