#!/usr/bin/env python3
# encoding: utf-8

import os, pickle
import numpy as np
import scipy.spatial
import Bio.PDB
from Bio.PDB.DSSP import DSSP

# {G,H,I: H}, {S,T,C: C}, {B,E: E}
SS3_TYPES = {'H':0, 'B':2, 'E':2, 'G':0, 'I':0, 'T':1, 'S':1, '-':1}
RESIDUE_TYPES = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
pdb_parser = Bio.PDB.PDBParser(QUIET = True)

def get_struc_feat(seq_len, model_file):
    '''
    Agrs:
        seq_len (int): sequence length.
        model_file (string): the path of model file.
    '''
    feature = {}

    structure = pdb_parser.get_structure("tmp_stru", model_file)
    model = structure.get_list()[0]
    residues = model.get_list()[0].get_list()

    # SS and RSA
    dssp = DSSP(model, model_file, dssp='dssp')
    SS3s, RSAs = np.zeros((seq_len, 3)), np.zeros((seq_len, 1))
    for _key in dssp.keys():
        res_index = _key[1][1]
        if res_index >= 1 and res_index <= seq_len:
            SS3s[res_index-1, SS3_TYPES[ dssp[_key][2]]] = 1
            RSAs[res_index-1] = [dssp[_key][3]]
    feature['SS3'] = SS3s
    feature['RSA'] = RSAs

    atom_types = ['CA', 'CB', 'N', 'O']
    # generate empty coordinates
    coordinates = []
    for _ in range(seq_len):
        _dict = {}
        for atom_type in atom_types: _dict[atom_type] = None
        coordinates.append(_dict)

    # extract coordinates from pdb
    for res in residues:
        for atom in res:
            if atom.name in atom_types: coordinates[res.id[1]-1][atom.name] = atom.coord
        # copy CA coordinate to CB if CB is None (GLY)
        if 'CB' in atom_types and coordinates[res.id[1]-1]['CB'] is None:
            coordinates[res.id[1]-1]['CB'] = coordinates[res.id[1]-1]['CA']
    
    # distance map
    atom_pairs=['CaCa', 'CbCb', 'NO']
    for atom_pair in atom_pairs:
        atom1, atom2 = atom_pair[:int(len(atom_pair)/2)].upper(), atom_pair[int(len(atom_pair)/2):].upper()
        X = [ list(c[atom1]) if (c is not None and c[atom1] is not None) else [0,0,0] for c in coordinates ]
        X_valid = [ 0 if (c is None or c[atom1] is None) else 1 for c in coordinates ]
        Y = [ list(c[atom2]) if (c is not None and c[atom2] is not None) else [0,0,0] for c in coordinates ]
        Y_valid = [ 0 if (c is None or c[atom2] is None) else 1 for c in coordinates ]
        dist = scipy.spatial.distance_matrix(X, Y).astype(np.float16)
        XY_valid = np.outer(X_valid, Y_valid)
        np.putmask(dist, XY_valid==0, -1)
        if atom1 == atom2: np.fill_diagonal(dist, 0) # set the self distance to 0

        feature[atom_pair] = dist.reshape((dist.shape[0], dist.shape[1], -1))*0.1

    return feature

def get_seq_onehot(seq):
    seq_onehot = np.zeros((len(seq), len(RESIDUE_TYPES)))
    for i, res in enumerate(seq.upper()):
        if res not in RESIDUE_TYPES: res = "X"
        seq_onehot[i, RESIDUE_TYPES.index(res)] = 1
    return seq_onehot

def get_rPos(seq_len):
    r_pos = np.linspace(0, 1, num=seq_len).reshape(seq_len, -1)
    return r_pos