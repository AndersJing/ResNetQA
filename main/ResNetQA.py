#!/usr/bin/env python3
# encoding: utf-8

from __future__ import print_function, division
import os, pickle, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Model import QAModel
import Utils as Utils

class ModelDataset(Dataset):
    def __init__(self, seq_feature, dist_potential, models_folder):
        """
        Args:
            seq_feature (string): sequence feature file.
            dist_potential (string): distance potential feature file.
            models_folder (str): folder of decoy models.
        """
        self.seq_feature = seq_feature
        self.dist_potential = dist_potential
        self.models_folder = models_folder
        self.seq_feat_types = {'onehot': '1D', 'PSSM': '1D', 'SS3': '1D', 'ACC': '1D', 'ccmpredZ': '2D', 'alnstats': '2D', 'DistPot': '2D'}
        self.struc_feat_types = {'SS3': '1D', 'RSA': '1D', 'CbCb': '2D', 'CaCa': '2D', 'NO': '2D'}

        self.models = []
        for _file in os.listdir(models_folder):
            if os.path.isfile(models_folder+_file): self.models.append(models_folder+_file)

    def __len__(self,):
        return len(self.models)

    def __get_seq_feat(self, ):
        input_features = pickle.load(open(self.seq_feature, 'rb'), encoding='latin1')
        dist_potential = pickle.load(open(self.dist_potential, 'rb'), encoding='latin1')[2]['CbCb_Discrete14C']

        seq_feat = {
            'sequence': input_features['sequence'],
            'PSSM': input_features['PSSM'],
            'SS3': input_features['SS3'],
            'ACC': input_features['ACC'],
            'ccmpredZ': input_features['ccmpredZ'],
            'alnstats': input_features['OtherPairs'],
            'DistPot': dist_potential,
        }

        return seq_feat

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        model_file = self.models[idx]
        feature = {"1D": None, "2D": None}

        # sequence feature
        seq_feat = self.__get_seq_feat()
        for _type in self.seq_feat_types:
            _shape = self.seq_feat_types[_type]
            _data = Utils.get_seq_onehot(seq_feat['sequence']) if _type=='onehot' else seq_feat[_type]
            if _shape=='2D': _data = _data.reshape((_data.shape[0], _data.shape[1], -1))
            feature[_shape] = _data if feature[_shape] is None else np.concatenate((feature[_shape], _data), axis=-1)
        
        # structure feature
        seq_len = len(seq_feat['sequence'])
        struc_feat = Utils.get_struc_feat(seq_len, model_file)
        for _type in self.struc_feat_types:
            _shape = self.struc_feat_types[_type]
            _data = struc_feat[_type]
            feature[_shape] = np.concatenate((feature[_shape], _data), axis=-1)
        
        feature = np.nan_to_num(feature)
        feature['1D'] = feature['1D'].transpose((1,0)).astype(np.float)
        feature['2D'] = feature['2D'].transpose((2,0,1)).astype(np.float)

        model_info = {
            'model': model_file,
        }

        return feature, model_info

class ResNetQA:
    def __init__(self, device_id, params_file):
        """
        Args:
            device_id (int): 0: GPU, -1: CPU.
        """
        self.device = torch.device('cuda:%s'%(device_id)) if device_id>=0 else torch.device('cpu')
        if device_id>=0: torch.cuda.set_device(device_id)
        self.model = QAModel(device=self.device).to(self.device)
        self.model.load_state_dict(torch.load(params_file))

    def get_prediction(self, x):
        pred_global, pred_local = self.model(x['1D'].float().to(self.device), x['2D'].float().to(self.device))
        pred_global, pred_local = pred_global.squeeze(1).cpu().numpy(), pred_local.squeeze(2).cpu().numpy()
        # convert local prediction to distance error
        pred_local[pred_local<0.001] = 0.001 # control the max value
        pred_local = np.sqrt(1/pred_local-1)*3.8
        return pred_global, pred_local

    def predict(self, data_dl):
        self.model.eval()
        QA_results = {}
        with torch.no_grad():
            for x, model_info in data_dl:
                print(model_info)
                pred_global, pred_local = self.get_prediction(x)
                for i, model in enumerate(model_info['model']):
                    QA_results[model] = {
                        'global': np.around(pred_global[i], decimals=4),
                        'local': np.around(pred_local[i], decimals=2),
                    }
                                        
        return QA_results

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("seq_feature", type=str, help="sequence feature file.")
    parser.add_argument("dist_potential", type=str, help="distance potential file.")
    parser.add_argument("models_folder", type=str, help="folder of decoy models.")
    parser.add_argument("output_file", type=str, help="output file.")
    
    parser.add_argument('-device_id', type=int, required=False, dest='device_id', default=-1, help='the device index to use (CPU: -1, GPU: 0,1,2,...).')
    parser.add_argument('-n_worker', type=int, required=False, dest='n_worker', default=0, help='worker num to load data.')
    parser.add_argument('-n_batch', type=int, required=False, dest='n_batch', default=1, help='minibatch size.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    # dataset
    dataset = ModelDataset(args.seq_feature, args.dist_potential, args.models_folder)
    model_dl = DataLoader(dataset, num_workers=args.n_worker, batch_size=args.n_batch, pin_memory=True)

    # Initialize
    resNetQA = ResNetQA(args.device_id, "model_params.pkl")

    # predict
    QA_results = resNetQA.predict(model_dl)

    # dump results
    pickle.dump(QA_results, open(args.output_file, "wb"))
    print(args.output_file, "done.")