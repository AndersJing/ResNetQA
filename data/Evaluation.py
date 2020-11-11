#!/usr/bin/env python3
# encoding: utf-8

import os, pickle, bz2
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

def get_result(pred_pkl, label_type="GDT_TS"):
    data = {
        'global_label': [], 'global_pred': [],
        'local_raw_labels': [], 'local_raw_preds': [], 'local_labels': [], 'local_preds': [], 'ASE': [], 
    }
    pred = pickle.load(bz2.BZ2File(pred_pkl, 'rb'))
    for decoy in pred:
        # # label
        if label_type=="GDT_TS":
            global_label = pred[decoy]['GDT_TS']
            local_label = pred[decoy]['Dist_Err']
        if label_type=="lDDT":
            global_label = pred[decoy]['lDDT']
            local_label = pred[decoy]['lDDT_local']

        # # pred
        global_pred = pred[decoy]['pred_global']
        local_pred = pred[decoy]['pred_local']

        # global
        data['global_label'].append(global_label)
        data['global_pred'].append(global_pred)

        # local
        local_pred = local_pred[local_label>0]
        local_label = local_label[local_label>0]
        data['local_raw_labels'].extend(local_label)
        data['local_raw_preds'].extend(local_pred)
        # S score
        if label_type=="GDT_TS":
            local_label = 1/(1+(local_label/5.0)**2)
            local_pred = 1/(1+(local_pred/5.0)**2)

        data['local_labels'].extend(local_label)
        data['local_preds'].extend(local_pred)
        # ASE
        ASE =  1 - np.mean(abs(local_label - local_pred))
        data['ASE'].append(ASE)

    # local
    L_PCC = np.corrcoef(data['local_labels'], data['local_preds'])[0, 1]
    L_ASE = np.mean(data['ASE'])
    local_labels, local_preds = np.array(data['local_raw_labels']), np.array(data['local_raw_preds'])
    if label_type=="GDT_TS":
        local_labels[local_labels<=3.8] = -1
        local_labels[local_labels>3.8] = 1
    if label_type=="lDDT":
        local_labels[local_labels>=0.4] = 1
        local_labels[local_labels<0.4] = -1
    L_AUC = roc_auc_score(local_labels, local_preds)

    # global
    G_PCC = np.corrcoef(data['global_label'], data['global_pred'])[0, 1]
    G_Spearman = stats.spearmanr(data['global_label'], data['global_pred'])[0]
    G_Kendall = stats.kendalltau(data['global_label'], data['global_pred'])[0]
    G_Diff = np.mean(abs(np.array(data['global_label'])-np.array(data['global_pred'])))
    G_Loss = abs(data['global_label'][np.argmax(data['global_label'])]-data['global_label'][np.argmax(data['global_pred'])])

    result = {
        'Local':{'PCC': L_PCC, 'ASE': L_ASE, 'AUC': L_AUC,},
        'Global':{'PCC': G_PCC, 'Spearman': G_Spearman, 'Kendall': G_Kendall, 'Diff': G_Diff, 'Loss': G_Loss,}
    }

    return result

def evaluation(pred_dir, label_type):
    print(pred_dir)
    results = []
    for _file in os.listdir(pred_dir):
        pred_pkl = "%s/%s"%(pred_dir, _file)
        result = get_result(pred_pkl, label_type=label_type)
        results.append(result)
        
    for _key in ['PCC', 'ASE', 'AUC']:
        print('  Local:', _key, "%.4f"%np.mean([result['Local'][_key] for result in results]))

    for _key in ['PCC', 'Spearman', 'Kendall', 'Diff', 'Loss']:
        print('  Global:', _key, "%.4f"%np.mean([result['Global'][_key] for result in results]))

if __name__ == "__main__":
    # CASP12_stage2
    for sub_dir in os.listdir("CASP12_stage2/"):
        pred_dir = "CASP12_stage2/%s/"%(sub_dir)
        label_type = 'lDDT' if sub_dir.split('-')[-1]=='lDDT' else 'GDT_TS'
        evaluation(pred_dir, label_type)
    # CASP13_stage2
    for sub_dir in os.listdir("CASP13_stage2/"):
        pred_dir = "CASP13_stage2/%s/"%(sub_dir)
        label_type = 'lDDT' if sub_dir.split('-')[-1]=='lDDT' else 'GDT_TS'
        evaluation(pred_dir, label_type)