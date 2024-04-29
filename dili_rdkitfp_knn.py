import os

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from fingerprint import smiles_to_rdkit_fp
from cross_validation import cv_knn
from cross_validation import knn_model


### CONFIG ###
dataset_dir = "Dataset"
log_dir = "Logs"
dataset_file = 'DILIrank_Liew_Greene_Xu_cid_metabolism.csv'
log_file = 'dili_rdkitfp_knn.log'
dataset = os.path.join(dataset_dir, dataset_file)
logs = os.path.join(log_dir, log_file)
seed = int(np.random.rand() * (2**32 - 1))


### LOAD DATA ###
df = pd.read_csv(dataset)
df.drop(columns=['Label'], inplace=True)
df.rename(columns={'Voted_Label':'Label'}, inplace=True)
n_notox, n_tox = df['Label'].value_counts().to_list()
print("dataset: ", dataset)
print("\tNo DILI #:", n_notox)
print("\tDILI #:", n_tox)
print()

### DATA PREPROCESSING ###
# 500개 random index 생성
n_data = 500
np.random.seed(seed)
notox_idx_500 = np.random.randint(0, n_notox, n_data)   # n_data=500
tox_idx_500 = np.random.randint(0, n_tox, n_data)


### dataset을 dili 약물과 no-dili 약물로 구분지어 각각 500 개씩 뽑아서 trainset으로,
### 나머지는 validation set으로 이용
df = df.sample(frac=1).reset_index(drop=True)       # shuffle
df_notox = df[df['Label']==0].reset_index(drop=True)      # no-DILI 약물만 선택
df_notox_train = df_notox.loc[notox_idx_500, ['Canonical SMILES', 'Label']]   # 500 개 random data
df_notox_train.reset_index(drop=True, inplace=True)
df_notox_val = df_notox.drop(df_notox_train.index).reset_index(drop=True)   # 나머지는 validation data로

df_tox = df[df['Label']==1].reset_index(drop=True)        # DILI 약물만 선택
df_tox_train = df_tox.loc[tox_idx_500, ['Canonical SMILES', 'Label']]     # 500 개 random data
df_tox_train.reset_index(drop=True, inplace=True)
df_tox_val = df_tox.drop(df_tox_train.index).reset_index(drop=True)         # 나머지는 validation data로


### TRAINING DATASET & TESTING DATASET 
df_train = pd.concat([df_tox_train, df_notox_train], axis=0).sample(frac=1).reset_index(drop=True)
df_val = pd.concat([df_notox_val, df_tox_val], axis=0).sample(frac=1).reset_index(drop=True)


### MACCS fingerprints
train_fp = [smiles_to_rdkit_fp(x) for x in df_train['Canonical SMILES']]
val_fp = [smiles_to_rdkit_fp(x) for x in df_val['Canonical SMILES']]
X_train = np.array(train_fp)
X_test = np.array(val_fp)
y_train = np.array(df_train['Label'])
y_test = np.array(df_val['Label'])


### CROSS VALIDATION SEETING ###
n_neighbors = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
leaf_sizes = [10, 20, 30, 40 ,50]
scores = {}
best_metrics = {
    'best_acc': 0.0,
    'best_neigh': 0,
    'best_leaf': 0,
    'seed': seed
}
with open(logs, 'w') as log:
    outs = f"seed: {seed}\n"
    for n_neigh in n_neighbors:
        for n_leaf in leaf_sizes:
            scores = cv_knn(5, (X_train, y_train), n_neigh, n_leaf, seed)
            train_accs = []
            test_accs = []
            outs += f"\nneighbor #: {n_neigh:2d}, "
            outs += f"leaf: {n_leaf:2d}\n"
            for i, score in scores.items():
                train_acc, test_acc = score
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                outs += f"-- fold {i+1}: "
                outs += f"train_acc: {train_acc:>1.5f}, "
                outs += f"test_acc: {test_acc:>1.5f}"
                outs += "\n"

            train_acc_mean = sum(train_accs)/len(scores)
            test_acc_mean = sum(test_accs)/len(scores)
            outs += f"-- mean train acc: {train_acc_mean:>1.5f}\n"
            outs += f"-- mean test acc: {test_acc_mean:>1.5f}\n"
            if test_acc_mean > best_metrics['best_acc']:
                best_metrics['best_acc'] = test_acc_mean
                best_metrics['best_neigh'] = n_neigh
                best_metrics['best_leaf'] = n_leaf
            print(f"neighbos: {n_neigh:2d}, leaf: {n_leaf:2d}, \
                  train_accuracy: {train_acc_mean:>1.5f}, \
                  test_accuracy: {test_acc_mean:>1.5f}")
    print(outs)
    outs += '\nBest Metrics:\n'
    for key, val in best_metrics.items():
        outs += f"  - {key}: {val}\n"
    
    ### BEST MODEL TEST
    n_neighbors = best_metrics['best_neigh']
    leaf_size = best_metrics['best_leaf']
    model = knn_model(
        n_neigh=n_neighbors,
        leaf_size = leaf_size,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('Confusion matrix:')
    cm = confusion_matrix(y_test, preds)
    print(cm)
    outs += "\nConfusion Matrix:\n"
    outs += f"  - TN: {cm[0,0]}\n"
    outs += f"  - FP: {cm[0,1]}\n"
    outs += f"  - FN: {cm[1,0]}\n"
    outs += f"  - TP: {cm[1,1]}"
    log.write(outs)
print()
print(best_metrics)
print()