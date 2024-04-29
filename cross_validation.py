import numpy as np
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def common(k, seed):
    skf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    return skf

def knn_model(n_neigh=5, weights='distance', 
                 leaf_size=30, metric='jaccard'):
    knn = KNeighborsClassifier(
        n_neighbors=n_neigh,
        weights = weights,
        leaf_size = leaf_size,
        metric = metric
    )
    return knn

def cv_knn(k, dataset, neighbor, leaf, seed):
    skf = common(k, seed)
    X_data, y_data = dataset
    scores = {}
    for idx ,(train_ids, test_ids) in enumerate(skf.split(X_data, y_data)):
        model = knn_model(n_neigh=neighbor, leaf_size=leaf)
        X_train = X_data[train_ids]
        y_train = y_data[train_ids]
        X_test = X_data[test_ids]
        y_test = y_data[test_ids]
        model.fit(X_train, y_train)
        tr_score = model.score(X_train, y_train)
        ts_score = model.score(X_test, y_test)
        scores[idx] = [tr_score, ts_score]
        del model
    return scores

def dc_model(depth=None, min_split=2, min_leaf=1, seed=123):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth, \
                                      min_samples_leaf=min_leaf, \
                                      random_state=seed, \
                                      class_weight='balanced')
    return clf

def cv_decision_tree(k, dataset, min_split, min_leaf, seed, depth=None):
    skf = common(k, seed)
    X_data, y_data =dataset
    scores = {}
    for idx ,(train_ids, test_ids) in enumerate(skf.split(X_data, y_data)):
        model = dc_model(depth=depth, min_split=min_split, \
                         min_leaf=min_leaf, seed=seed)
        X_train = X_data[train_ids]
        y_train = y_data[train_ids]
        X_test = X_data[test_ids]
        y_test = y_data[test_ids]
        model.fit(X_train, y_train)
        tr_score = model.score(X_train, y_train)
        ts_score = model.score(X_test, y_test)
        scores[idx] = [tr_score, ts_score]
        del model
    return scores

