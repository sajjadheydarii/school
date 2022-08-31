
"""
Feature Selection: Spectural Method (SPEC)

@author: Sajad Heydari
"""

import scipy.io
from skfeature.function.similarity_based import SPEC
from skfeature.utility import unsupervised_evaluation


def main():
    # load data
    data = pd.read_csv("data.csv")
    X = data['X'].astype(float)    # main_features
    y = data['Y']                   # label_feature

    # specify the second ranking function which uses all except the 1st eigenvalue
    kwargs = {'style': 0}

    # obtain the scores of features
    score = SPEC.spec(X, **kwargs)

    # sort the feature scores in an descending order according to the feature scores
    sorted_scores = SPEC.feature_ranking(score, **kwargs)

    ## perform evaluation on clustering task

    selected_feature_numbers = 20       # number of selected features (for example)
    num_cluster = 5                     # number of clusters, it is usually set as the number of classes in the ground truth (for example)

    # obtain the dataset on the selected features
    selected_features = X[:, sorted_scores[0:selected_feature_numbers]]

    # perform kmeans clustering based on the selected features and repeats 5 times (for example)
    nmi_total = 0
    acc_total = 0
    for i in range(0, 5):
        nmi, acc = unsupervised_evaluation.evaluation(X_selected = selected_features, n_clusters = num_cluster, y = y)
        nmi_total += nmi
        acc_total += acc