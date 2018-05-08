from typing import List
import math
import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    tmp = [(y_true[k]-y_pred[k])**2 for k,v in enumerate(y_true)]
    return float(sum(tmp)/len(y_true))


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    score = float(sum([1 if real_labels[k]==predicted_labels[k] else 0 for k,v in enumerate(real_labels)])/len(real_labels))
    return score


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    feature_extend = []
    for f in features:
        feature_extend.append([round(float(np.power(np.array(c),s)),6) for s in range(1,k+1) for c in f])
    return feature_extend


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    if len(point1) is not len(point2):
        while(len(point2)>len(point1)):
            point1.append(float(0))
        while(len(point1)>len(point2)):
            point2.append(float(0))
    tmp = [(point1[k]-point2[k])**2 for k,v in enumerate(point1)]
    ed = float(sum(tmp)**(0.5))
    return ed


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    if len(point1) is not len(point2):
        while(len(point2)>len(point1)):
            point1.append(float(0))
        while(len(point1)>len(point2)):
            point2.append(float(0))
    ipd = sum([point1[k]*point2[k] for k,v in enumerate(point1)])
    return ipd


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    return math.exp(-0.5*euclidean_distance(point1, point2))


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        f_l = []
        for f in features:
            c_l = []
            for c in f:
                c_l.append(float(c/inner_product_distance(f, f)**(0.5)))
            f_l.append(c_l)
        return f_l


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        scaled =[]
        mins = []
        maxs = []
        
        for c in range(len(features[0])):
            tmp = []
            for f in features:  
                tmp.append(f[c])
            mins.append(min(tmp))
            maxs.append(max(tmp))
        tmp = []
        for f in features:
            for k,v in enumerate(f):
                tmp.append((f[k]-mins[k])/(maxs[k]-mins[k]))
            scaled.append(tmp)
            tmp = []
            
        return scaled
