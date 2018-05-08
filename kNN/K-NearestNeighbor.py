from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy

class KNN:
    k_n = None
    datas = None
    lbl = None
    df = None
    def __init__(self, k: int, distance_function) -> float:
        global k_n
        global df
        self.k = k
        k_n = k
        self.distance_function = distance_function
        df = distance_function
    def train(self, features: List[List[float]], labels: List[int]):
        global datas
        global lbl
        datas = features
        lbl = labels
        return None

    def predict(self, features: List[List[float]]) -> List[int]:
        tmp = {}
        max_lbl_features = []
        for f in features:
            for k,d in enumerate(datas):
                tmp[k]= df(f,d)
            nn_labels = [lbl[key] for key,value in sorted(tmp.items(), key=lambda x: x[1])[:k_n]]
            max_lbl_features.append(int(max(set(nn_labels),key=nn_labels.count)))
        return max_lbl_features


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
