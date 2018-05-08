from __future__ import division, print_function

from typing import List

import numpy
import scipy


class LinearRegression:
    wtilde = None
    nfeatures = None
    def __init__(self, nb_features: int):
        global nfeatures
        self.nb_features = nb_features
        nfeatures = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        global wtilde
        Xtilde = numpy.matrix([[1]+p for p in features])
        Xtildetrans = Xtilde.transpose()
        y = numpy.matrix(values)
        wtilde = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(Xtildetrans,Xtilde)),Xtildetrans),numpy.array(y).reshape((-1,1)))
        return None

    def predict(self, features: List[List[float]]) -> List[float]:
        tmp = []
        cbne = []
        for f in features:
            tmp.append([1]+f)
        lst = []
        for f in tmp:
            for k,v in enumerate(f):
                cbne.append(float(wtilde.tolist()[k][0])*(f[k]))
            lst.append(sum(cbne))
            cbne = []

        return lst

    def get_weights(self) -> List[float]:
        return [float(w) for w in wtilde]

        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """


class LinearRegressionWithL2Loss:
    wtilde = None
    lmda = None
    nfeatures = None
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        global lmda
        global nfeatures
        self.alpha = alpha
        lmda = alpha
        nfeatures = nb_features
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):    
        global wtilde
        Xtilde = numpy.matrix([[1]+p for p in features])
        Xtildetrans = Xtilde.transpose()
        y = numpy.matrix(values)
        alphamat = lmda#numpy.matrix([lmda for s in range(len(features[0])+1)])
        idmat = numpy.identity(len(features[0])+1)
        #print(idmat)
        #print((alphamat*idmat))
        wtilde = numpy.matmul(numpy.matmul(numpy.linalg.inv(numpy.matmul(Xtildetrans,Xtilde)+(alphamat*idmat)),Xtildetrans),numpy.array(y).reshape((-1,1)))
        
        return None   

    def predict(self, features: List[List[float]]) -> List[float]:
        tmp = []
        cbne = []
        for f in features:
            tmp.append([1]+f)
        lst = []
        for f in tmp:
            for k,v in enumerate(f):
                cbne.append(float(wtilde.tolist()[k][0])*(f[k]))
            lst.append(sum(cbne))
            cbne = []

        return lst 

    def get_weights(self) -> List[float]:
        return [float(w) for w in wtilde]
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
