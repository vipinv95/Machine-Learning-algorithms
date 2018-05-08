import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
	# Boosting from pre-defined classifiers
		def __init__(self, clfs: Set[Classifier], T=0):
				self.clfs = clfs
				self.num_clf = len(clfs)
				if T < 1:
						self.T = self.num_clf
				else:
						self.T = T

				self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
				self.betas = []       # list of weights beta_t for t=0,...,T-1
				return

		@abstractmethod
		def train(self, features: List[List[float]], labels: List[int]):
				return

		def predict(self, features: List[List[float]]) -> List[int]:
			h = [1 if sum([self.betas[t]*self.clfs_picked[t].predict(features)[n] for t in range(self.T)]) > 0 else -1 for n in range(len(features))]
			return h

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
			Boosting.__init__(self, clfs, T)
			self.clf_name = "AdaBoost"
			return

	def train(self, features: List[List[float]], labels: List[int]):
		f_size = len(features)
		w = np.array([1./f_size for n in range(f_size)]).transpose()
		for t in range(self.T):
			h_err_min = float('inf')
			for clf in self.clfs:
				h_err = float(sum([w[n] for n in range(f_size) if not (clf.predict(features)[n] == labels[n])]))
				if h_err < h_err_min:
					h_t = clf
					h_err_min = h_err
			self.clfs_picked.append(h_t)
			e_t = float(sum([w[n] for n in range(f_size) if not (h_t.predict(features)[n] == labels[n])]))
			b_t = 0.5*float(np.log([(1-e_t)/e_t]))
			self.betas.append(b_t)
			w = [w[n]*float(np.exp([-1*b_t])) if (h_t.predict(features)[n] == labels[n]) else w[n]*float(np.exp([b_t])) for n in range(f_size)]
			w_tot = sum(w)
			w = [wi/w_tot for wi in w]

	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		pi_t = [0.5 for n in range(len(features))]
		z_t = []
		w = []
		f_x = None
		for t in range(self.T):
			z_t = [(((labels[n]+1)/2)-pi_t[n])/(pi_t[n]*(1-pi_t[n])) for n in range(len(features))]
			w = [pi_t[n]*(1-pi_t[n]) for n in range(len(features))]
			w_tot = sum(w)
			w = [wi/w_tot for wi in w]
			h_err_min = float('inf')
			for clf in self.clfs:
				h_err = float(sum([w[n]*((z_t[n]-clf.predict(features)[n])**2) for n in range(len(features))]))
				if h_err < h_err_min:
					h_t = clf
					h_err_min = h_err
			self.clfs_picked.append(h_t)
			self.betas.append(0.5)
		
			if f_x is not None:
				f_x = [f_x[n] + (0.5*h_t.predict(features)[n]) for n in range(len(features))]
			else:
				f_x = [(0.5*h_t.predict(features)[n]) for n in range(len(features))]
			pi_tmp = [1/(1+float(np.exp([-2*f_x[n]]))) for n in range(len(pi_t))] 
			pi_t = pi_tmp
				
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
