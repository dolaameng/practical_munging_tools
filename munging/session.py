"""
## preconditions of data transformaions
1. centering & scaling <- unskewed log-transformation for skewed data (or outlier/invalid removal)
2. unskewed log-transformation <- None
3. missing value imputation <- None 
4. feature l2 normalization <- centering & scaling 
5. pca <- centering & scaling 
6. discretization <- missing value imputation
"""

import pandas as pd 
import numpy as np 

class Session(object):
	def __init__(self, data, target_feature, validation_frac = 0.3, copy = True):
		self.data = data.copy() if copy else data
		self.target_feature = target_feature

		self.removed_features = np.array([])

		self.params = {
			"MIN_NUM_VALUES_FOR_NUMERICAL": 5
		}
	def is_numerical_feature(self, feature_name):
		ftype = self.data[feature_name].dtype
		if ftype in np.array([np.double, np.float]):
			return True
		elif ftype in np.array([np.int]):
			return len(self.data[feature_name].unique()) >= self.params["MIN_NUM_VALUES_FOR_NUMERICAL"]
	def is_categorical_feature(self, feature_name):
		pass
	def is_na_feature(self, feature_name):
		pass
	def is_skewed_numerical_feature(self, feature_name):
		pass
	def is_noninformative_feature(self, feature_name):
		pass
	def get_features_of(self, criterion = None):
		return np.asarray([f for f in self.get_all_input_features()
			if criterion(f)])
	def get_all_input_features(self):
		return np.asarray([f for f in self.data.columns 
			if f not in self.removed_features
			if f is not self.target_feature])
	def remove_features(self, feature_names):
		self.removed_features = np.hstack([self.removed_features, feature_names])
		return self
	def unremove_features(self, feature_names):
		self.removed_features = np.asarray([f for f in self.removed_features
									if f not in feature_names])
		return self