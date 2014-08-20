"""
## preconditions of data transformaions
1. centering & scaling <- unskewed log-transformation for skewed data (or outlier/invalid removal)
2. unskewed log-transformation <- missing value imputation / noninformative feature removal
3. missing value imputation <- None 
4. feature l2 normalization <- centering & scaling 
5. pca <- centering & scaling 
6. discretization <- missing value imputation
7. zero-variance features <- None 
"""

import pandas as pd 
import numpy as np 
from scipy import stats 
from sklearn.cross_validation import train_test_split
from transform import *

class Session(object):
	def __init__(self, data, target_feature, test_frac = 0.3, copy = True):
		self.data = data.copy() if copy else data
		self.target_feature = target_feature

		self.train_index, self.test_index = train_test_split(np.arange(data.shape[0]), 
												test_size = test_frac)

		self.removed_features = np.array([])

		self.params = {
			  "MIN_NUM_VALUES_FOR_NUMERICAL": 5
			, "FRAC_OF_NA_TO_IGNORE": 0.95
			, "FRAC_OF_FEAT_TO_BE_NONINFORMATIVE": 0.95
			, "SKEWNESS_THR": 20
		}
	def set_parameters(self, **params):
		self.params.update(params)
	def get_parameters(self):
		return self.params

	
	def is_numerical_feature(self, feature_name):
		ftype = self.data[feature_name].dtype
		if ftype in np.array([np.double, np.float]):
			return True
		elif ftype in np.array([np.int]):
			return len(self.data[feature_name].unique()) >= self.params["MIN_NUM_VALUES_FOR_NUMERICAL"]
	def is_categorical_feature(self, feature_name):
		ftype = self.data[feature_name].dtype
		if ftype in np.array([np.bool, np.object]):
			return True
		elif ftype in np.array([np.int]):
			return len(self.data[feature_name].unique()) < self.params["MIN_NUM_VALUES_FOR_NUMERICAL"]
	def is_na_feature(self, feature_name):
		return np.any(pd.isnull(self.data[feature_name]))
	def is_na_heavy(self, feature_name):
		return np.mean(pd.isnull(self.data[feature_name])) >= self.params["FRAC_OF_NA_TO_IGNORE"]
	def is_skewed_numerical_feature(self, feature_name):
		if not self.is_numerical_feature(feature_name):
			return False 
		skewness, pvalue = stats.skewtest(self.data[feature_name].dropna())
		if skewness >= self.params["SKEWNESS_THR"] and pvalue <= 0.01:
			return True
		else:
			return False 
	def is_noninformative_feature(self, feature_name):
		value_counts = pd.value_counts(self.data[feature_name], dropna = False)
		if len(value_counts) == 1:
			return True 
		elif value_counts.max()*1./self.data.shape[0] >= self.params["FRAC_OF_FEAT_TO_BE_NONINFORMATIVE"]:
			return True 
		return False 
	def get_features_of(self, criterion = None):
		return np.asarray([f for f in self.get_all_input_features()
			if criterion(f)])
	def get_all_input_features(self):
		return np.asarray([f for f in self.data.columns 
			if f not in self.removed_features
			if f is not self.target_feature])


	def remove_features(self, feature_names):
		self.removed_features = np.unique(np.hstack([self.removed_features, feature_names]))
		return self
	def unremove_features(self, feature_names):
		self.removed_features = np.asarray([f for f in self.removed_features
									if f not in feature_names])
		return self
	def impute_features(self, feature_names = None, auto_remove = True):
		feature_names = feature_names or self.get_features_of(self.is_na_feature)
		feature_types = ['categorical' if self.is_categorical_feature(f) else 'numerical'
							for f in feature_names]
		feature_imputer = FeatureImputer(dict(zip(feature_names, feature_types)))
		feature_imputer.fit(self.data.loc[self.train_index, feature_names])
		self.data = feature_imputer.transform(self.data)
		if auto_remove:
			self.remove_features(feature_names)
		return feature_imputer