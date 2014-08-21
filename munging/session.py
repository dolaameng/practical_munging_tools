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
			, "REDUNDANT_FEAT_CORR_THR": 0.95
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
		else:
			return False 
	def is_categorical_feature(self, feature_name):
		ftype = self.data[feature_name].dtype
		if ftype in np.array([np.bool, np.object]):
			return True
		elif ftype in np.array([np.int]):
			return len(self.data[feature_name].unique()) < self.params["MIN_NUM_VALUES_FOR_NUMERICAL"]
		else:
			return False 
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
			if f != self.target_feature])
	def find_redundant_features(self, feature_names = None):
		feature_names = feature_names or self.get_features_of(self.is_numerical_feature)
		corrmat = self.data.loc[:, feature_names].dropna().corr().abs()
		for i in xrange(corrmat.shape[0]):
			corrmat.iloc[i, i] = 0
		corrmean = corrmat.mean(axis = 0)
		redundant_feats = []
		while True:
			corr_max = np.asarray(corrmat).max()
			if corr_max <= self.params["REDUNDANT_FEAT_CORR_THR"]:
				break
			f1, f2 = corrmat.columns[np.where(corrmat == corr_max)[0]]
			f = f1 if corrmean[f1] > corrmean[f2] else f2
			redundant_feats.append(f)
			corrmat.loc[:, f] = 0
			corrmat.loc[f, :] = 0
		return redundant_feats 


	def remove_features(self, feature_names):
		self.removed_features = np.unique(np.hstack([self.removed_features, feature_names]))
		remover = FeatureRemover(feature_names)
		return remover
	def impute_features(self, feature_names = None, auto_remove = True):
		feature_names = feature_names or self.get_features_of(self.is_na_feature)
		feature_types = ['categorical' if self.is_categorical_feature(f) else 'numerical'
							for f in feature_names]
		feature_imputer = FeatureImputer(dict(zip(feature_names, feature_types)))
		feature_imputer.fit(self.data.iloc[self.train_index, :])
		self.data = feature_imputer.transform(self.data)
		if auto_remove:
			remover = self.remove_features(feature_names)
			return  TransformPipeline([feature_imputer, remover])
		else:
			return feature_imputer
	def evenize_skew_features(self, feature_names = None, auto_remove = False):
		feature_names = feature_names or self.get_features_of(self.is_skewed_numerical_feature)
		feature_transforms = ['log' if self.data[f].min() > 0
									else 'log_plus1' if self.data[f].min() >= 0
									else 'signed_log'
								 for f in feature_names]
		feature_evenizer = NumericalFeatureEvenizer(dict(zip(feature_names, feature_transforms)))
		feature_evenizer.fit(self.data.iloc[self.train_index, :])
		self.data = feature_evenizer.transform(self.data)
		if auto_remove:
			remover = self.remove_features(feature_names)
			return  TransformPipeline([feature_evenizer, remover])
		else:
			return feature_evenizer
	def whiten_features(self, feature_names = None, auto_remove = False):
		feature_names = feature_names or self.get_features_of(self.is_numerical_feature)
		feature_whitener = NumericalFeatureWhitener(feature_names)
		feature_whitener.fit(self.data.iloc[self.train_index, :])
		self.data = feature_whitener.transform(self.data)
		if auto_remove:
			remover = self.remove_features(feature_names)
			return  TransformPipeline([feature_whitener, remover])
		else:
			return feature_whitener
	def numerize_categorical_features(self, feature_names = None, auto_remove = False):
		if not self.is_categorical_feature(self.target_feature):
			raise ValueError("this method is for classifiation problem")
		feature_names = feature_names or self.get_features_of(self.is_categorical_feature)
		numerizer = CategoricalFeatureNumerizer(feature_names, self.target_feature)
		numerizer.fit(self.data.iloc[self.train_index, :])
		self.data = numerizer.transform(self.data)
		if auto_remove:
			remover = self.remove_features(feature_names)
			return  TransformPipeline([numerizer, remover])
		else:
			return numerizer

	def _get_crossvalue_table(self, feats, targets):
		value_tables = []
		for prefix, index in zip(["train_", "validation_", "overall_"], 
								[self.train_index, self.test_index, None]):
			df = self.data.iloc[index, :] if index is not None else self.data
			value_table = pd.crosstab(columns = [df[t] for t in targets], 
							index = [df[f] for f in feats],
	                        margins=True, dropna = False)
			value_table = value_table.divide(value_table.All, axis = 'index', ).iloc[:, :-2]
			value_table = value_table.replace([-np.inf, np.inf], np.nan).dropna()
			value_table = value_table.rename(columns = {f: prefix+f for f in value_table.columns})
			value_tables.append(value_table)
		result = pd.concat(value_tables, axis = 1, join = 'outer')
		result = result.sort(columns=result.columns[0], ascending=False)
		return result