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
import math
import matplotlib.pyplot as plt 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm 

from transform import *
from model import *

class Session(object):
	def __init__(self, data, target_feature, test_frac = 0.3, copy = True, random_state = None):
		self.data = data.copy() if copy else data
		self.target_feature = target_feature

		self.train_index, self.test_index = train_test_split(np.arange(data.shape[0]), 
												test_size = test_frac, random_state=random_state)

		self.removed_features = np.array([])

		self.params = {
			  "MIN_NUM_VALUES_FOR_NUMERICAL": 5
			, "FRAC_OF_NA_TO_IGNORE": 0.95
			, "FRAC_OF_FEAT_TO_BE_NONINFORMATIVE": 0.96
			, "SKEWNESS_THR": 20
			, "REDUNDANT_FEAT_CORR_THR": 0.95
		}
	def set_parameters(self, **params):
		self.params.update(params)
	def get_parameters(self):
		return self.params
	def get_data(self, selected_features = None):
		if selected_features is None:
			selected_features =  self.get_all_input_features()
		selected_features = np.append(selected_features, self.target_feature)
		train_data = self.data.iloc[self.train_index, :].loc[:, selected_features]
		test_data = self.data.iloc[self.test_index, :].loc[:, selected_features]
		return (train_data, test_data)
	def get_transform_combiners(self, transformers):
		combiner = TransformPipeline(transformers)
		return combiner

	########################## Feature Filtering ##########################
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
	def is_numerized_from_categorical_feature(self, feature_name):
		return feature_name.endswith("_NUMERIZED")

	def get_features_of(self, criterion = None):
		return np.asarray([f for f in self.get_all_input_features()
			if criterion(f)])
	def get_all_input_features(self):
		return np.asarray([f for f in self.data.columns 
			if f not in self.removed_features
			if f != self.target_feature])
	def find_redundant_features(self, feature_names = None):
		if feature_names is None:
			feature_names = self.get_features_of(self.is_numerical_feature)
		corrmat = self.data.loc[:, feature_names].dropna().corr().abs()
		corrmat = corrmat.fillna(value = 0)
		for i in xrange(corrmat.shape[0]):
			corrmat.iloc[i, i] = 0
		corrmean = corrmat.mean(axis = 0)
		redundant_feats = []
		while True:
			try:
				corr_max = np.asarray(corrmat).max()
				if corr_max <= self.params["REDUNDANT_FEAT_CORR_THR"]:
					break
				f1, f2 = corrmat.columns[list(zip(*np.where(corrmat == corr_max))[0])]
				f = f1 if corrmean[f1] > corrmean[f2] else f2
				redundant_feats.append(f)
				corrmat.loc[:, f] = 0
				corrmat.loc[f, :] = 0
			except:
				print corr_max
				print corrmat.columns[list(zip(*np.where(corrmat == corr_max))[0])]
				break 
		return redundant_feats 

		########################## Feature Transformation ##########################
	def remove_features(self, feature_names):
		self.removed_features = np.unique(np.hstack([self.removed_features, feature_names]))
		remover = FeatureRemover(feature_names)
		return remover
	def impute_features(self, feature_names = None, auto_remove = True):
		if feature_names is None:
			feature_names = self.get_features_of(self.is_na_feature)
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
		if feature_names is None:
			feature_names = self.get_features_of(self.is_skewed_numerical_feature)
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
		if feature_names is None:
			feature_names = self.get_features_of(self.is_numerical_feature)
		feature_whitener = NumericalFeatureWhitener(feature_names)
		feature_whitener.fit(self.data.iloc[self.train_index, :])
		self.data = feature_whitener.transform(self.data)
		if auto_remove:
			remover = self.remove_features(feature_names)
			return  TransformPipeline([feature_whitener, remover])
		else:
			return feature_whitener
	def minmax_scale_features(self, feature_names = None, auto_remove = False):
		if feature_names is None:
			feature_names = self.get_features_of(self.is_numerical_feature)
		feature_scaler = NumericalFeatureMinMaxScaler(feature_names)
		feature_scaler.fit(self.data.iloc[self.train_index, :])
		self.data = feature_scaler.transform(self.data)
		if auto_remove:
			remover = self.remove_features(feature_names)
			return TransformPipeline([feature_scaler, remover])
		else:
			return feature_scaler
	def numerize_categorical_features(self, feature_names = None, auto_remove = False):
		if not self.is_categorical_feature(self.target_feature):
			raise ValueError("this method is for classifiation problem")
		if feature_names is None:
			feature_names = self.get_features_of(self.is_categorical_feature)
		numerizer = CategoricalFeatureNumerizer(feature_names, self.target_feature)
		numerizer.fit(self.data.iloc[self.train_index, :])
		self.data = numerizer.transform(self.data)
		if auto_remove:
			remover = self.remove_features(feature_names)
			return  TransformPipeline([numerizer, remover])
		else:
			return numerizer

	########################## Feature Selection ##########################
	def rank_features(self, feature_names, by, *args, **kwargs):
		train_scores, test_scores = zip(*[by(feature_name = f, *args, **kwargs) for f in feature_names])
		return sorted(zip(feature_names, test_scores), key=lambda (f,s): s, reverse=True)
	def numerized_feature_auc_metric(self, feature_name, target_value):
		train_data = self.data.iloc[self.train_index, :][feature_name]
		train_target = self.data.iloc[self.train_index, :][self.target_feature] == target_value
		test_data = self.data.iloc[self.test_index, :][feature_name]
		test_target = self.data.iloc[self.test_index, :][self.target_feature] == target_value
		train_score = roc_auc_score(train_target, train_data)
		test_score = roc_auc_score(test_target, test_data)
		return (train_score, test_score)
	def numerized_feature_logloss_metric(self, feature_name, target_value):
		train_data = self.data.iloc[self.train_index, :][feature_name]
		train_target = self.data.iloc[self.train_index, :][self.target_feature] == target_value
		test_data = self.data.iloc[self.test_index, :][feature_name]
		test_target = self.data.iloc[self.test_index, :][self.target_feature] == target_value
		train_score = -np.mean(np.log(np.where(train_target==target_value, train_data, 1-train_data)))
		test_score = -np.mean(np.log(np.where(test_target==target_value, test_data, 1-test_data)))
		return (train_score, test_score)

	########################## Data Exploration  ##########################
	def print_categorial_crosstable(self, feature_names = None, targets = None):
		if feature_names is None:
			feature_names = self.get_features_of(self.is_categorical_feature)
		targets = targets or [self.target_feature]
		value_tables = []
		for prefix, index in zip(["train_", "test_", "overall_"], 
								[self.train_index, self.test_index, None]):
			df = self.data.iloc[index, :] if index is not None else self.data
			value_table = pd.crosstab(columns = [df[t] for t in targets], 
							index = [df[f] for f in feature_names],
	                        margins=True, dropna = False)
			value_table = value_table.divide(value_table.All, axis = 'index', ).iloc[:, :-2]
			value_table = value_table.replace([-np.inf, np.inf], np.nan).dropna()
			value_table = value_table.rename(columns = {f: prefix+str(f) for f in value_table.columns})
			value_tables.append(value_table)
		result = pd.concat(value_tables, axis = 1, join = 'outer')
		result = result.sort(columns=result.columns[0], ascending=False)
		return result
	def plot_feature_pair(self, xname, yname, ax = None, legend = True, figsize = None, *args, **kwargs):
		"""
		Plot the 'scatter plot' of a pair of two features based on the types of features, 
		e.g., 
		1. numberical vs numbercial - scatter plot with lowess 
		2. numericla vs categorical - density plot grouped by categorical vars 
		3. categorical vs categorical - stacked barchart (hexbin or confusion matrix plot)
		This will help spot useful features that are both common and have extreme patterns (for classification)
		df: DataFrame
		xname: name of feature x (usually an input feature of interest)
		yname: name of feature y (usually the output feature )
		args, kwargs: plotting parameters
		"""
		df = self.data.loc[:, [xname, yname]].dropna()
		if ax is None:
			fig, ax = plt.subplots(1, 1, figsize = figsize)

		x_dtype = "numerical" if self.is_numerical_feature(xname) else "categorical"
		y_dtype = "numerical" if self.is_numerical_feature(yname) else "categorical"
		x, y = df[xname], df[yname]
		if x_dtype is "numerical" and y_dtype is "numerical":
			ax.scatter(x, y, color = "blue", s = 10, marker = ".", *args, **kwargs)
			lowessy = sm.nonparametric.lowess(y, x, return_sorted = False)
			ax.plot(sorted(x), sorted(lowessy), "r-", label="lowess", alpha = 1)
			ax.set_xlabel("%s(%s)" % (xname, x_dtype))
			ax.set_ylabel("%s(%s)" % (yname, y_dtype))
		elif x_dtype is "numerical" and y_dtype is "categorical":
			for value, subdf in df.groupby(by = yname):
				if subdf.shape[0] > 1:
					subdf[xname].plot(kind = "density", label = value, ax = ax)
			ax.set_xlabel("%s|%s" % (xname, yname))
		elif x_dtype is "categorical" and y_dtype is "numerical":
			for value, subdf in df.groupby(by = xname):
				if subdf.shape[0] > 1:
					subdf[yname].plot(kind = "density", label = value, ax = ax)
			ax.set_xlabel("%s|%s" % (yname, xname))
		else: # categorical and categorical
			pd.crosstab(df[xname], df[yname], margins = False).plot(kind = 'barh', stacked = True, ax = ax)
			ax.set_xlabel("dist. of %s" % yname)
		if legend: 
			ax.legend(loc = "best")
		return self 
	def plot_numerical_feature_density(self, feature_names=None):
		if feature_names is None:
			feature_names = [f for f in self.get_features_of(self.is_numerical_feature)
								if f not in self.get_features_of(self.is_numerized_from_categorical_feature)]
		nfeats = len(feature_names)
		nrows, ncols = int(math.ceil(nfeats / 4)), 4
		fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (4 * ncols, 4 * nrows))
		axes = axes.ravel()
		for f, ax in zip(feature_names, axes):
			try:
				self.plot_feature_pair(xname = f, yname = self.target_feature, ax = ax, legend=False)
			except:
				pass
		return self 

	########################## Model Fitting ##################################
	def blend_biclass_models(self, models, blender, 
		score_function = None, 
		feature_names = None, target_value_index = 1, n_folds = 5):
		"""
		Idea credited to https://github.com/emanuele/kaggle_pbr/blob/master/blend.py
		"""
		if feature_names is None:
			feature_names = self.get_all_input_features()
		blender = BiClassModelBlender(feature_names, self.target_feature, models, blender, 
				target_value_index, n_folds)
		blender.fit(self.data.iloc[self.train_index, :])
		return blender 