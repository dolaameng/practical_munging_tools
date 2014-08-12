"""
Create a data analysis session by maintaining train and test data information.


Most of the functions in munging package will deal with a Session object.
"""

import pandas as pd 
import numpy as np 
import math
from scipy import stats 
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt 
import statsmodels.api as sm 

import transform

class Session():
	"""
	Data Analysis Session - manage all data, intermediate results
	and models. Specially it needs to record what has been done?
	"""
	def __init__(self, data, target_name, copy = True, validation_frac = 0.3):
		## interface to data 
		self.data = data.copy() if copy else data
		self.target_name = target_name
		data_index = np.arange(data.shape[0])
		self.train_index, self.validation_index = train_test_split(data_index, 
			test_size = validation_frac)
		## controal parameters
		self.catgorical_feature_n_values = 5
		self.skewness_thr = 20
		self.unique_value_frac_thr = 0.1
		self.feature_skewness_thr = 20
		self.corr_thr = 0.95
	################ Data Exploration / Inspection #########################################
	def get_features(self):
		return self.data.columns
	def get_train_data(self):
		return self.data.iloc[self.train_index, :]
	def get_validation_data(self):
		return self.data.iloc[self.validation_index, :]
	def find_numerical_features(self):
		## lazy evaluation of feature names in case new features are added
		feature_names = np.asarray([f for f in self.data.columns if f != self.target_name])
		return np.asarray([f for f in feature_names if self.is_numerical(f)])
	def is_numerical(self, f):
		return ((self.data[f].dtype in np.array([np.double, np.float])) or 
			   (self.data[f].dtype in np.array([np.int]) 
				and len(self.data[f].unique()) >= self.catgorical_feature_n_values))
	def find_categorical_features(self):
		feature_names = np.asarray([f for f in self.data.columns if f != self.target_name])
		return np.asarray([f for f in feature_names if self.is_categorical(f)])
	def is_categorical(self, f):
		return ((self.data[f].dtype in np.array([np.bool, np.object])) or 
			   (self.data[f].dtype in np.array([np.int]) and
			   	len(self.data[f].unique()) <= self.catgorical_feature_n_values))
	def find_na_features(self):
		feature_names = np.asarray([f for f in self.data.columns if f != self.target_name])
		na_pattern = pd.isnull(self.data).any(axis = 0)
		return np.asarray([f for f in feature_names if na_pattern[f]])
	def find_skewed_features(self):
		"""
		1. it is a numerical feature 
		2. its max_value / min_value >= 20 -- not accurate
		3. or optionally, use the skewness test in "scipy.stats.skewtest"
		"""
		skewed_feats = []
		for f in self.find_numerical_features():
			xs = self.data[f].dropna()
			try:
				skewness, pvalue = stats.skewtest(xs)
				if skewness >= self.skewness_thr and pvalue <= 0.01:
					skewed_feats.append(f)
			except:
				if xs.max() * 1. / xs.min() >= self.skewness_thr:
					skewed_feats.append(f)
		return np.asarray(skewed_feats)
	def find_noninformative_features(self):
		"""
		1. it can be either a categorical/numerical feature 
		2. #_unique_values / sample_size <= 0.1
		3. #_most_freq_value/#_second_most_freq_value is large >= 20
		"""
		feature_names = np.asarray([f for f in self.data.columns if f != self.target_name])
		noninformative_feats = []
		for f in feature_names:
			value_counts = pd.value_counts(self.data[f], dropna = False)
			if len(value_counts) <= 1:
				noninformative_feats.append(f)
			elif len(value_counts)*1. / self.data.shape[0] <= self.unique_value_frac_thr:
				max_1st, max_2nd = np.asarray(sorted(value_counts))[[-1, -2]]
				if max_1st*1. / max_2nd >= self.feature_skewness_thr:
					noninformative_feats.append(f)
		return np.asarray(noninformative_feats)
	def find_redundant_features(self):
		cmatrix = self.data.corr().abs()
		for i in xrange(cmatrix.shape[0]):
			cmatrix.iloc[i, i] = 0
		cmean = cmatrix.mean(axis = 0)
		redundant_feats = []
		while True:
			max_corr = np.asarray(cmatrix).max()
			if max_corr <= self.corr_thr:
				break
			print max_corr, cmatrix.columns[np.where(cmatrix == max_corr)[0]]
			f1, f2 = cmatrix.columns[np.where(cmatrix == max_corr)[0]]
			f = f1 if cmean[f1] > cmean[f2] else f2 
			redundant_feats.append(f)
			cmatrix.loc[:, f] = 0
			cmatrix.loc[f, :] = 0
		return redundant_feats 
	def get_crossvalue_table(self, feats, targets):
		value_tables = []
		for prefix, index in zip(["train_", "validation_", "overall_"], 
								[self.train_index, self.validation_index, None]):
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

	############# Data Exploration / Plotting #############################################
	def plot_feature_density(self, feat_names = None, kind="density", bins = 30):
		"""
		Plot the density of feature values 
		df: DataFrame
		feat_names: feature of interest, by default all numerical features 
		kind: {"density", "hist"}
		"""
		## numerical features
		feat_names = self.find_numerical_features() if feat_names is None else np.asarray(feat_names)
		df = self.data.loc[:, feat_names]
		nrows, ncols = int(math.ceil(feat_names.shape[0] / 3.)), 3
		fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (ncols * 6, nrows * 4))
		fig.subplots_adjust(wspace = 0.25, hspace = 0.5)
		axes = axes.ravel()
		for ax, f in zip(axes, feat_names):
			try:
				zscore, pvalue = stats.skewtest(df[f].dropna())
				if kind is 'density':
					df[f].dropna().plot(kind = kind, ax = ax, rot = 90)
				else:
					_ = ax.hist(df[f].dropna(), bins = bins)
				ax.set_title("zscore=%.2g, pvalue=%.2g" % (zscore, pvalue))
				ax.set_xlabel(f)
			except:
				pass
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

		x_dtype = "numerical" if self.is_numerical(xname) else "categorical"
		y_dtype = "numerical" if self.is_numerical(yname) else "categorical"
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
	################ Feature Transformation/Removal - removal skewness/outliers#############
	def remove_features(self, feats):
		feats = np.asarray([f for f in feats if f in self.data.columns])
		remover = transform.FeatureRemover(feats, copy = False).fit(self.data.iloc[self.train_index, :])
		self.data = remover.transform(self.data)
		return remover
	################ Missing Values / Outliers Handling ####################################
	################ Feature Extraction / Ranking ##########################################
	################ Build Models ##########################################################
	################ Calibrate Model Performances ##########################################
	################ Make Predictions ######################################################
