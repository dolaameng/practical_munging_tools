"""
Inspect feature columns in data frame, e.g., 
1. Find categorical and numberical features by their dtypes
2. Inspect the missing value patterns in the data
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from utility import is_numerical, is_categorical

############# Columns Inspection ##############
def find_numerical_features(df):
	return np.asarray([f for f in df.columns if is_numerical(df, f)])

def find_categorical_features(df):
	return np.asarray([f for f in df.columns if is_categorical(df, f)])

def find_features_with_nas(df):
	return np.asarray(df.columns[pd.isnull(df).any(axis = 0)])


################### Single Variables ###################################

def plot_features_density(df, feat_names = None, plot_type="density", bins = 30):
	"""
	Plot the density of feature values 
	df: DataFrame
	feat_names: feature of interest, by default all numerical features 
	plot_type: {"density", "hist"}
	"""
	## numerical features
	feat_names = find_numerical_features(df) if feat_names is None else feat_names
	nrows, ncols = feat_names.shape[0] / 3 + 1, 3
	fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (ncols * 6, nrows * 4))
	fig.subplots_adjust(wspace = 0.25, hspace = 0.5)
	axes = axes.ravel()
	for ax, f in zip(axes, feat_names):
		try:
			zscore, pvalue = stats.skewtest(df[f])
			if plot_type is 'density':
				df[f].plot(kind = plot_type, ax = ax, rot = 90)
			else:
				_ = ax.hist(df[f], bins = bins)
			ax.set_title("zscore=%.2g, pvalue=%.2g" % (zscore, pvalue))
			ax.set_xlabel(f)
		except:
			pass

def find_features_skewed(df, skew_thr, feat_names = None):
	"""
	Find the features whose values are skewed above skew_thr. Use plot_features_density to 
	inspect the skewness of features to find the threshold. 
	df: DataFrame
	skew_thr : features with higher skewness will be returned
	feat_names : subset of features of interest, if None, it chooses from numerical features 
	"""
	## find all numerical features by default
	feat_names = find_numerical_features(df) if feat_names is None else feat_names
	return np.asarray([f for f in feat_names if stats.skewtest(df[f])[0] >= skew_thr])


################# Pairwise feature plotting ####################
def plot_feature_pair(df, xname, yname, ax = None, legend = True, figsize = None, *args, **kwargs):
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
	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize = figsize)

	x_dtype = "numerical" if is_numerical(df, xname) else "categorical"
	y_dtype = "numerical" if is_numerical(df, yname) else "categorical"
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


############ Missing Value Inspection ###############

def na_pattern(df):
	"""
	Find the missing value pattern in the data, inspired by MICE in R 
	df: DataFrame
	"""
	na_data = pd.isnull(df)
	na_feat_totals = na_data.sum(axis = 0)
	na_data = na_data.astype(np.object)
	na_data[na_data==True] = "missing"
	na_data[na_data==False] = "-"
	na_patterns = na_data.drop_duplicates()
	occurrences = [(na_data == na_patterns.iloc[i, :]).all(axis = 1).sum()
					for i in xrange(na_patterns.shape[0])]
	na_patterns["occurrence"] = occurrences
	na_patterns = na_patterns.reset_index(drop=True).append(na_feat_totals, ignore_index=True)
	na_patterns.iloc[-1, -1] = df.shape[0]
	return na_patterns
