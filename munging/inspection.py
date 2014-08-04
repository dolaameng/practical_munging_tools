"""
Inspect feature columns in data frame, e.g., 
1. Find categorical and numberical features by their dtypes
2. Inspect the missing value patterns in the data
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt 
from utility import NUMERICAL_FEAT_DTYPES, CATEGORICAL_FEAT_DTYPES

############# Columns Inspection ##############
def find_features_with_dtypes(df, dtypes):
	"""
	Find feature names in df with specific dtypes
	df: DataFrame
	dtypes: data types (defined in numpy) to look for
	e.g, categorical features usually have dtypes np.object, np.bool
	and some of them have np.int (with a limited number of unique items)
	"""
	return np.asarray([fname for (fname, ftype) in df.dtypes.to_dict().items()
						if ftype in dtypes]) 

def find_features_with_nas(df):
	return np.asarray(df.columns[pd.isnull(df).any(axis = 0)])




def plot_features_density(df, feat_names = None, plot_type="density", bins = 30):
	"""
	Plot the density of feature values 
	df: DataFrame
	feat_names: feature of interest, by default all numerical features 
	plot_type: {"density", "hist"}
	"""
	## numerical features
	feat_names = find_features_with_dtypes(df, NUMERICAL_FEAT_DTYPES) if feat_names is None else feat_names
	nrows, ncols = feat_names.shape[0] / 3 + 1, 3
	fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (ncols * 6, nrows * 4))
	fig.subplots_adjust(wspace = 0.25, hspace = 0.5)
	axes = axes.ravel()
	for ax, f in zip(axes, feat_names):
		zscore, pvalue = stats.skewtest(df[f])
		if plot_type is 'density':
			df[f].plot(kind = plot_type, ax = ax, rot = 90)
		else:
			_ = ax.hist(df[f], bins = bins)
		ax.set_title("zscore=%.2g, pvalue=%.2g" % (zscore, pvalue))
		ax.set_xlabel(f)

def find_features_skewed(df, skew_thr, feat_names = None):
	"""
	Find the features whose values are skewed above skew_thr. Use plot_features_density to 
	inspect the skewness of features to find the threshold. 
	df: DataFrame
	skew_thr : features with higher skewness will be returned
	feat_names : subset of features of interest, if None, it chooses from numerical features 
	"""
	## find all numerical features by default
	feat_names = find_features_with_dtypes(df, NUMERICAL_FEAT_DTYPES) if feat_names is None else feat_names
	return np.asarray([f for f in feat_names if stats.skewtest(df[f])[0] >= skew_thr])


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
