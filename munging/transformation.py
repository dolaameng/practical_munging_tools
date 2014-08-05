"""
Transformation of features in df, including,
1. log transformation for positive skewed data 
1. sign-log transformation for real value skewed data 

References: 
1. "Practical Data Science with R" by Nina Zumel and John Mount, Chapter 3, 4
"""

import numpy as np 
import pandas as pd

##################### variable transformation #######################

def log_transform(df, feat_names, copy = True, prefix = "log_"):
	"""
	Transform data using log, specially for data generated by mulplicative process.
	The data to be transformed must be POSITIVE, otherwise remove the bad data first 
	or use signed_log_transform instead.
	df: DataFrame
	feat_names: names of features to be transformed. 
	"""
	for f in feat_names:
		if np.any(df[f] <= 0):
			raise ValueError(f+":log_transform is for positive values, remove bad data or use signed_log_transform")
	result = df.copy() if copy else df 
	#result.loc[:, feat_names] = np.log(result.loc[:, feat_names])
	for f in feat_names:
		result[prefix+f] = np.log(result[f])
	return result

def arcsinh_transform(df, feat_names, copy = True, prefix = "arcsinh_"):
	"""
	One of the universal methods other than signed logarithms to transform real values, 
	- the arcsinh function (see http://mng.bz/ZWQa)
	but they also distort data near zero and make almost any data appear to be bimodal.
	"""
	result = df.copy() if copy else df 
	#result.loc[:, feat_names] = np.arcsinh(result.loc[:, feat_names])
	for f in feat_names:
		result[prefix+f] = np.arcsinh(result[f])
	return result

def signed_log_transform(df, feat_names, copy = True, prefix = "slog_"):
	"""
	One of the universal methods other than signed logarithms to transform real values, 
	- the signed log function.
	"""
	result = df.copy() if copy else df 
	#d = result.loc[:, feat_names]
	#result.loc[:, feat_names] = np.where(np.abs(d) <= 1, 0, np.sign(d) * np.log(np.abs(d)))
	for f in feat_names:
		d = df[f]
		result[prefix+f] = np.where(np.abs(d) <= 1, 0, np.sign(d) * np.log(np.abs(d)))
	return result


#################### discretization of continouse numerical variable #####################

def discretize_numerical(df, feat_names, copy = True, max_qcut = 10, feat_bins = None, prefix = "discrete_"):
	"""
	Discretize numerical variables by cut them into bins (equal size / equal percentile).
	For a specific feature, it can be discretiezed in 3 different ways:
	(1) customized bins to cut - specify in feat_bins that {fname: bins_to_cut}
	(2) customized number of enqual bins to cut - specify in feat_bins that {fname: ncut}
	(3) quantile cut - specify max_qcut and leave in feat_bins {fname: None }  
	df: DataFrame
	feat_names: names of features to discretize
	copy: whether to make a new copy or use df 
	feat_bins : either None (default quantile-cut of 10 peices) or dictionary {fname: bins_to_cut}, e.g.
	{"x1": 10, "x2": [1, 2, 3], "x3": None}
	max_qcut: number of quantile cuts if feat_bins is not specified for that feature
	"""
	result = df.copy() if copy else df 
	feat_bins = {} if feat_bins is None else feat_bins
	for f in feat_names:
		bins = feat_bins.get(f, None)
		if bins is None:
			## there is a bug in current numpy percentile - use pandas quantile instead
			bins = np.unique(df[f].quantile(np.linspace(0., 1., max_qcut)))
		result[prefix+f] = np.asarray(pd.cut(result[f], bins, include_lowest=True))
	return result 