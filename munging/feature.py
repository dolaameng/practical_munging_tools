"""
Feature Engineering

References: 
1. "Practical Data Science with R" by Nina Zumel and John Mount
"""

import utility
import pandas as pd 
#from sklearn.base import TransformerMinin, BaseEstimator

class BiClassProbabilityFeatureExtractor:
	"""
	Extract new features based on the conditional probability of class label,
	For now it uses a different interface than sklearn.TransformerMinin
	see Ref 1 chapter 6 for details.  
	"""
	def __init__(self):
		self.cprobs = None
		self.fnames = None
	def fit(self, df, fnames, by):
		self.cprobs, self.fnames = [], []
		for f in fnames:
			if not utility.is_categorical(df, f):
				raise ValueError(f+" must be categorical, use encoding or discretizing")
			cprob = pd.crosstab(df[by], df[f])
			cprob = cprob / cprob.sum(axis = 0)
			cprob = cprob.iloc[1, :]
			cprob.name = "%sIs%s_on_%s" % (by, cprob.name, f)
			self.cprobs.append(cprob)
			self.fnames.append(f)
		return self
	def transform(self, df, copy = True):
		result = df.copy() if copy else df 
		for f, cprob in zip(self.fnames, self.cprobs):
			result = result.join(cprob, on = f)
		return result
	def fit_transform(self, df, fnames, by, copy = True):
		return self.fit(df, fnames, by).transform(df, copy)

def _extract_cprobs_by_biclass(df, fnames, by, copy = True):
	"""
	Depreciated - use BiClassProbabilityFeatureExtractor to better handle train and validate data 
	Extract conditional probability features based on binary class labels, 
	see Ref 1 chapter 6 for details. 
	df: DataFrame
	fnames: features to be extracted from - the features must be categorical or discretized numerical
	(call transformation.discretize_numerical for that)
	by: binary class labels (for multiple labels, use one-hot-encoding to get the cprobs-features separately)
	copy: whether copy dataframe or modify in place 
	"""
	result = df.copy() if copy else df 
	for f in fnames:
		if not utility.is_categorical(df, f):
			raise ValueError(f+" must be categorical, use encoding or discretizing")
		cprobs = pd.crosstab(df[by], df[f])
		cprobs = cprobs / cprobs.sum(axis = 0)
		cprobs = cprobs.iloc[1, :]
		cprobs.name = "%sIs%s_on_%s" % (by, cprobs.name, f)
		result = result.join(cprobs, on = f)
	return result

