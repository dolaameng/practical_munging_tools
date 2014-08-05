"""
Feature Engineering

References: 
1. "Practical Data Science with R" by Nina Zumel and John Mount
"""

import utility
import pandas as pd 

def extract_cprobs_by_biclass(df, fnames, by, copy = True):
	"""
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
		cprobs = cprobs.iloc[0, :]
		cprobs.name = "%s_on_%s" % (by, f)
		result = result.join(cprobs, on = f)
	return result

##TODO: rank features