"""
Inspect feature columns in data frame, e.g., 
1. Find categorical and numberical features by their dtypes
2. Inspect the missing value patterns in the data
"""
import numpy as np
import pandas as pd


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
