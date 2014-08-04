import numpy as np


def convert_features_to_dtype(df, fnames, totype = np.str, copy = True):
	"""
	Convert certain features to specific dtype - totype. Its similiar to pd.Categorical 
	but more general in the sense that the feature values don't need to be encoded as integers 
	from 0.
	df: DataFrame
	fnames: names of features to Convert
	totype: data type (defined in numpy) to convert to
	e.g., it is usually useful to convert certain categorical features 
	with np.bool and np.int to np.object, this normalization is convienent to 
	missing values imputation task.
	As a side-effect, the np.nan will be converted to "nan" in these features
	"""
	result = df.copy() if copy else df
	result.loc[:, fnames] = result.loc[:, fnames].astype(totype)
	return result