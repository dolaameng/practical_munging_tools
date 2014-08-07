import numpy as np

#NUMERICAL_FEAT_DTYPES = np.array([np.float, np.int])
#CATEGORICAL_FEAT_DTYPES = np.array([np.bool, np.object])


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

def is_numerical(df, fname):
	return df[fname].dtype in np.array([np.float, np.int, np.double])

def is_categorical(df, fname):
	return df[fname].dtype in np.array([np.bool, np.object])

def is_continuous(df, fname):
	if df[fname].dtype in np.array([np.float, np.double]):
		return True
	elif df[fname].dtype in np.array([np.int]):
		return len(np.unique(df[fname])) > 10
	else:
		return False

def is_discrete(df, fname):
	if df[fname].dtype in np.array([np.bool, np.object]):
		return True
	elif df[fname].dtype in np.array([np.int]):
		return len(np.unique(df[fname])) <= 10
	else:
		return False


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