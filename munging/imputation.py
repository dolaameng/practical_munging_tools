"""
Missing Value Imputation tools.

Based on the suggestions given in reference 1, the best pratice distinguishs imputation for
(1) Categorical data: imput the na values as new label "missing"
(2) Numerical data: 
	(2.1) random missing values (usually we don't know that is the case): na as mean
	(2.2) systematic missing values (default assumption): na as 0, adding a new feature indicating missing values


"Bad values" (e.g., negative income) are very customerized, so better leave them out of the library
for now.

References: 
1. "Practical Data Science with R" by Nina Zumel and John Mount, Chapter 3, 4
"""
import inspection
import numpy as np 
import pandas as pd


def imput_categorical_features(df, feat_names, na_label = "missing", copy = True):
	"""
	Imput na values in categorical (np.object, np.bool, np.int: with small set) features
	The method replaces "na" in feat_names as na_label
	df: DataFrame
	feat_names: names of CATEGORICAL features to imput (e.g., use inspection.find_features_with_dtypes to find them)
	na_label: new label for missing values in the features 
	copy: whether to get a new copy or use the existing df 
	"""
	imputed = df.copy() if copy else df 
	replace_pattern = {f: {np.nan: na_label} for f in feat_names}
	imputed.replace(replace_pattern, inplace = True)
	return imputed

def imput_numerical_features(df, feat_names, na_value = None, copy = True):
	"""
	Impute na values in numerical features (e.g., np.float, np.int)
	The method replaces all na values in feat_names as na_value, and create a new feature for each 
	with suffix "_isna" to indicate whether it is an original 0 or missing value imputation 
	df: DataFrame
	feat_names: names of NUMERICAL features to impute (e.g. by using inspection.find_features_with_dtypes)
	na_value: values to impute for missing values 
	copy: whether to get a new copy or use the existing df
	"""
	if na_value is None:
		na_value = {f:0. for f in feat_names}
	#print na_value
	imputed = df.copy() if copy else df 
	suffix = "_isna"
	for f in feat_names:
		na_pattern = pd.isnull(imputed[f])
		if np.any(na_pattern):
			imputed[f+suffix] = na_pattern
	replace_pattern = {f: {np.nan: na_value[f]} for f in feat_names}
	imputed.replace(replace_pattern, inplace = True)
	return imputed


def imput(df, na_categorical="missing", na_numerical=None, copy = True):
	"""
	Take the guess of types of features, and impute their missing values accordingly.
	df: DataFrame
	na_categorical: value for categorical feature impution 
	na_numerical: value for numerical feature impution
	copy: whether to make a new copy of data frame 
	"""
	na_features = inspection.find_features_with_nas(df)
	categorical_features = inspection.find_categorical_features(df)
	numerical_features = inspection.find_numerical_features(df)
	imputed = df.copy() if copy else df 
	imputed = imput_categorical_features(imputed, np.intersect1d(na_features, categorical_features), 
		na_label = na_categorical,
		copy = False)
	imputed = imput_numerical_features(imputed, np.intersect1d(na_features, numerical_features), 
		na_value = na_numerical,
		copy = False)
	return imputed