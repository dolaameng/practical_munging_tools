"""
Transformation of features in df, including,
1. log transformation for positive skewed data 
1. sign-log transformation for real value skewed data 

References: 
1. "Practical Data Science with R" by Nina Zumel and John Mount, Chapter 3, 4
"""

def log_transform(df, feat_names):
	"""
	Transform data using log, specially for data generated by mulplicative process.
	The data to be transformed must be POSITIVE, otherwise remove the bad data first 
	or use signed_log_transform instead.
	df: DataFrame
	feat_names: names of features to be transformed. 
	"""
	pass