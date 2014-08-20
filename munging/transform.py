import numpy as np
import pandas as pd  

class FeatureImputer(object):
	def __init__(self, feature_names_to_types):
		self.feature_names_to_types = feature_names_to_types
		self.imputed_values = None
	def fit(self, data):
		self.imputed_values = {}
		for fname, ftype in self.feature_names_to_types.items():
			if ftype == 'categorical':
				self.imputed_values[fname] = "MISSING"
			elif ftype == 'numerical':
				self.imputed_values[fname] = data[fname].mean()
		return self 
	def transform(self, data):
		impute_suffix = "%s_IMPUTED"
		isimpute_suffix = "%s_IS_IMPUTED"
		for fname, ftype in self.feature_names_to_types.items():
			if ftype == 'categorical':
				imputed_values = data[fname].replace({np.nan: self.imputed_values[fname]})
				data[impute_suffix % fname] = imputed_values
			elif ftype == 'numerical':
				imputed_values = data[fname].replace({np.nan: self.imputed_values[fname]})
				data[impute_suffix % fname] = imputed_values
				data[isimpute_suffix % fname] = pd.isnull(data[fname])
		return data 