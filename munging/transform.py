import numpy as np
import pandas as pd  
from sklearn.preprocessing import StandardScaler

class TransformPipeline(object):
	def __init__(self, transformers):
		self.transformers = transformers
	def fit(self, data):
		for tr in self.transformers:
			tr.fit(data)
		return self
	def transform(self, data):
		transformed_data = data
		for tr in self.transformers:
			transformed_data = tr.transform(transformed_data) 
		return transformed_data

class FeatureRemover(object):
	def __init__(self, feature_names):
		self.removed_features = feature_names
	def fit(self, data):
		return self 
	def transform(self, data):
		removed_features = np.intersect1d(np.asarray(data.columns), self.removed_features)
		return data.drop(labels = removed_features, axis = 1)

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

class NumericalFeatureEvenizer(object):
	def __init__(self, feature_names_to_transforms):
		self.feature_names_to_transforms = feature_names_to_transforms
		self.transforms = {
			 'log': NumericalFeatureEvenizer.log
			, 'log_plus1': NumericalFeatureEvenizer.log_plus1
			, 'arcsinh': NumericalFeatureEvenizer.arcsinh
			, 'signed_log': NumericalFeatureEvenizer.signed_log 
		}
		self.transform_suffixs = {
			  'log': "%s_LOG"
			, 'log_plus1': "%s_LOG1"
			, 'arcsinh': "%s_ARCSINH"
			, 'signed_log': "%s_SIGNEDLOG"
		}
	def fit(self, data):
		return self 
	def transform(self, data):
		for fname, ftrans in self.feature_names_to_transforms.items():
			data[self.transform_suffixs[ftrans] % fname] = self.transforms[ftrans](data[fname])
		return data 
	@staticmethod 
	def log(xs):
		return np.log(xs)
	@staticmethod
	def log_plus1(xs):
		return np.log(xs + 1.)
	@staticmethod
	def arcsinh(xs):
		return np.arcsinh(xs)
	@staticmethod
	def signed_log(xs):
		return np.where(np.abs(xs) <= 1, 0., np.sign(xs)*np.log(np.abs(xs)))

class NumericalFeatureWhitener(object):
	def __init__(self, feature_names):
		self.feature_names = feature_names
		self.scaler = None 
	def fit(self, data):
		self.scaler = StandardScaler(copy = True, with_mean = True, with_std = True)
		self.scaler.fit(np.asarray(data.loc[:, self.feature_names]))
		return self 
	def transform(self, data):
		WHITE_SUFFIX = "%s_WHITE"
		scaled_data = self.scaler.transform(np.asarray(data.loc[:, self.feature_names]))
		scaled_data = pd.DataFrame(data = scaled_data, 
			columns = [WHITE_SUFFIX%f for f in self.feature_names], 
			index = data.index)
		data = pd.concat([data, scaled_data], axis = 1)
		return data

class CategoricalFeatureNumerizer(object):
	def __init__(self, feature_names, target_name):
		self.feature_names = feature_names
		self.target_name = target_name
		self.suffix = "%s_%s_NUMERIZED"
		self.lookups = None 
	def fit(self, data):
		self.lookups = {}
		for f in self.feature_names:
			table = pd.crosstab(columns = data[f], index = data[self.target_name],
								margins = True, dropna = False)
			table = table * 1. / table.iloc[-1, :] ## normalize 
			fvalue = table.iloc[:-2, :].T ## -1 ALL, -2 the last value=1-others
			fvalue.rename(columns = {c: self.suffix % (f, c) 
										for c in fvalue.columns}, inplace=True) ## change_name
			fvalue.reset_index(level = 0, inplace = True) # reset index as index column
			fvalue.rename(columns = {"index": f}, inplace = True)
			self.lookups[f] = fvalue
	def transform(self, data):
		for f in self.feature_names:
			#data = data.join(self.lookups[f], on = f, how = "left", rsuffix="!")
			data = pd.merge(data, self.lookups[f], on = f, how = "left", copy = False)
		data = data.fillna(0.0)
		return data