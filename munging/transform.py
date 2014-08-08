class FeatureRemover(object):
	def __init__(self, feat_names, copy = True):
		self.feat_names = feat_names
		self.copy = copy
	def fit(self, df):
		return self
	def transform(self, df):
		result = df.copy() if self.copy else df
		result.drop(labels = self.feat_names, axis = 1, inplace = True)
		return result