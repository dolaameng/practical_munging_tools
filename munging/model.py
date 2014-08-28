import numpy as np
import pandas as pd  
from sklearn.base import clone 
from sklearn.cross_validation import KFold

class BiClassModelBlender(object):
	"""
	Idea credited to https://github.com/emanuele/kaggle_pbr/blob/master/blend.py
	"""
	def __init__(self, feature_names, target_name, models, blender, 
				target_value_index = 1, n_folds = 5):
		self.feature_names = feature_names
		self.target_name = target_name
		self.models = models 
		self.blender = clone(blender)
		self.target_value_index = target_value_index
		self.n_folds = n_folds
		self.cv_models = None 
	def fit(self, data):
		## train individual models for each cross validation fold
		kfold = KFold(data.shape[0], n_folds = self.n_folds)
		X, y = data.loc[:, self.feature_names], data.loc[:, self.target_name]
		self.cv_models = []
		blender_X = np.zeros((data.shape[0], len(self.models)))
		for imodel, template in enumerate(self.models):
			self.cv_models.append([])
			for ifold, (train_index, test_index) in enumerate(kfold):
				train_X, train_y = X.iloc[train_index, :], y.iloc[train_index]
				test_X, test_y = X.iloc[test_index, :], y.iloc[test_index]
				model = clone(template).fit(train_X, train_y)
				self.cv_models[imodel].append(model)
				blender_X[test_index, imodel] = model.predict_proba(test_X)[:, self.target_value_index]
		## train blender model for weights of different models
		self.blender.fit(blender_X, y)
		return self 
	def predict(self, data):
		X = data.loc[:, self.feature_names]
		n_models = len(self.cv_models)
		blender_X = np.zeros( (X.shape[0], n_models) )
		for imodel in xrange(n_models):
			cvhat = np.asarray([self.cv_models[imodel][ifold].predict_proba(X)[:, self.target_value_index]
								for ifold in xrange(len(self.cv_models[imodel]))]).mean(axis = 0)
			blender_X[:, imodel] = cvhat
		yhat = self.blender.predict_proba(blender_X)[:, self.target_value_index]
		return yhat