"""
Different measures of performance calibration
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def biclassification_density_plot(y, yhat, ax = None, y_name="", yhat_name=""):
	"""
	Density plot of yhat (posterior probablity) for different classes in y 
	y: true class labels (Series)
	yhat: (Series) predicted posterior probability (or predicted labels)
	ax : the plotting ax
	y_name: the name for y
	yhat_name: the name for yhat
	"""
	df = pd.DataFrame(data = np.c_[y, yhat], columns = ["truth", "prediction"])
	if ax is None:
		fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
	for v, sdf in df.groupby("truth"):
		sdf.prediction.plot(kind = "density", ax = ax, label = "%s=%s"%(y_name,v))
	ax.set_xlabel("Prediction Prob by %s" % yhat_name)
	ax.legend(loc="best")

def biclassification_likelihood_score(y, yhat, y_positive, y_name="", yhat_name=""):
	"""
	Score calibrates the usefulness of a prediction in terms of likelihood, combined 
	with feature.extract_cprobs_by_biclass, it can be used to measure the usefulness of 
	a single variable in binary classification problem.
	y: the series of true label class 
	yhat: the series of predicted posterior probability 
	y_positive: positive value for y, and so yhat corresponds to the posterior probability of y==y_positive
	y_name: the name of y 
	yhat_name: the name of yhat  
	"""
	prior_p = np.mean(y == y_positive)
	base_loglikelihood = np.sum(np.log(np.where(y == y_positive, prior_p, 1-prior_p)))
	feat_loglikelihood = np.sum(np.log(np.where(y == y_positive, yhat, 1-yhat)))
	return feat_loglikelihood - base_loglikelihood