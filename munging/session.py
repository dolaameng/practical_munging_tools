"""
Create a data analysis session by maintaining train and test data information.


Most of the functions in munging package will deal with a Session object.
"""


class Session():
	"""
	Data Analysis Session - manage all data, intermediate results
	and models. Specially it needs to record what has been done?
	"""
	def __init__(self, data_frame, target_name, 
		copy = True, validation_frac = 0.25):
		pass
	################ Data Exploration / Inspection ##############
	def inspect(self, method):
		pass
	################ Missing Values / Outliers Handling #########
	################ Feature Extraction / Ranking ###############
	################ Build Models ###############################
	################ Calibrate Model Performances ###############
	################ Make Predictions ###########################
