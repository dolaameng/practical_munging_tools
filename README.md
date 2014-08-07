practical_munging_tools
=======================

Python implementation of a set of tips collected for practical data munging. The main purpose of the package is to facilitate practical tasks in data munging, such as missing value imputation, data transfromation, feature ranking and etc. 

The purpose of the library is to implement the best practice in DA in easier way in practice. For example, multiple feature encoding/extraction methods exist (feature engineering). And the most features depend on (1) the relationship between input and output to be discovered (2) the model to be used to find the relationship. If the relationship between input and output are piecewise log, then a tree/forest model will be straightforward to find it. But for a "functional model" such as regression model, this kind relationship is easier to find if we transform input into its log-form. However, there are some patterns of using different methods for different natures of data, via inspecting the data in certain ways.

It interfaces with pandas DataFrame.

# Source of Tips:
1. "Practical Data Science with R" by Nina Zumel and John Mount
2. "Machine Learning: The Art and Science of Algorithms that Make Sense of Data Paperback" by Peter Flach
3. "Applied Predictive Modeling" by Max Kuhn and Kjell Johnson

# Notes & Thoughts:
1. "Good" feature transformation depend on (1) relation between inputs and outputs and (2) the model to be used (e.g., tree model and linear regression model). Some models are very insenstive to feature transformations (e.g. most tree models).
2. Good feature transformation/extraction are key to successful predictive models
3. Most of time there is a natural order of applying different transformation steps. 
4. Most feature transformations are needed because some models need them or other transformation steps need them. And because of that, most of time there is an order of applying those transformations in data (e.g., skewness_removal -> center_scaling -> etc. -> na_imputation). Feature Transformations include, 
	3.1 Normalization (centered by mean, scaled by std) - it is usually required by certain models that are sensitive to scales of different features (e.g., those based on Euclidean distances)
	3.2 Skewness Removal (e.g., BoxCox) - the main reasons are (1) to alleviate the influence of "outliers" on some models (e.g. linear regression) and (2) to make a nonlinear relation more linear (if output is symmetric, so to be linearly correlated, the input must be kinda symmetric). In practice it can be detected as max/min >= 20 in feature values 
	3.3 Outlier Removal - First make sure those that are far from main clusters are REAL outliers instead of just normal sampling of a skewed distribution or another emerging population (when getting more data) or encoding for missing values. Obvious outliers also include values that are outside normal ranges (e.g. negative ages and etc). If you are very confident that it is a REAL outlier, just apply spatial sign (spherical normalization) or cosine distance or use models that are resistant (e.g, tree models, SVC and etc...).
	However, before using spatial sign (spherical projection), the data must be centered and scaled beforehand, otherwise large-value features will dominate.
	3.4 PCA for "reduction" of dimensionality among highly correlated features. Another reason for PCA is that it creates uncorrelated features, which might be a necessary assumption for some models (again e.g., linear regression when explaining features by their coeffcients is important or for numerical stability, i.e., variance deduction). Centering & Scaling (and sometime skewness/outlier removal) is necessary before PCA, because it is based on the estimate of data variance. Visual inspection of PCA result is a critical step to check data quality and gain intuition (but beware not to over-estimate any pattern as the PCA projected variable values will naturally have different scales). 
	3.5 Missing value imputation - several ways of doing it (1) if its occurances is too high in a feature, considering simply exclude the feature (as no much information in it) (2) using masking label for categorical variable and masking_varialble + dummy_ value for numerical variable. (3) impute data by using the highly correated features (e.g., imputing by linear models) or nearest neighbors of samples (knn) (4) removing rows (data samples) with missing values is generally NOT a good idea except for early analysis (such as centering, log-transformation and etc.), as the test data may also contain NA values. HOWEVER, ITs suggested to do exploratory analysis by excluding missing values first (except missing value pattern analsysis)
	3.6 Removing non-informative/redudant features - (1) near-zero variance features (2) highly correlated features. This is both for reduction of computation cost and improvement of stability.
		- detection of near-zero variance features - (a) #_unique_values/sample_size is low <= 0.1, (b) #_most_freq_value/#_second_most_freq_value is large >= 20) - NOT applicable to numerical variables.
		- find clusters (components) of correlated features - either by visually or by PCA (# of significant components corresponds to # of correlated clusters, and the large coefficients indicates involved features.)
		- algorithm to remove colinearity in features:
	3.7 Some reference[1] recommends manual categorizing numerical values, whereas others [3] argues that it is not smart to do so, instead of using a more sphosticated model.
