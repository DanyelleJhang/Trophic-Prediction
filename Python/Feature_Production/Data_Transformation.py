import numpy as np
def LogTransform(input_X_array,
					base="e"):
	if base=="e":
		log_base= np.log10(np.e)
	else:
		log_base= np.log10(base)
	transformed_array = np.log10(input_X_array.astype(float))/log_base
	transformed_array = np.nan_to_num(transformed_array, nan=0.0, posinf=None, neginf=0.0)
	return transformed_array


def ScalingTransform(input_X_array,
						method,
						par_norm='l2',
						par_feature_range=(-1,1),
						par_method='yeo-johnson',
						par_quantile_range=(25.0,75.0)):
	if method=="MaxAbsScaler":
		from sklearn.preprocessing import MaxAbsScaler
		scaler = MaxAbsScaler()
	if method=="Normalizer":
		'''
		norm: {‘l1’, ‘l2’, ‘max’}, default=’l2’
		The norm to use to normalize each non zero sample. 
		If norm=’max’ is used, values will be rescaled by the maximum of the absolute values.
		'''
		from sklearn.preprocessing import Normalizer
		scaler = Normalizer(norm=par_norm)
	if method=="MinMaxScaler":
		from sklearn.preprocessing import MinMaxScaler
		'''
		feature_range: (min, max), default=(-1, 1)
		Desired range of transformed data.
		'''
		scaler = MinMaxScaler(feature_range=par_feature_range)
	if method=="PowerTransformer":
		from sklearn.preprocessing import PowerTransformer
		'''
		method: {‘yeo-johnson’, ‘box-cox’}, default=’yeo-johnson’
		The power transform method. Available methods are:
			‘yeo-johnson’ [1], works with positive and negative values
			‘box-cox’ [2], only works with strictly positive values
			[1] I.K. Yeo and R.A. Johnson, “A new family of power transformations to improve normality or symmetry.” Biometrika, 87(4), pp.954-959, (2000).
			[2] G.E.P. Box and D.R. Cox, “An Analysis of Transformations”, Journal of the Royal Statistical Society B, 26, 211-252 (1964).
		'''
		scaler = PowerTransformer(method=par_method)
	if method=="RobustScaler":
		'''
		quantile_range: tuple(q_min, q_max), 0.0 < q_min < q_max < 100.0, default=(25.0, 75.0)
		Quantile range used to calculate scale_. 
		By default this is equal to the IQR, i.e., q_min is the first quantile and q_max is the third quantile.
		'''
		from sklearn.preprocessing import RobustScaler
		scaler = RobustScaler(quantile_range=par_quantile_range)
	if method=="StandardScaler":
		from sklearn.preprocessing import StandardScaler
		scaler = StandardScaler()
	transformed_array = scaler.fit_transform(input_X_array)
	return transformed_array

def DiscreteTransform(input_X_array,
						N_bins,
						par_encode='ordinal',
						par_strategy='uniform'):
	# Bin continuous data into intervals.
	# http://www.taroballz.com/2019/06/09/ML_continueVar_Preprocessing/
	'''
	N_bins: int or array-like of shape (n_features,)
		The number of bins to produce
	
	encode: {‘onehot’, ‘onehot-dense’, ‘ordinal’}
		Method used to encode the transformed result
	1) onehot: Encode the transformed result with one-hot encoding and return a sparse matrix. Ignored features are always stacked to the right.
	2) onehot-dense: Encode the transformed result with one-hot encoding and return a dense array. Ignored features are always stacked to the right.
	3) ordinal: Return the bin identifier encoded as an integer value.
	
	strategy: {‘uniform’, ‘quantile’, ‘kmeans’}
		Strategy used to define the widths of the bins.
	1) uniform: All bins in each feature have identical widths.
	2) quantile: All bins in each feature have the same number of points.
	3) kmeans: Values in each bin have the same nearest center of a 1D k-means cluster.
	'''
	from sklearn.preprocessing import KBinsDiscretizer
	
	est = KBinsDiscretizer(n_bins=N_bins,encode=par_encode, strategy=par_strategy) 
	transformed_array = est.fit_transform(input_X_array)
	return transformed_array