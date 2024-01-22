## Source From:
# https://github.com/Edderic/causal_discovery/blob/cf578822be8eaee529bb1652c03e3107be827301/information_theory.py
import sys, os
call_module_for_bash = os.path.dirname(str(os.getcwd())) + "/Python"
sys.path.insert(1, call_module_for_bash)
from utility.usage_measurement import hardware_usage
"""
	Information Theory
	------------------
	Provides Information Theory helpers.
	Available functions
	-------------------
	- entropy
	- conditional_entropy
	- conditional_mutual_information
	- multinomial_normalizing_sum
	- regret
	- sci_is_independent
"""

# import cupy as cp
# def entropy(x):

#     """
#     source from : https://stats.stackexchange.com/questions/346137/how-to-compute-joint-entropy-of-high-dimensional-data
#     x refer to high dimentioanl array
#     [[... , ... , ...],
#      [... , ... , ...]], 
#      [... , ... , ...]],
#      ..., 
#      [... , ... , ...]]]
#     """
#     counts = cp.histogramdd(x)[0]
#     dist = counts / cp.sum(counts)
#     logs = cp.log2(cp.where(dist > 0, dist, 1))
#     return -cp.sum(dist * logs)

# def conditional_mutual_information(x, y, z):
#     """
#     I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
#     """
#     xz = cp.column_stack((x,z))
#     yz = cp.column_stack((y,z))
#     xyz = cp.column_stack((x,y,z))

#     return entropy(xz) + entropy(yz) - entropy(xyz) - entropy(z) 


# def mutual_information(x, y):
#     """
#     I(X;Y) = H(X) + H(Y) - H(X,Y)
#     """
#     xy = cp.column_stack((x,y))
#     return entropy(x) + entropy(y) - entropy(xy)

import numpy as np
def entropy(data, variables=[], base_2=False):
	"""
		Computes Shannon entropy.
		Parameters:
			data : pandas.DataFrame
				A dataframe where variables are columns.
			variables: list[str]
				A list of variable names to include in the entropy calculation.
		Examples:
			Say that X is multinomially distributed with 4 classes, and they are
			uniformly distributed. The Shannon entropy is:
			- 4 * (0.25 * np.log2(0.25)) = -1 * np.log2(0.25)
										 = -1 * -2
										 = 2
			>>> from pytest import approx
			>>> size = 10000
			>>> multinomials = np.random.multinomial(
			>>>         n=1,
			>>>         pvals=[0.25,
			>>>             0.25,
			>>>             0.25,
			>>>             0.25],
			>>>         size=size)
			>>> x = multinomials[:, 0] \
			>>>         + multinomials[:, 1] * 2 \
			>>>         + multinomials[:, 2] * 3 \
			>>>         + multinomials[:, 3] * 4
			>>> df = pd.DataFrame({'x': x})
			>>> calc = EntropyCalculator(data=df, variables=['x'])
			>>> assert calc.calculate() == approx(2, abs=0.01)
	"""

	data = data.copy()
	count_col_name = 'tmp. count'
	data[count_col_name] = 0
	total_count = data.shape[0]

	assert len(variables) > 0
	variable_counts = data.groupby(list(variables)).count()
	probas = variable_counts / total_count

	if base_2:
		log_func = np.log2
	else:
		log_func = np.log

	return -(probas * log_func(probas)).sum()[count_col_name]

def conditional_entropy(data, variables=[], conditioning_set=[], base_2=False):
	"""
		H(X | Y) = H(X,Y) - H(Y)
		H(variables | conditioning_set) = H(variables,conditioning_set) - H(conditioning_set)
		Computes H(X | Y) = H(X,Y) - H(Y) where Y is the conditioning_set and X
		and Y are the variables.
		Parameters:
			data : pandas.DataFrame
				A dataframe where variables are columns.
			variables: list[str]
				A list of variable names.
			conditioning_set: list[str]. Defaults to empty list.
				A list of variable names that are being conditioned on. If the
				conditionals is not empty, then we'll be computing conditional
				entropy.
				If conditioning_set is an empty list, this function returns the
				entropy for the set of variables (i.e. instead of computing
				H(X|Y), it'll return H(X), the entropy of X).
		Examples:
			Say there's a variable X and Y and they are independent. X and Y are
			multinomial variables with 4 possible values:
			>>> from pytest import approx
			>>> size = 10000
			>>> x = np.random.multinomial(
			>>>         n=1,
			>>>         pvals=[0.25,
			>>>             0.25,
			>>>             0.25,
			>>>             0.25],
			>>>         size=size)
			>>>
			>>> x = x[:, 0] \
			>>>         + x[:, 1] * 2 \
			>>>         + x[:, 2] * 3 \
			>>>         + x[:, 3] * 4
			>>>
			>>> y = np.random.multinomial(
			>>>         n=1,
			>>>         pvals=[0.25,
			>>>             0.25,
			>>>             0.25,
			>>>             0.25],
			>>>         size=size)
			>>>
			>>> y = y[:, 0] \
			>>>         + y[:, 1] * 2 \
			>>>         + y[:, 2] * 3 \
			>>>         + y[:, 3] * 4
			>>>
			>>> df_2_multinomial_indep_RVs = pd.DataFrame({'x': x, 'y': y})
			# Entropy of X, without conditioning on anything, is 2. Conditional
			# entropy of X given Y is still 2. In other words, knowing about Y
			# doesn't change the entropy (i.e. the uncertainty) on X.
			# Therefore X and Y are independent.
			>>> assert conditional_entropy(
			>>>     data=df_2_multinomial_indep_RVs,
			>>>     variables=['x', 'y'],
			>>>     conditioning_set=['y']
			>>> ) == approx(2, abs=0.01)
	"""
	assert len(set(variables)) > 0

	if len(conditioning_set) == 0:
		return entropy(data=data, variables=variables, base_2=base_2)

	vars_and_conditioning_set = \
		list(set(variables).union(set(conditioning_set)))

	return entropy(
			   data=data,
			   variables=vars_and_conditioning_set,
			   base_2=base_2
		   ) - entropy(
			   data=data,
			   variables=conditioning_set,
			   base_2=base_2
		   )

def conditional_mutual_information(data, vars_1, vars_2, conditioning_set=[], base_2=False):
	"""
		I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
		I(vars_1;vars_2|conditioning_set) = H(vars_1|conditioning_set) - H(vars_1|vars_2,conditioning_set)
		Computes I(X;Y|Z) = H(X|Z) - H(X|Y,Z). Essentially, this tells us
		whether or not Y tells us something about X, after we've known about
		Z. In the large sample limit, if Y is independent of X given Z, then
		conditional mutual information I(X;Y|Z) is 0. However, with finite
		samples, even if in the true generating process, X and Y are
		independent given Z, it's very possible that the conditional mutual
		information is greater than 0.
		Parameters:
			data: pandas.DataFrame
			vars_1: list[str]
				Represents X in I(X;Y|Z).
			vars_2: list[str]
				Represents Y in I(X;Y|Z).
			conditioning_set: list[str]. Defaults to empty list.
				Represents Z in I(X;Y|Z).
				If conditioning_set is empty, this computes mutual information:
				I(X;Y) = H(X) - H(X|Y).
		Examples:
			Ex 1: Say there's a variable X and Y and they are independent. X
			and Y are multinomial variables with 4 possible values:
			>>> from pytest import approx
			>>> size = 10000
			>>> x = np.random.multinomial(
			>>>         n=1,
			>>>         pvals=[0.25,
			>>>             0.25,
			>>>             0.25,
			>>>             0.25],
			>>>         size=size)
			>>>
			>>> x = x[:, 0] \
			>>>         + x[:, 1] * 2 \
			>>>         + x[:, 2] * 3 \
			>>>         + x[:, 3] * 4
			>>>
			>>> y = np.random.multinomial(
			>>>         n=1,
			>>>         pvals=[0.25,
			>>>             0.25,
			>>>             0.25,
			>>>             0.25],
			>>>         size=size)
			>>>
			>>> y = y[:, 0] \
			>>>         + y[:, 1] * 2 \
			>>>         + y[:, 2] * 3 \
			>>>         + y[:, 3] * 4
			>>>
			>>> df_2_multinomial_indep_RVs = pd.DataFrame({'x': x, 'y': y})
			>>>
			>>> assert conditional_mutual_entropy(
			>>>     data=df_2_multinomial_indep_RVs,
			>>>     variables=['x', 'y'],
			>>>     conditioning_set=[]
			>>> ) == approx(0, abs=0.01)
			Ex 2: Z causes X and Y:
			>>> from pytest import approx
			>>> size = 10000
			>>> z = np.random.multinomial(
			>>>         n=1,
			>>>         pvals=[0.25,
			>>>             0.25,
			>>>             0.25,
			>>>             0.25],
			>>>         size=size)
			>>>
			>>> z = z[:, 0] \
			>>>         + z[:, 1] * 2 \
			>>>         + z[:, 2] * 3 \
			>>>         + z[:, 3] * 4
			>>> y = (z == 2)
			>>> x = (z == 1)
			>>>
			>>> df_2_multinomial_indep_RVs = pd.DataFrame(
			>>>     {'x': x, 'y': y, 'z': z}
			>>> )
			>>>
			>>> assert conditional_mutual_entropy(
			>>>     data=df_2_multinomial_indep_RVs,
			>>>     variables=['x', 'y'],
			>>>     conditioning_set=['z']
			>>> ) == approx(0, abs=0.01)
	"""
	return conditional_entropy(
		data=data,
		variables=vars_1,
		conditioning_set=conditioning_set,
		base_2=base_2
	) - conditional_entropy(
		data=data,
		variables=vars_1,
		conditioning_set=list(set(conditioning_set).union(vars_2)),
		base_2=base_2
	)

# Mutual information
def mutual_information(data, vars_1, vars_2, base_2=False):
	'''
	I(vars_1;vars_1) = H(vars_1)−H(vars_1|vars_2)
	I(X;Y)=H(X)−H(X|Y)
	I(X_1;X_2;...;X_n) = H(X_1;X_2;...;X_n-1)−H(X_1;X_2;...;X_n-1|X_n)
	'''
	return entropy(data, variables=vars_1, base_2=False) - conditional_entropy(data, variables=vars_1, conditioning_set=vars_2, base_2=False)


##### Inofrmation Change Ratio
def DualFeatureInofrmationChangeRatio(entire_dataframe,feature_1,feature_2,single_label):
	#### reference: Feature-specific mutual information variation for multi-label feature selection
	#### https://doi.org/10.1016/j.ins.2022.02.024
	#### Hu, L., Gao, L., Li, Y., Zhang, P., & Gao, W. (2022). Feature-specific mutual information variation for multi-label feature selection. Information Sciences, 593, 449-471.
	
	# I(f_2;y_1|f_1)/I(f_2;f_1) > 1
	# => f_1 has a positive effect on the amount of information.

	# I(f_2;y_1|f_1)/I(f_2;f_1) == 1
	# => f_1 and f_2 are independent

	# 0=< I(f_2;y_1|f_1)/I(f_2;f_1) < 1
	# => f_1 has a negative effect on the amount of information.
	#### Targeted information ratio: x_j
	feature_1_mutual_information = mutual_information(entire_dataframe,[feature_1],[single_label], base_2=True)
	if feature_1_mutual_information in [0,np.nan]:
		feature_1_information_change_ratio = 0
	else:
		feature_1_conditional_mutual_information = conditional_mutual_information(entire_dataframe, [feature_2], [single_label], conditioning_set=[feature_1], base_2=True)
		feature_1_information_change_ratio = feature_1_conditional_mutual_information/feature_1_mutual_information


	#### Targeted information ratio: x_k
	feature_2_mutual_information = mutual_information(entire_dataframe,[feature_2],[single_label], base_2=True)
	if feature_2_mutual_information in [0,np.nan]:
		feature_2_information_change_ratio = 0
	else:
		feature_2_conditional_mutual_information = conditional_mutual_information(entire_dataframe, [feature_1], [single_label], conditioning_set=[feature_2], base_2=True)
		feature_2_information_change_ratio = feature_2_conditional_mutual_information/feature_2_mutual_information
	#
	return feature_1_information_change_ratio,feature_2_information_change_ratio

def singleLabelInformativeColIndex(input_X_array,input_y_1Darray):
	from pandas import DataFrame,concat
	y_dataframe = DataFrame(input_y_1Darray,columns=["single_label"])
	X_dataframe= DataFrame(input_X_array)
	yx_dataframe = concat([y_dataframe,X_dataframe],axis=1)
	
	from itertools import combinations
	select_feature_list = []
	remove_feature_list = [] 
	'''
	we take information between each label and features independently
	if there is an informative feature to one label (ChangeRatio>1), such feature will be remain due to multi-output classification methods
	'''
	for f_1,f_2 in combinations(list(X_dataframe.columns),2):
		if (f_1 and f_2) not in remove_feature_list:
			f_1_ratio,f_2_ratio = DualFeatureInofrmationChangeRatio(entire_dataframe=yx_dataframe,feature_1=f_1,feature_2=f_2,single_label="single_label")
			if (f_1_ratio >1) and (f_2_ratio >1):
				select_feature_list.append(f_1)
				select_feature_list.append(f_2)
				#print("Label name",y_i," ","SELECT feauture: ",f_1,":",f_1_ratio,"  ",f_2,":",f_2_ratio)
			else:
				remove_feature_list.append(f_1)
				remove_feature_list.append(f_2)
				#print("Label name",y_i,"remove feauture: ",f_1,":",f_1_ratio,"  ",f_2,":",f_2_ratio)
		else:
			continue
	feature_index = list(map(int,list(set(select_feature_list))))
	return feature_index

@hardware_usage
def multiLabelInformativeColIndex(input_X_array,input_y_array,input_y_name,n_jobs):
	from joblib import Parallel, delayed
	multi_colidnex_list = Parallel(n_jobs=n_jobs)(delayed(singleLabelInformativeColIndex)(input_y_1Darray=input_y_array[:,i],
																				   input_X_array=input_X_array
																						 ) for i in range(len(input_y_name)))
	multiLbaelInfoIndex = {}
	for name,index_num in zip(input_y_name,multi_colidnex_list):
		multiLbaelInfoIndex[name] = index_num
	return multiLbaelInfoIndex




##### GPU acceleration


# import cupy as cp
# def entropy(data, variables=[], base_2=False):
# 	"""
# 		Computes Shannon entropy.
# 		Parameters:
# 			data : pandas.DataFrame
# 				A dataframe where variables are columns.
# 			variables: list[str]
# 				A list of variable names to include in the entropy calculation.
# 		Examples:
# 			Say that X is multinomially distributed with 4 classes, and they are
# 			uniformly distributed. The Shannon entropy is:
# 			- 4 * (0.25 * np.log2(0.25)) = -1 * np.log2(0.25)
# 										 = -1 * -2
# 										 = 2
# 			>>> from pytest import approx
# 			>>> size = 10000
# 			>>> multinomials = np.random.multinomial(
# 			>>>         n=1,
# 			>>>         pvals=[0.25,
# 			>>>             0.25,
# 			>>>             0.25,
# 			>>>             0.25],
# 			>>>         size=size)
# 			>>> x = multinomials[:, 0] \
# 			>>>         + multinomials[:, 1] * 2 \
# 			>>>         + multinomials[:, 2] * 3 \
# 			>>>         + multinomials[:, 3] * 4
# 			>>> df = pd.DataFrame({'x': x})
# 			>>> calc = EntropyCalculator(data=df, variables=['x'])
# 			>>> assert calc.calculate() == approx(2, abs=0.01)
# 	"""
# 	data = data.copy()
# 	count_col_name = "temp"# data.columns[-1]+1 # rename it
# 	data[count_col_name] = 0
# 	total_count = data.shape[0]

# 	assert len(variables) > 0
# 	variable_counts = data.groupby(list(variables)).count()
# 	probas = variable_counts / total_count

# 	if base_2:
# 		log_func = cp.log2
# 	else:
# 		log_func = cp.log
# 	probas = cp.array(probas.to_numpy())
# 	# 這個寫法有顯著加快速度，透過舉證轉換，先把cuDF矩陣放到NUMPY再轉到cuPy矩陣上做運算，效果顯著
# 	return - cp.sum(probas*log_func(probas),axis=0)[-1]

# def conditional_entropy(data, variables=[], conditioning_set=[], base_2=False):
# 	"""
# 		H(X | Y) = H(X,Y) - H(Y)
# 		H(variables | conditioning_set) = H(variables,conditioning_set) - H(conditioning_set)
# 		Computes H(X | Y) = H(X,Y) - H(Y) where Y is the conditioning_set and X
# 		and Y are the variables.
# 		Parameters:
# 			data : pandas.DataFrame
# 				A dataframe where variables are columns.
# 			variables: list[str]
# 				A list of variable names.
# 			conditioning_set: list[str]. Defaults to empty list.
# 				A list of variable names that are being conditioned on. If the
# 				conditionals is not empty, then we'll be computing conditional
# 				entropy.
# 				If conditioning_set is an empty list, this function returns the
# 				entropy for the set of variables (i.e. instead of computing
# 				H(X|Y), it'll return H(X), the entropy of X).
# 		Examples:
# 			Say there's a variable X and Y and they are independent. X and Y are
# 			multinomial variables with 4 possible values:
# 			>>> from pytest import approx
# 			>>> size = 10000
# 			>>> x = np.random.multinomial(
# 			>>>         n=1,
# 			>>>         pvals=[0.25,
# 			>>>             0.25,
# 			>>>             0.25,
# 			>>>             0.25],
# 			>>>         size=size)
# 			>>>
# 			>>> x = x[:, 0] \
# 			>>>         + x[:, 1] * 2 \
# 			>>>         + x[:, 2] * 3 \
# 			>>>         + x[:, 3] * 4
# 			>>>
# 			>>> y = np.random.multinomial(
# 			>>>         n=1,
# 			>>>         pvals=[0.25,
# 			>>>             0.25,
# 			>>>             0.25,
# 			>>>             0.25],
# 			>>>         size=size)
# 			>>>
# 			>>> y = y[:, 0] \
# 			>>>         + y[:, 1] * 2 \
# 			>>>         + y[:, 2] * 3 \
# 			>>>         + y[:, 3] * 4
# 			>>>
# 			>>> df_2_multinomial_indep_RVs = pd.DataFrame({'x': x, 'y': y})
# 			# Entropy of X, without conditioning on anything, is 2. Conditional
# 			# entropy of X given Y is still 2. In other words, knowing about Y
# 			# doesn't change the entropy (i.e. the uncertainty) on X.
# 			# Therefore X and Y are independent.
# 			>>> assert conditional_entropy(
# 			>>>     data=df_2_multinomial_indep_RVs,
# 			>>>     variables=['x', 'y'],
# 			>>>     conditioning_set=['y']
# 			>>> ) == approx(2, abs=0.01)
# 	"""
# 	assert len(set(variables)) > 0

# 	if len(conditioning_set) == 0:
# 		return entropy(data=data, variables=variables, base_2=base_2)

# 	vars_and_conditioning_set = \
# 		list(set(variables).union(set(conditioning_set)))

# 	return entropy(
# 			   data=data,
# 			   variables=vars_and_conditioning_set,
# 			   base_2=base_2
# 		   ) - entropy(
# 			   data=data,
# 			   variables=conditioning_set,
# 			   base_2=base_2
# 		   )

# def conditional_mutual_information(data, vars_1, vars_2, conditioning_set=[], base_2=False):
# 	"""
# 		I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
# 		I(vars_1;vars_2|conditioning_set) = H(vars_1|conditioning_set) - H(vars_1|vars_2,conditioning_set)
# 		Computes I(X;Y|Z) = H(X|Z) - H(X|Y,Z). Essentially, this tells us
# 		whether or not Y tells us something about X, after we've known about
# 		Z. In the large sample limit, if Y is independent of X given Z, then
# 		conditional mutual information I(X;Y|Z) is 0. However, with finite
# 		samples, even if in the true generating process, X and Y are
# 		independent given Z, it's very possible that the conditional mutual
# 		information is greater than 0.
# 		Parameters:
# 			data: pandas.DataFrame
# 			vars_1: list[str]
# 				Represents X in I(X;Y|Z).
# 			vars_2: list[str]
# 				Represents Y in I(X;Y|Z).
# 			conditioning_set: list[str]. Defaults to empty list.
# 				Represents Z in I(X;Y|Z).
# 				If conditioning_set is empty, this computes mutual information:
# 				I(X;Y) = H(X) - H(X|Y).
# 		Examples:
# 			Ex 1: Say there's a variable X and Y and they are independent. X
# 			and Y are multinomial variables with 4 possible values:
# 			>>> from pytest import approx
# 			>>> size = 10000
# 			>>> x = np.random.multinomial(
# 			>>>         n=1,
# 			>>>         pvals=[0.25,
# 			>>>             0.25,
# 			>>>             0.25,
# 			>>>             0.25],
# 			>>>         size=size)
# 			>>>
# 			>>> x = x[:, 0] \
# 			>>>         + x[:, 1] * 2 \
# 			>>>         + x[:, 2] * 3 \
# 			>>>         + x[:, 3] * 4
# 			>>>
# 			>>> y = np.random.multinomial(
# 			>>>         n=1,
# 			>>>         pvals=[0.25,
# 			>>>             0.25,
# 			>>>             0.25,
# 			>>>             0.25],
# 			>>>         size=size)
# 			>>>
# 			>>> y = y[:, 0] \
# 			>>>         + y[:, 1] * 2 \
# 			>>>         + y[:, 2] * 3 \
# 			>>>         + y[:, 3] * 4
# 			>>>
# 			>>> df_2_multinomial_indep_RVs = pd.DataFrame({'x': x, 'y': y})
# 			>>>
# 			>>> assert conditional_mutual_entropy(
# 			>>>     data=df_2_multinomial_indep_RVs,
# 			>>>     variables=['x', 'y'],
# 			>>>     conditioning_set=[]
# 			>>> ) == approx(0, abs=0.01)
# 			Ex 2: Z causes X and Y:
# 			>>> from pytest import approx
# 			>>> size = 10000
# 			>>> z = np.random.multinomial(
# 			>>>         n=1,
# 			>>>         pvals=[0.25,
# 			>>>             0.25,
# 			>>>             0.25,
# 			>>>             0.25],
# 			>>>         size=size)
# 			>>>
# 			>>> z = z[:, 0] \
# 			>>>         + z[:, 1] * 2 \
# 			>>>         + z[:, 2] * 3 \
# 			>>>         + z[:, 3] * 4
# 			>>> y = (z == 2)
# 			>>> x = (z == 1)
# 			>>>
# 			>>> df_2_multinomial_indep_RVs = pd.DataFrame(
# 			>>>     {'x': x, 'y': y, 'z': z}
# 			>>> )
# 			>>>
# 			>>> assert conditional_mutual_entropy(
# 			>>>     data=df_2_multinomial_indep_RVs,
# 			>>>     variables=['x', 'y'],
# 			>>>     conditioning_set=['z']
# 			>>> ) == approx(0, abs=0.01)
# 	"""
# 	return conditional_entropy(
# 		data=data,
# 		variables=vars_1,
# 		conditioning_set=conditioning_set,
# 		base_2=base_2
# 	) - conditional_entropy(
# 		data=data,
# 		variables=vars_1,
# 		conditioning_set=list(set(conditioning_set).union(vars_2)),
# 		base_2=base_2
# 	)

# # Mutual information
# def mutual_information(data, vars_1, vars_2, base_2=False):
# 	'''
# 	I(vars_1;vars_1) = H(vars_1)−H(vars_1|vars_2)
# 	I(X;Y)=H(X)−H(X|Y)
# 	I(X_1;X_2;...;X_n) = H(X_1;X_2;...;X_n-1)−H(X_1;X_2;...;X_n-1|X_n)
# 	'''
# 	return entropy(data, variables=vars_1, base_2=False) - conditional_entropy(data, variables=vars_1, conditioning_set=vars_2, base_2=False)


# ##### Inofrmation Change Ratio
# def DualFeatureInofrmationChangeRatio(entire_dataframe,feature_1,feature_2,single_label):
# 	#### reference: Feature-specific mutual information variation for multi-label feature selection
# 	#### https://doi.org/10.1016/j.ins.2022.02.024
# 	#### Hu, L., Gao, L., Li, Y., Zhang, P., & Gao, W. (2022). Feature-specific mutual information variation for multi-label feature selection. Information Sciences, 593, 449-471.

# 	# I(f_2;y_1|f_1)/I(f_2;y_1) > 1
# 	# => f_1 has a positive effect on the amount of information.

# 	# I(f_2;y_1|f_1)/I(f_2;y_1) == 1
# 	# => f_1 and f_2 are independent

# 	# 0=< I(f_2;y_1|f_1)/I(f_2;y_1) < 1
# 	# => f_1 has a negative effect on the amount of information.
# 	#### Targeted information ratio: x_j
# 	feature_j_mutual_information = mutual_information(entire_dataframe,feature_2,single_label, base_2=False)
# 	if feature_j_mutual_information in [0,cp.nan]:
# 		feature_j_information_change_ratio = 0
# 	else:
# 		feature_j_conditional_mutual_information = conditional_mutual_information(entire_dataframe, feature_2, single_label, conditioning_set=feature_1, base_2=True)
# 		feature_j_information_change_ratio = feature_j_conditional_mutual_information/feature_j_mutual_information
# 	#
# 	return feature_j_information_change_ratio


# from numba import jit
# from cudf import DataFrame,concat
# from joblib import Parallel, delayed

# #@jit
# def singleLabelInformativeColIndex(input_X_array,input_y_1Darray,n_jobs):
#     y_dataframe = DataFrame(input_y_1Darray.reshape(-1,1),columns=["single_label"])
#     X_dataframe= DataFrame(input_X_array)
#     yx_dataframe = concat([y_dataframe,X_dataframe],axis=1)

#     c = list(X_dataframe.columns)
#     yx_dataframe.columns = list(map(str,yx_dataframe.columns))
#     # result_list = Parallel(n_jobs=n_jobs)(delayed(DualFeatureInofrmationChangeRatio)(entire_dataframe = yx_dataframe,
#     #                                                                                  feature_1 = [str(i)],
#     #                                                                                  feature_2 = list(map(str,c[:i] + c[i+1:])),
#     #                                                                                   single_label = ["single_label"]
#     #                                                                                 ) for i in range(len(c)))
#     index_array = cp.array([])
#     for i in range(len(c)):
#         r = DualFeatureInofrmationChangeRatio(entire_dataframe = yx_dataframe,
#                                             feature_1 = [str(i)],
#                                             feature_2 = list(map(str,c[:i] + c[i+1:])),
#                                             single_label = ["single_label"])
#         if r > 1:
#             index_array = cp.append(index_array,i)
#     return index_array

