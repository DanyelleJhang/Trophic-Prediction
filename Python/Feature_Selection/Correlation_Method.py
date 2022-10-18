import numpy as np
import sys, os
from joblib import Parallel, delayed
call_module_for_bash = os.path.dirname(str(os.getcwd())) + "/Python"
sys.path.insert(1, call_module_for_bash)
from utility.usage_measurement import hardware_usage
### User's guide to correlation coefficients
### Akoglu, H. (2018). User's guide to correlation coefficients. Turkish journal of emergency medicine, 18(3), 91-93.

# Correlation Coefficient    Dancey & Reidy    Quinnipiac University     Chan YH
#                            (Psychology)      (Politics)                (Medicine)
# +1            −1           Perfect           Perfect                   Perfect
# +0.9          −0.9         Strong            Very Strong               Very Strong
# +0.8          −0.8         Strong            Very Strong               Very Strong
# +0.7          −0.7         Strong            Very Strong               Moderate
# +0.6          −0.6         Moderate          Strong                    Moderate
# +0.5          −0.5         Moderate          Strong                    Fair
# +0.4          −0.4         Moderate          Strong                    Fair
# +0.3          −0.3         Weak              Moderate                  Fair
# +0.2          −0.2         Weak              Weak                      Poor
# +0.1          −0.1         Weak              Negligible                Poor
# 0              0           Zero              None                      None

# The coefficients designed for this purpose are Spearman's rho (denoted as rs) and Kendall's Tau. 
# In fact, normality is essential for the calculation of the significance and confidence intervals, not the correlation coefficient itself. 
# Kendall's tau is an extension of Spearman's rho. 
# It should be used when the same rank is repeated too many times in a small dataset. 
# Some authors suggest that Kendall's tau may draw more accurate generalizations compared to Spearman's rho in the population.
def singleColsingleLabelFeatureCorreltionColIndex(input_X_array,column_index,input_y_1Darray,correlation_test,pvalue):
	if correlation_test ==  "Pearson":
		from scipy.stats import pearsonr
		corr_test = pearsonr
	elif correlation_test ==  "Spearman":
		from scipy.stats import spearmanr
		corr_test = spearmanr
	elif correlation_test ==  "KendallTau":
		from scipy.stats import kendalltau
		corr_test = kendalltau
	elif correlation_test ==  "PointBiserial":
		from scipy.stats import pointbiserialr
		corr_test = pointbiserialr
	_,p_value = corr_test(input_y_1Darray,input_X_array[:,column_index])
	if p_value < pvalue:
		return column_index
	else:
		return np.nan

def singleLabelFeatureCorreltionColIndex(input_X_array,input_y_1Darray,correlation_test,pvalue,n_jobs):
	col_remain_index = Parallel(n_jobs=n_jobs)(delayed(singleColsingleLabelFeatureCorreltionColIndex)(input_X_array=input_X_array,
																									  column_index=col_index,
																									  input_y_1Darray=input_y_1Darray,
																									  correlation_test=correlation_test,
																									  pvalue=pvalue) for col_index in range(input_X_array.shape[1])
										  )
	col_remain_index = [item for item in col_remain_index if str(item) != 'nan']
	return col_remain_index

@hardware_usage
def multiLabelFeatureCorreltionColIndex(input_X_array,input_y_array,input_y_name,correlation_test,pvalue,n_jobs):
	multi_colidnex_list = Parallel(n_jobs=n_jobs)(delayed(singleLabelFeatureCorreltionColIndex)(input_X_array=input_X_array,
																								input_y_1Darray=input_y_array[:,i],
																								correlation_test=correlation_test,
																								pvalue=pvalue,
																								n_jobs=n_jobs) for i in range(len(input_y_name)))
	multiLbaelInfoIndex = {}
	for name,index_num in zip(input_y_name,multi_colidnex_list):
		multiLbaelInfoIndex[name] = index_num
	return multiLbaelInfoIndex


@hardware_usage
def FeatureFeatureCorrelationColIndex(input_array,correlation_test,relevance_threshold):
	from pandas import DataFrame
	from itertools import combinations
	
	main_dataframe = DataFrame(input_array).astype(float)
	if main_dataframe.shape[1] > 1:
		if correlation_test ==  "Pearson":
			coef_dataframe = main_dataframe.corr(method="pearson")
		elif correlation_test ==  "Spearman":
			coef_dataframe = main_dataframe.corr(method="spearman")
		elif correlation_test ==  "KendallTau":
			coef_dataframe = main_dataframe.corr(method="kendall")
		remain_col_index = ()

		for i_1,i_2 in list(combinations(coef_dataframe,2)):
			coef = coef_dataframe.loc[i_1,i_2]
			if abs(coef) < abs(relevance_threshold):
				remain_col_index += (i_1,i_2)
		feature_index = list(set(remain_col_index))  
		return feature_index
	else:
		return [0]

