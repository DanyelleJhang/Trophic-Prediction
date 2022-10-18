import numpy as np
import sys, os
from joblib import Parallel, delayed
# from time import time, strftime, gmtime
call_module_for_bash = os.path.dirname(str(os.getcwd())) + "/Python"
sys.path.insert(1, call_module_for_bash)
from utility.usage_measurement import hardware_usage
# imbalanced-learn==0.9.1


###### Distinguish Method
#### Step 1. Feautre Scaling
#### Step 2. Single Feature sperate to (Single_Feature_class_A,Single_Feature_class_B,Single_Feature_class_C,...)
#### Step 3. Undersampling[ n_iter = N or 30] (Single_Feature_class_A,Single_Feature_class_B,Single_Feature_class_C,...)
# compare difference amongst (Single_Feature_class_A,Single_Feature_class_B,Single_Feature_class_C,...) 
# in ""1"" iteration
# and do p value correction
### if N iteration shows SIGNIFICANT DIFFERENCE amongst (Single_Feature_class_A,Single_Feature_class_B,Single_Feature_class_C,...)  in different sampling 
### such Single_Feature will be reamin

###### 卡方檢定
#### Independent test
#### 檢定FEAUTRE INDEPENDENCE ???
# 兩個名義或次序變相是否獨立，列連表檢定

###### 平均數檢定
### T-test for the means of two independent samples of scores.
# 獨立樣本
# scipy.stats.ttest_ind

### T-test on TWO RELATED samples of scores, a and b.
# 成對樣本
# scipy.stats.ttest_rel

############ 無母數
##### 中位數檢定
### Wilcoxon rank-sum statistic
# 獨立樣本
# scipy.stats.ranksums

### Wilcoxon signed-rank test.
# 成對樣本
# wilcoxon(x[, y, zero_method, correction, ...])

##### 分布差異檢定
### Mann-Whitney U rank test on two independent samples.
# mannwhitneyu
# 檢定兩母體分配是某相同

### Kolmogorov-Smirnov test
# scipy.stats.kstest
# 檢定兩個經驗分布是否不同或一個經驗分布與另一個理想分布是否不同

### Kruskal Wallis Test (吳母樹)
# 檢定三母體以上分配是否相同

### Friedman檢定
# 檢定三個或以上相關母體是否相同



# statistic_based_methods = ["independent_Ttest","paired_Ttest","independent_WilcoxonRankSum","paired_WilcoxonSignedRank","MannWhitney_U_Rank","KolmogorovSmirnov"]
# correction_methods = ["bonferroni","sidak","holm-sidak","holm","simes-hochberg","hommel","fdr_bh","fdr_by","fdr_tsbh","fdr_tsbky"]
#  : one-step correction
#  : one-step correction
#  : step down method using Sidak adjustments
#  : step-down method using Bonferroni adjustments
#  : step-up method (independent)
#  : closed method based on Simes tests (non-negative)
#  : Benjamini/Hochberg (non-negative)
#  : Benjamini/Yekutieli (negative)
#  : two stage fdr correction (non-negative)
#  : two stage fdr correction (non-negative)

def UnderBalancedSampling(unbalanced_X_matrix,unbalanced_y_1Darray,random_state_i):
	from collections import Counter
	from imblearn.under_sampling import RandomUnderSampler
	class_freq = Counter(unbalanced_y_1Darray)
	balance_freq = {k: min(class_freq.values()) for k, _ in class_freq.items()}
	UnderBalanceSample = RandomUnderSampler(sampling_strategy=balance_freq,random_state=random_state_i)
	X_matrix_undersmaple, y_1Darray_undersmaple = UnderBalanceSample.fit_resample(unbalanced_X_matrix, unbalanced_y_1Darray)
	return X_matrix_undersmaple,y_1Darray_undersmaple

def singleColsingleLabelDiscriminationColIndex(unbalanced_X_matrix,column_index,unbalanced_y_1Darray,statistic_difference_test,pvalue,pvalue_correction_mehthod,n_iter):
	from collections import Counter
	if statistic_difference_test == "independent_Ttest":
		# This is a test for the null hypothesis that 2 independent samples have identical average (expected) values. 
		# This test assumes that the populations have identical variances by default.
		from scipy.stats import ttest_ind
		Difference_Test = ttest_ind
	if statistic_difference_test == "paired_Ttest":
		from scipy.stats import ttest_rel
		# This is a test for the null hypothesis that two related or repeated samples have identical average (expected) values.
		Difference_Test = ttest_rel
	if statistic_difference_test == "independent_WilcoxonRankSum":
		# The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn from the same distribution. 
		# The alternative hypothesis is that values in one sample are more likely to be larger than the values in the other sample.
		# This test should be used to compare two samples from continuous distributions. 
		# It does not handle ties between measurements in x and y.
		from scipy.stats import ranksums
		Difference_Test = ranksums
	if statistic_difference_test == "paired_WilcoxonSignedRank":
		# The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. 
		# In particular, it tests whether the distribution of the differences x - y is symmetric about zero. 
		# It is a non-parametric version of the paired T-test.
		from scipy.stats import wilcoxon
		Difference_Test = wilcoxon
	if statistic_difference_test == "MannWhitney_U_Rank":
		from scipy.stats import mannwhitneyu
		# 檢定兩母體分配是某相同
		# The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distribution underlying sample x is the same as the distribution underlying sample y. 
		# It is often used as a test of difference in location between distributions.
		Difference_Test = mannwhitneyu
	if statistic_difference_test == "KolmogorovSmirnov":
		# 檢定兩個經驗分布是否不同或一個經驗分布與另一個理想分布是否不同
		# The one-sample test compares the underlying distribution F(x) of a sample against a given distribution G(x). 
		# The two-sample test compares the underlying distributions of two independent samples. 
		# Both tests are valid only for continuous distributions.
		from scipy.stats import kstest
		Difference_Test = kstest

	from itertools import combinations
	from statsmodels.stats.multitest import multipletests
	pvalue_list = []
	for i in range(n_iter):
		balanced_X_matrix,balanced_y_1Darray = UnderBalancedSampling(unbalanced_X_matrix,unbalanced_y_1Darray,i)
		# find index which corresponded to unique label 
		label_index_list = []
		for k in list(Counter(balanced_y_1Darray).keys()):
			label_index = np.where(balanced_y_1Darray==k)[0]
			label_index_list.append(label_index)
			try:
				if len(label_index_list)==2:
					#print("INNER: ",column_index)
					_,alpha_error= Difference_Test(balanced_X_matrix[label_index_list[0],column_index],balanced_X_matrix[label_index_list[1],column_index])
					#print(alpha_error)
					pvalue_list.append(alpha_error)
				elif len(label_index_list)>2:
					## inner multi testing
					inner_pvalue_list=[]
					for i in combinations(label_index_list,2):
						_,inner_pvalue= Difference_Test(balanced_X_matrix[label_index_list[0],column_index],balanced_X_matrix[label_index_list[1],column_index])
						inner_pvalue_list.append(inner_pvalue)
					## innner multi testing correction
					_,pvalue_correction_list,_,_ = multipletests(inner_pvalue_list,alpha=pvalue,method=pvalue_correction_mehthod)
					general_true_probability = np.prod(1-np.array(pvalue_correction_list))
					alpha_error = 1-general_true_probability
					pvalue_list.append(alpha_error)
				else:
					raise TypeError("Number of Label Type must be equal or greater than 2")
			except:
				#print("i: ",i,"k: ",k,"...ERROR")
				pvalue_list.append(np.nan)
				#col_nan_index_list.append(col_index)
	pvalue_list = [item for item in pvalue_list if str(item) != 'nan']

	#print(pvalue_list)
	if pvalue_list == []:
		return np.nan
	else:
		Significant_list,_,_,_= multipletests(pvalue_list,alpha=pvalue,method=pvalue_correction_mehthod)
		if np.all(Significant_list) ==True:
			return column_index
		else:
			return np.nan


# 2022/5/26 revise(single feauture doing resample,discirminate and which p value correction based on iteration times)
def singleLabelDiscriminationColIndex(unbalanced_X_matrix,unbalanced_y_1Darray,statistic_difference_test,pvalue,pvalue_correction_mehthod,n_iter,n_jobs):
	col_remain_index = Parallel(n_jobs=n_jobs)(delayed(singleColsingleLabelDiscriminationColIndex)(unbalanced_X_matrix=unbalanced_X_matrix,
																							   column_index=col_index,
																							   unbalanced_y_1Darray=unbalanced_y_1Darray,
																							   statistic_difference_test=statistic_difference_test,
																							   pvalue=pvalue,
																							   pvalue_correction_mehthod=pvalue_correction_mehthod,
																							   n_iter=n_iter) for col_index in range(unbalanced_X_matrix.shape[1])
											  )
	col_remain_index = [item for item in col_remain_index if str(item) != 'nan']

	return col_remain_index

@hardware_usage
def multiLabelDiscriminationColIndex(input_X_array,input_y_array,input_y_name,statistic_difference_test,pvalue,pvalue_correction_mehthod,n_iter,n_jobs):
	multi_colidnex_list = Parallel(n_jobs=n_jobs)(delayed(singleLabelDiscriminationColIndex)(unbalanced_X_matrix=input_X_array,
																							 unbalanced_y_1Darray=input_y_array[:,i],
																							 statistic_difference_test=statistic_difference_test,
																							 pvalue=pvalue,
																							 pvalue_correction_mehthod=pvalue_correction_mehthod,
																							 n_iter=n_iter,
																							 n_jobs=n_jobs) for i in range(len(input_y_name)))
	multiLbaelInfoIndex = {}
	for name,index_num in zip(input_y_name,multi_colidnex_list):
		multiLbaelInfoIndex[name] = index_num
	return multiLbaelInfoIndex




# TAKE TWO TAIL
# becuase https://www.twblogs.net/a/5c4b3c32bd9eee6e7d81ed10
# TWO TAIL is asked for 
# difference of ABOSULATE value of statistic and resampling
# one tail (less ot greater) is not useful for our requirement
# we are trying to understand difference only

def singleColsingleLabelPermutationDiscriminationColIndex(input_X_array,column_index,input_y_1Darray,statistic_difference,permutation_type_name,resample_number,alternative_name,pvalue):
	# require scipy==1.8.1
	'''
	this is not usedful for multiclass problem
	'''

	def mean_difference(x, y, axis):
		return np.mean(x, axis=axis) - np.mean(y, axis=axis)
	def median_difference(x, y, axis):
		return np.median(x, axis) - np.median(y, axis)
	def fun(statistic_difference):
		if statistic_difference == "mean":
			return mean_difference
		if statistic_difference == "median":
			return median_difference
	from scipy.stats import permutation_test
	input_X_array = input_X_array[:,column_index]
	'''
	if you require to do by multiclass, you should revise right here
	'''
	non_exist = input_X_array[np.where(input_y_1Darray==0)[0]]
	exist = input_X_array[np.where(input_y_1Darray==1)[0]]
	res = permutation_test((exist,non_exist),statistic=fun(statistic_difference),permutation_type=permutation_type_name,n_resamples=resample_number,alternative=alternative_name, vectorized=True)

	if res.pvalue < pvalue:
		return column_index
	else:
		return np.nan

def singleLabelPermutationDiscriminationColIndex(input_X_array,input_y_1Darray,statistic_difference,permutation_type_name,resample_number,alternative_name,pvalue,n_jobs):
	
	col_remain_index = Parallel(n_jobs=n_jobs)(delayed(singleColsingleLabelPermutationDiscriminationColIndex)(input_X_array=input_X_array,
																											  column_index=col_index,
																											  input_y_1Darray=input_y_1Darray,
																											  statistic_difference=statistic_difference,
																											  permutation_type_name=permutation_type_name,
																											  resample_number=resample_number,
																											  alternative_name=alternative_name,
																											  pvalue=pvalue) for col_index in range(input_X_array.shape[1])
											  )
	col_remain_index = [item for item in col_remain_index if str(item) != 'nan']

	return col_remain_index


@hardware_usage
def multiLabelPermutationDiscriminationColIndex(input_X_array,input_y_array,input_y_name,statistic_difference,permutation_type_name,resample_number,alternative_name,pvalue,n_jobs):
	from joblib import Parallel, delayed
	multi_colidnex_list = Parallel(n_jobs=n_jobs)(delayed(singleLabelPermutationDiscriminationColIndex)(input_X_array=input_X_array,
																							 input_y_1Darray=input_y_array[:,i],
																							 statistic_difference=statistic_difference,
																							 permutation_type_name=permutation_type_name,
																							 resample_number=resample_number,
																							 alternative_name=alternative_name,pvalue=pvalue,n_jobs=n_jobs) for i in range(len(input_y_name)))
	multiLbaelInfoIndex = {}
	for name,index_num in zip(input_y_name,multi_colidnex_list):
		multiLbaelInfoIndex[name] = index_num
	return multiLbaelInfoIndex





