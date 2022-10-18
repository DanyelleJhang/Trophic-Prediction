#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
# Note: **********
# Evaluation.py in the future
# Multi label matrix and various report
def MCC(y_1Darray,y_pred_1Darray):
	from sklearn.metrics import matthews_corrcoef
	if eval("-1") in set(y_1Darray):
		unkown_label_index = np.where(y_1Darray!=eval("-1"))[0]
		y_1Darray = y_1Darray[unkown_label_index] # resotre unknown label index
		y_pred_1Darray = y_pred_1Darray[unkown_label_index]
	evaluation = matthews_corrcoef(y_1Darray,y_pred_1Darray)
	return evaluation

def Hamming_Loss(y_1Darray,y_pred_1Darray):
	from sklearn.metrics import hamming_loss
	if eval("-1") in set(y_1Darray):
		unkown_label_index = np.where(y_1Darray!=eval("-1"))[0]
		y_1Darray = y_1Darray[unkown_label_index] # resotre unknown label index
		y_pred_1Darray = y_pred_1Darray[unkown_label_index]
	evaluation = hamming_loss(y_1Darray,y_pred_1Darray)
	return evaluation

def F1Score(y_1Darray,y_pred_1Darray):
	from sklearn.metrics import f1_score
	if eval("-1") in set(y_1Darray):
		unkown_label_index = np.where(y_1Darray!=eval("-1"))[0]
		y_1Darray = y_1Darray[unkown_label_index] # resotre unknown label index
		y_pred_1Darray = y_pred_1Darray[unkown_label_index]
	evaluation = f1_score(y_1Darray,y_pred_1Darray)
	return evaluation

def find_TP_ID(input_y_pred_df,input_y_array,input_y_name,input_X_array,match):
	y_pred_df = input_y_pred_df.set_index("ID")
	y_pred_name = list(y_pred_df.columns)
	y_true_name = list(map(lambda x: x + "_true",input_y_name))
	y_true_df = pd.DataFrame(np.column_stack((input_X_array[:,0],input_y_array)),columns=["ID"]+y_true_name).set_index("ID")
	y_true_df = y_true_df[y_true_df.sum(axis=1) != 0] # remove unknown
	del input_y_pred_df
	del input_X_array
	y_true_pred_df = pd.concat([y_true_df,y_pred_df],axis=1).dropna(axis=0) # remove unknown
	del y_true_df
	del y_pred_df
	match_index_list = []
	for input_y_name_i in input_y_name:
		y_i_pair = y_true_pred_df[[input_y_name_i,input_y_name_i+"_true"]]
		match_index = np.where(y_i_pair.iloc[:,0]==y_i_pair.iloc[:,1])[0]
		match_index_list.append(match_index)
	if match == "complete":
		ID_num = list(set.intersection(*map(set,match_index_list)))
	elif match == "partial":
		ID_num = list(set.union(*map(set,match_index_list)))
	else:
		print("please choose complete or partial (intersection or union)")
	return np.array(y_true_pred_df.index[ID_num])


def filter_unknown_label(input_origin_array,input_predicted_array,input_unknown_index_array):
	origin_filter_array = np.delete(input_origin_array, input_unknown_index_array, axis=0)
	predicted_filter_array = np.delete(input_predicted_array, input_unknown_index_array, axis=0)
	return origin_filter_array, predicted_filter_array

#### Mulit Class Random Expectation 
def random_expectation(input_total_lable_list):
	from collections import Counter
	from pandas import DataFrame
	labe_number = Counter(input_total_lable_list)
	labe_probability_dict = {}
	for k in list(labe_number.keys()):
		labe_probability_dict[k] = labe_number[k]/sum(labe_number.values())
	# Multi class random expectation 
	lable_type= list(labe_probability_dict.keys())
	# = np.zeros((len(lable_type),len(lable_type)), dtype=float)
	#np.add.at(r, (2,3), 5)
	confusion_matrix = np.zeros((len(lable_type),len(lable_type)), dtype=float)
	for p_i,p_k in enumerate(lable_type):
		for t_i,t_k in enumerate(lable_type):
			probability = labe_probability_dict[p_k]*labe_probability_dict[t_k]
			np.add.at(confusion_matrix, (int(p_i),int(t_i)), probability)
	# (Predict, True) => 
	#          Predict
	#        TN    FP      
	# True   FN    TP
	TP_list = list(np.diagonal(confusion_matrix))
	np.fill_diagonal(confusion_matrix,0)
	# => False Negative (class i) [i,:]
	# => False Positve (class i) [:,i]
	# Precision (class=i) => TP (class=i) / TP (class=i) + FP (class=i)
	# Recall (class=i) => TP(class=i) / TP(class=i) + FN(class=i)
	precision_list = []
	recall_list = []
	for i,v in enumerate(TP_list):
		p = v/(v+sum(confusion_matrix[i,:]))
		r = v/(v+sum(confusion_matrix[:,i]))
		precision_list.append(p)
		recall_list.append(r)
	recall_array = np.array(recall_list)
	precision_array = np.array(precision_list)
	F1_array = 2*(recall_array*precision_array)/(recall_array+precision_array)
	support_array = np.array(list(labe_probability_dict.values()))
	performance_matrix = np.stack((precision_array,recall_array,F1_array,support_array),axis=-1)
	maro_avg_array = performance_matrix.sum(axis=0)/len(performance_matrix)
	np.add.at(maro_avg_array,-1,np.nan)
	weight_avg_array= np.array([sum(precision_array*support_array),sum(recall_array*support_array),sum(F1_array*support_array),sum(support_array)])
	np.add.at(weight_avg_array,-1,np.nan)
	#Total_accuracy = sum(TP_list)/(sum(TP_list)+sum(sum(confusion_matrix)))
	#total_accuracy_array = np.array([np.nan,np.nan,np.nan,Total_accuracy])
	performance_matrix_2 = np.vstack((performance_matrix,maro_avg_array,weight_avg_array))
	performance_dataframe = DataFrame(performance_matrix_2,columns=["random_precision","random_recall","random_f1-score","support_proportion"],index=lable_type+["macro avg","weighted avg"])
	return performance_dataframe

def Random_Expectation(input_y_array,input_y_name,input_lable_type):
	from pandas import DataFrame
	if input_lable_type == "MultiLabel":
		multilable_freq = dict(DataFrame(input_y_array,columns=input_y_name).sum())
		R = random_expectation(multilable_freq)
	if input_lable_type == "MultiClass":
		y_name_dict = {}
		for i,k in enumerate(input_y_name):
			y_name_dict[i]=k
		multiclass_freq = np.array([y_name_dict[i] for i in input_y_array])
		R = random_expectation(multiclass_freq)
	return R
# Multi_Class_Random_Expectation({"A":72,"B":30,"C":3000,"D":100,"E":97,"T":89,"G":20,"R":9})

def Result_Report(input_y_val_test,input_y_val_pre,input_y_test,input_y_pre,input_y_val_test_NA_index,input_y_test_NA_index,input_y_name,input_ALG_Lable_Type):
	from sklearn.metrics import classification_report
	from pandas import DataFrame, merge
	input_y_val_test,input_y_val_pre = filter_unknown_label(input_y_val_test,input_y_val_pre,input_y_val_test_NA_index)
	input_y_test,input_y_pre = filter_unknown_label(input_y_test,input_y_pre,input_y_test_NA_index)
	# Random_Expectation(input_y_array,input_y_name,input_lable_type)
	Val_Report = DataFrame(classification_report(input_y_val_test,input_y_val_pre,target_names=input_y_name, output_dict=True)).T
	Val_Random_Expectation = Random_Expectation(input_y_val_test,input_y_name,input_ALG_Lable_Type)
	Val_Result = merge(Val_Report, Val_Random_Expectation, left_index=True, right_index=True)
	Val_Result.columns = "validation_"+Val_Result.columns
	Test_Report = DataFrame(classification_report(input_y_test,input_y_pre,target_names=input_y_name, output_dict=True)).T
	Test_Random_Expectation = Random_Expectation(input_y_test,input_y_name,input_ALG_Lable_Type)
	Test_Result = merge(Test_Report, Test_Random_Expectation, left_index=True, right_index=True)
	Test_Result.columns = "test_"+Test_Result.columns
	All_Result = merge(Val_Result, Test_Result, left_index=True, right_index=True)
	All_Result_1 = All_Result.drop(index=["macro avg","weighted avg"]).fillna(0)
	macro_avg_list = []
	weighted_avg_list = []
	for i in list(All_Result_1.columns):
		if "validation_" in i:
			if i == "validation_support":
				total_number = All_Result_1[i].sum(axis=0)
				macro_avg_list.append(total_number)
				weighted_avg_list.append(total_number)
			else:
				macro = All_Result_1[i].sum(axis=0)/len(All_Result_1[i])
				weighted = (All_Result_1[i]*All_Result_1["validation_support"]).sum(axis=0)/All_Result_1["validation_support"].sum(axis=0)
				macro_avg_list.append(macro)
				weighted_avg_list.append(weighted)
		elif "test_" in i:
			if i == "test_support_proportion":
				total_number = All_Result_1[i].sum(axis=0)
				macro_avg_list.append(total_number)
				weighted_avg_list.append(total_number)
			else:
				macro = All_Result_1[i].sum(axis=0)/len(All_Result_1[i])
				weighted = (All_Result_1[i]*All_Result_1["test_support_proportion"]).sum(axis=0)/All_Result_1["test_support_proportion"].sum(axis=0)
				macro_avg_list.append(macro)
				weighted_avg_list.append(weighted)
		else:
			print("ERROR..........")
	All_Result.loc["Revised_Macro_Avg"] = macro_avg_list
	All_Result.loc["Revised_Weighted_Avg"] = weighted_avg_list
	print("============ All Report Done ==============================================================================")
	return All_Result