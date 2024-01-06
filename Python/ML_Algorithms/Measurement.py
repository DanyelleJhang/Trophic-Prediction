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

def get_prediction_result(input_origin_result,input_prediction_result,input_label_information,input_na_index,input_transform):
	merge_result = pd.merge(input_origin_result, input_prediction_result, on="ID")
	# note: should not chang posistion
	# [0, 1]
	hard_name_list = ["--Hard_Zero","--Hard_One"]
	mean_name_list = ["--Negative_mean","--Positive_mean"]
	median_name_list = ["--Negative_median","--Positive_median"] # note: should not chang posistion

	hard_label_dict = {}
	for i in input_label_information:
		hard_label_dict[i]= [i + s for s in hard_name_list]
	mean_label_dict = {}
	for i in input_label_information:
		mean_label_dict[i]= [i + s for s in mean_name_list]
	median_label_dict = {}
	for i in input_label_information:
		median_label_dict[i]= [i + s for s in median_name_list]

	# label_Info[label_name]
	hard_pred_list = []
	for i in input_label_information:
		#if input_transform[i]["Hard"] == True:
		#pred= np.argmin(np.array(merge_result[hard_label_dict[i]]), axis=1)
		#else:
		pred= np.argmax(np.array(merge_result[hard_label_dict[i]]), axis=1)
		hard_pred_list.append(pred)
	hard_pred = pd.DataFrame(np.column_stack(hard_pred_list),columns=input_label_information)

	mean_pred_list = []
	for i in input_label_information:
		if input_transform[i]["Mean"] == True:
			pred= np.argmin(np.array(merge_result[mean_label_dict[i]]), axis=1)
		else:
			pred= np.argmax(np.array(merge_result[mean_label_dict[i]]), axis=1)
		mean_pred_list.append(pred)
	mean_pred = pd.DataFrame(np.column_stack(mean_pred_list),columns=input_label_information)

	median_pred_list = []
	for i in input_label_information:
		if input_transform[i]["Median"] == True:
			pred= np.argmin(np.array(merge_result[median_label_dict[i]]), axis=1)
		else:
			pred= np.argmax(np.array(merge_result[median_label_dict[i]]), axis=1)
		median_pred_list.append(pred)
	median_pred = pd.DataFrame(np.column_stack(median_pred_list),columns=input_label_information)

	true_label = merge_result[input_label_information]

	true_label_dropna, hard_pred_dropna = filter_unknown_label(np.array(true_label),np.array(hard_pred),input_na_index)
	true_label_dropna, mean_pred_dropna = filter_unknown_label(np.array(true_label),np.array(mean_pred),input_na_index)
	true_label_dropna, median_pred_dropna = filter_unknown_label(np.array(true_label),np.array(median_pred),input_na_index)
	return true_label_dropna, hard_pred_dropna, mean_pred_dropna, median_pred_dropna

def measurement(input_confusion_matrix):
	TN = input_confusion_matrix[0,0]
	FP = input_confusion_matrix[0,1]
	FN = input_confusion_matrix[1,0]
	TP = input_confusion_matrix[1,1]

	accuracy = (TP+TN)/(TN+FP+FN+TP)
	false_positive_rate = FP/(FP+TN)
	false_negative_rate = FN/(FN+TP) # miss rate
	zero_rate = (TN+FN)/(TN+FP+FN+TP) # zero_rate
	specificity = TN/(TN+FP) # specificity
	precision = TP/(TP+FP)
	recall = TP/(TP+FN)
	f1 = (2*precision*recall )/(precision + recall)
	mcc = (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
	normalize_mcc = (mcc+1)/2
	return accuracy, false_positive_rate, false_negative_rate,zero_rate,specificity, precision,recall,f1,mcc,normalize_mcc

def multi_lable_measurement(input_confusion_matrix):
	if len(input_confusion_matrix.shape) ==3:
		result_list = []
		for i in input_confusion_matrix:
			r = measurement(i)
			result_list.append(list(r))
		result_array = np.nansum(np.array(result_list),axis=0)/len(result_list)
		# accuracy, false_positive_rate, false_negative_rate,zero_rate,specificity, precision, recall, f1, mcc, normalize_mcc
		return result_array
	elif len(input_confusion_matrix.shape) ==2:
		r = measurement(input_confusion_matrix)
		result_list= list(r)
		result_array = np.array(result_list)
		# accuracy, false_positive_rate, false_negative_rate,zero_rate,specificity,precision, recall, f1, mcc, normalize_mcc
		return result_array
	else:
		raise Exception("shap of confusion matrix should be 2 or 3")
def multi_lable_evaluation(input_y_true_array, input_y_pred_array):
	from sklearn.metrics import multilabel_confusion_matrix
	mll_macro_confusion_matrix = multilabel_confusion_matrix(input_y_true_array, input_y_pred_array)
	mll_instance_confusion_matrix = multilabel_confusion_matrix(input_y_true_array, input_y_pred_array,samplewise=True)
	mll_micro_confusion_matrix = sum(mll_macro_confusion_matrix)
	#              predicted zero     predicted one
	# true zero       TN                  FP                 
	# true one        FN                  TP
	macro_result = multi_lable_measurement(mll_macro_confusion_matrix)
	instance_result = multi_lable_measurement(mll_instance_confusion_matrix)
	micro_result = multi_lable_measurement(mll_micro_confusion_matrix)
	evaluation = np.row_stack((macro_result,instance_result,micro_result))
	evaluation_result = pd.DataFrame(evaluation,index=["macro avg","samples avg","micro avg"],columns = ["Accuracy", "False_Positive_Rate", "False_Negative_Rate", "Zero_Rate", "Specificity", "Precision","Recall","F1","MCC","nMCC"])
	return evaluation_result
def random_expectation(input_y_array,input_y_name):
	multilable_freq = dict(pd.DataFrame(input_y_array,columns=input_y_name).sum())
	from collections import Counter
	labe_number = Counter(multilable_freq)
	labe_probability_dict = {}
	for k in list(labe_number.keys()):
		labe_probability_dict[k] = labe_number[k]/sum(labe_number.values())

	######################### label based ############################
	#              predicted zero     predicted one
	# true zero       TN                  FP                 
	# true one        FN                  TP
	lable_type= list(labe_probability_dict.keys())
	confusion_matrix = np.zeros((len(lable_type),len(lable_type)), dtype=float)
	for p_i,p_k in enumerate(lable_type):
		for t_i,t_k in enumerate(lable_type):
			probability = labe_probability_dict[p_k]*labe_probability_dict[t_k]
			np.add.at(confusion_matrix, (int(p_i),int(t_i)), probability)
	TP_list = list(np.diagonal(confusion_matrix))
	np.fill_diagonal(confusion_matrix,0)
	# => False Positve (class i) [:,i]
	# => False Negative (class i) [i,:]
	# Precision (class=i) => TP (class=i) / TP (class=i) + FP (class=i)
	# Recall (class=i) => TP(class=i) / TP(class=i) + FN(class=i)
	macro_list = []
	micro_list = []
	for i,v in enumerate(TP_list):
		TP = v
		TN = sum(np.delete(np.array(TP_list), i))
		FP = sum(confusion_matrix[:,i])
		FN = sum(confusion_matrix[i,:])
		macro_cm = np.array([[TN,FP],
							 [FN,TP]])
		macro_reults = measurement(macro_cm)
		micro_list.append(macro_cm)
		macro_list.append(list(macro_reults))

	macro_random_expectaion = np.average(np.array(macro_list),axis=0)
	micro_random_expectaion= np.array(measurement(np.average(np.array(micro_list),axis=0)))
	########################## instance based ##########################
	#              predicted zero     predicted one
	# true zero       TN                  FP                 
	# true one        FN                  TP
	labe_probability = np.array(list(labe_probability_dict.values()))
	instance_list = []
	for i,v in enumerate(labe_probability):
		TP = v
		TN = TN = sum(np.delete(np.array(labe_probability), i))

		from itertools import permutations
		init_array = np.zeros(len(labe_probability), dtype=float)
		init_i = len(init_array) # change from len(init_array) -1
		i = 0 
		FN_value_list = []
		while i < init_i:
			init_array[i] = 1
			for ii in set(permutations(init_array)):
				FN_probability= np.nanprod(np.abs(np.array(ii)-labe_probability))
				FN_value_list.append(FN_probability)
			i += 1
		FN = np.sum(FN_value_list)

		init_array = np.zeros(len(1.0-labe_probability), dtype=float)
		init_i = len(init_array) # change from len(init_array) -1
		i = 0 
		FP_value_list = []
		while i < init_i:
			init_array[i] = 1
			for ii in set(permutations(init_array)):
				FP_probability= np.nanprod(np.abs(np.array(ii)-(1.0-labe_probability)))
				FP_value_list.append(FP_probability)
			i += 1
		FP = np.sum(FP_value_list)
		instance_results = np.array(measurement(np.array([[TN,FP],[FN,TP]])))
		instance_list.append(instance_results)
	instance_random_expectaion = np.average(instance_list,axis=0)
	random_expectaion= np.row_stack((macro_random_expectaion,instance_random_expectaion,micro_random_expectaion))
	random_expectaion_result = pd.DataFrame(random_expectaion,index=["macro avg","samples avg","micro avg"],columns = ["Precision","Recall","F1","MCC","nMCC"])
	return random_expectaion_result




# from itertools import product
# bool_list = list(product([True,False],repeat=2))
# bool_comba_list = list(product(*[bool_list,bool_list,bool_list]))

# for i in bool_comba_list:
#     transform_dict = {'Pathotroph':{'Mean':i[0][0],'Median':i[0][1]},'Saprotroph':{'Mean':i[1][0],'Median':i[1][1]},'Symbiotroph':{'Mean':i[2][0],'Median':i[2][1]}}
#     True_Label_Dropna, Hard_Pred_Dropna, Mean_Pred_Dropna, Median_Pred_Dropna = get_prediction_result(input_origin_result = origin_result,
#                                                                                                   input_prediction_result = prediction_result,
#                                                                                                   input_label_information = label_Info['trophicMode'],
#                                                                                                   input_na_index = na_index_test_Info['trophicMode'],
#                                                                                                   input_transform= transform_dict)
#     print("-----------------------------------------------------")
#     print(transform_dict)
#     print("-----------------------------------------------------")
#     for name,method_array in zip(["hard","mean","median"],[Hard_Pred_Dropna, Mean_Pred_Dropna, Median_Pred_Dropna]): 
#         print(">>>>>>> ",name)
#         #y_true, y_pred = filter_unknown_label(trophicMode_test_array,np.array(method_df.iloc[:,1:]),na_index_test_Info['trophicMode'])
#         print(classification_report(True_Label_Dropna.astype(int).tolist(), method_array.tolist(), target_names=label_Info['trophicMode']))