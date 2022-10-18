import numpy as np
import pandas as pd
import joblib
import sys, os
call_module_for_bash = os.path.dirname(str(os.getcwd())) + "/Python"
sys.path.insert(1, call_module_for_bash)
from utility.usage_measurement import hardware_usage
# combination_type: "Type_1" "Type_2" "Type_3"
class MultiLabelLearning:
	def __init__(self,
				 input_feature_data_name,
				 input_basic_model_directory,
				 input_label_name,
				 input_na_index_Info,
				 input_NA_replace_value,
				 input_test_proportion,
				 input_Learning_Type,
				 input_cpu_core,
				 input_inner_cv,
				 input_best_score_name,
				 input_Hyperparameter_Method,
				 input_iterations,
				 tag):
		self.input_feature_data_name = input_feature_data_name
		self.input_label_name = input_label_name #'trophicMode' 'guild'
		self.input_basic_model_directory = input_basic_model_directory # Saving directory
		self.input_na_index_Info = input_na_index_Info[self.input_label_name]
		self.input_NA_replace_value = input_NA_replace_value # NA_replace_value
		self.input_test_proportion = float(input_test_proportion)
		self.input_Learning_Type = input_Learning_Type# "SelfTraining" # SelfTraining ; Supervise
		self.input_cpu_core = int(input_cpu_core) #10
		self.input_inner_cv = int(input_inner_cv) #10
		self.input_best_score_name = input_best_score_name # custom_metrics
		self.input_Hyperparameter_Method = input_Hyperparameter_Method # "RandomizedSearch"
		self.input_iterations = int(input_iterations) # 10
		self.tag = str(tag)
		self.input_SelfTrain_threshold = 0.75
		self.input_SelfTrain_max_iter = 100

		self.basic_name = str(self.input_feature_data_name+"--"+self.input_best_score_name+"--"+self.input_Hyperparameter_Method+"--"+self.input_Learning_Type+"--"+self.input_label_name)

	def fit_model(self,input_X_array,input_X_name,input_y_array,input_feature_selected_name_information,input_Algorithm_Name_list):
		from Feature_Production.Data_Management import train_test_split_at_least_one_label,encode_unknown_label
		from ML_Algorithms.Classification_Algorithms import integratSingleModel
		from sklearn.metrics import make_scorer
		from ML_Algorithms.Classification_Algorithms import hyper_parameters
		if self.input_best_score_name.lower() in ["matthewcorrelationcoefficient","mcc"]:
			from ML_Algorithms.Measurement import MCC
			custom_metrics = make_scorer(MCC,greater_is_better=True)
		elif self.input_best_score_name.lower() in ["hamming_loss","hammingloss"]:
			from ML_Algorithms.Measurement import Hamming_Loss
			custom_metrics = make_scorer(Hamming_Loss,greater_is_better=False)
		elif self.input_best_score_name.lower() in ["f1score","f1_score","fonescore","fmeasurement"]:
			from ML_Algorithms.Measurement import F1Score
			custom_metrics = make_scorer(F1Score,greater_is_better=False)
		else:
			print("Please Select one of following parameter: MCC, F1Score, or Hamming_Loss")
		
		if self.input_Learning_Type != "SelfTraining":
				input_X_array = np.delete(input_X_array, self.input_na_index_Info, axis=0)
				input_y_array = np.delete(input_y_array, self.input_na_index_Info, axis=0)
		feature_selected_index_information = {}
		for _,(y_i,y_i_feature_list) in enumerate(input_feature_selected_name_information.items()):
			# find index location of first occurrence of each value of interest
			# this is quite useful 
			sorter = np.argsort(input_X_name)
			y_i_feature_index = sorter[np.searchsorted(input_X_name, y_i_feature_list, sorter=sorter)]
			#
			feature_selected_index_information[y_i] = y_i_feature_index

		model_info = {}
		for i, (y_i,y_i_feature_index) in enumerate(feature_selected_index_information.items()):
			print(">>>>>>>>>> ",y_i," <<<<<<<<<<"," feature numbers: ",len(y_i_feature_index))
			# train_test_split_at_least_one_label(input_X_array,input_y_array,"MultiLabel",input_test_proportion)
			# or is it better to put right here ???
			alg_info = {}
			for algorithm_name in input_Algorithm_Name_list:
				print(algorithm_name)
				alg_iteration_info ={}
				for iteration in range(self.input_iterations):
					# make sure different parameters
					input_parameters = hyper_parameters(self.input_Learning_Type,n_iter=30,seed_num=int((i+1)*np.log(iteration+1)+iteration+1))
					# train_test_split_at_least_one_label(input_X_array,input_y_array,"MultiLabel",input_test_proportion)
					# Personally Opinion ... insert  at this block is better to sampling different and various instance
					X_train, X_test, y_train, y_test,random_state_num = train_test_split_at_least_one_label(input_X_array,input_y_array,"MultiLabel",self.input_test_proportion)
					X_val_train, X_val_test, y_val_train, y_val_test,random_state_num_val = train_test_split_at_least_one_label(X_train,y_train,"MultiLabel",self.input_test_proportion)
					y_train,_ = encode_unknown_label(y_train,self.input_NA_replace_value)
					y_val_train,_ = encode_unknown_label(y_val_train,self.input_NA_replace_value)
					"""
					Is it nessary to do binay label undersampling here ??
					"""
					print("Iteration ",str(iteration+1),"Label Value: ",list(set(y_val_train[:,i])))
					clf = integratSingleModel(Algorithm_Name=algorithm_name,
											  Learning_Type=self.input_Learning_Type,
											  Hyperparameter_Method=self.input_Hyperparameter_Method,
											  cpu_core=self.input_cpu_core,
											  inner_cv=self.input_inner_cv,
											  best_score=custom_metrics,
											  parameters=input_parameters,
											  SelfTrain_threshold=self.input_SelfTrain_threshold, 
											  SelfTrain_max_iter=self.input_SelfTrain_max_iter
											 )
					model = clf.fit(X_val_train[:,y_i_feature_index],y_val_train[:,i])
					single_model_name = self.basic_name+"--"+str(y_i)+"--"+str(algorithm_name)+"--"+str(iteration+1)+"--"+self.tag+".pkl"
					y_val_pred = model.predict(X_val_test[:,y_i_feature_index])
					y_pred = model.predict(X_test[:,y_i_feature_index])
					MCC_val_test= MCC(y_val_test[:,i],y_val_pred)
					MCC_test = MCC(y_test[:,i],y_pred)
					# add score
					print("MCC_val_test: ",MCC_val_test,"; MCC_test: ",MCC_test)
					alg_iteration_info[str(iteration+1)]=model
					alg_iteration_info[str(iteration+1)+"_MCC_validation"]=MCC_val_test
					alg_iteration_info[str(iteration+1)+"_MCC_test"]=MCC_test
					joblib.dump(model,self.input_basic_model_directory+single_model_name)
					print(single_model_name," has been saved !!")
				alg_info[algorithm_name]=alg_iteration_info
			model_info[y_i] = alg_info
		return self.basic_name, model_info
	
	def get_model(self,input_model_directory):
		from glob import glob
		from ML_Algorithms.Measurement import MCC, Hamming_Loss, F1Score
		global MCC
		global Hamming_Loss
		global F1Score
		main_model_path_list = glob(input_model_directory+"*.pkl")
		# this code stupid but work
		raw_model_info = {}
		for main_model_information in main_model_path_list:
			whole_model_information = main_model_information.replace('\\','/').split("/")[-1].split(".pkl")[0].split("--")
				### all elements in list
			if all(element in whole_model_information for element in [self.input_feature_data_name, self.tag, self.input_label_name, self.input_Learning_Type]):
				### at least one element in list
				# any(element in whole_model_information for element in [self.input_feature_data_name, self.tag, self.input_label_name, self.input_Learning_Type])
				model_basic_name = "--".join(whole_model_information[:5])

				model_label_name = whole_model_information[-4]
				model_algorithm_name = whole_model_information[-3]
				model_iteration = whole_model_information[-2]

				raw_model_info[model_label_name+"--"+model_algorithm_name+"--"+model_iteration] =joblib.load(main_model_information)

		raw_model_info_key_matrix= np.array(list(map(lambda x :x.split("--"),raw_model_info.keys())))
		model_label_name = list(set(raw_model_info_key_matrix[:,0]))
		model_algorithm_name = list(set(raw_model_info_key_matrix[:,1]))
		model_iteration = list(set(raw_model_info_key_matrix[:,2]))
		model_info = {}
		for label_name in model_label_name:
			alg_info = {}
			for algorithm_name in model_algorithm_name:
				alg_iteration_info ={}
				for iteration in model_iteration:
					try:
						alg_iteration_info[iteration]=raw_model_info[label_name+"--"+algorithm_name+"--"+iteration]
					except:
						continue
				alg_info[algorithm_name]=alg_iteration_info
			model_info[label_name] = alg_info
		return model_info





class EnsembleLearning(MultiLabelLearning):
	def __init__(self,input_feature_data_name,input_basic_model_directory,input_label_name,input_na_index_Info,input_NA_replace_value,input_test_proportion,input_Learning_Type,input_cpu_core,input_inner_cv,input_best_score_name,input_Hyperparameter_Method,input_iterations,
				input_feature_selected_name_information,
				input_X_array,
				input_X_name,
				input_y_array,
				input_y_name,
				input_ensemble_model_directory,
				tag):
		super().__init__(input_feature_data_name,input_basic_model_directory,input_label_name,input_na_index_Info,input_NA_replace_value,input_test_proportion,input_Learning_Type,input_cpu_core,input_inner_cv,input_best_score_name,input_Hyperparameter_Method,input_iterations,tag)
		self.input_feature_selected_name_information =input_feature_selected_name_information
		self.input_X_array = input_X_array
		self.input_X_name = input_X_name
		self.input_y_array = input_y_array
		self.input_y_name = input_y_name
		self.input_ensemble_model_directory = input_ensemble_model_directory
	
	@hardware_usage
	def train_model(self,input_Algorithm_Name_list):
		basic_name, model_info = self.fit_model(self.input_X_array,self.input_X_name,self.input_y_array,self.input_feature_selected_name_information,input_Algorithm_Name_list=input_Algorithm_Name_list)
		return basic_name, model_info

	def model_combination(self,combination_type):
		# Type_1 : different algorithms(M) in N round but best to vote; numeber of different types of algorithms; M different results
		# Type_2 : best model in N_th round(maybe XGB in 2nd round is best but Logistic in 3rd is the best) to vote; N different reulst
		# Type_3 : all model participate voting; 1 result ; 2*16*N columns
		feature_selected_index_information = {}
		m = self.input_feature_selected_name_information
		for _,(y_i,y_i_feature_list) in enumerate(m.items()):
			# find index location of first occurrence of each value of interest
			# this is quite useful 
			# because it search original X_name index
			# it will return true feature position
			# unlike previous code
			sorter = np.argsort(self.input_X_name)
			y_i_feature_index = sorter[np.searchsorted(self.input_X_name, y_i_feature_list, sorter=sorter)]
			feature_selected_index_information[y_i] = y_i_feature_index
		from Feature_Production.Data_Management import train_test_split_at_least_one_label,encode_unknown_label
		if self.input_test_proportion == 1.0:
			X_test = self.input_X_array
			y_test = self.input_y_array
		else:
			_, X_test, _, y_test,random_state_num = train_test_split_at_least_one_label(self.input_X_array,self.input_y_array,"MultiLabel",self.input_test_proportion)

		y_test,_ = encode_unknown_label(y_test,self.input_NA_replace_value)
		from ML_Algorithms.Measurement import MCC, Hamming_Loss, F1Score
		# model_inofrmation 這邊 IF ELSE 看要重TRAIN還是抓資料
		model_inofrmation = self.get_model(self.input_basic_model_directory)
		alg_set = set()
		for input_y_name_i in self.input_y_name:
			alg_set.update(model_inofrmation[input_y_name_i].keys())
		alg_iteration_set = set()
		for input_y_name_i in self.input_y_name:
			for alg in alg_set:
				alg_iteration_set.update(model_inofrmation[input_y_name_i][alg].keys())
		df_list = []
		for i,input_y_name_i in enumerate(self.input_y_name):
			for alg in alg_set:
				for alg_iteration in alg_iteration_set:
					#try:
					# no need to revise
					# because it search original 
					y_i_feature_index = feature_selected_index_information[input_y_name_i]
					clf = model_inofrmation[input_y_name_i][alg][alg_iteration]
					y_pred = clf.predict(X_test[:,y_i_feature_index])
					mcc_score = MCC(y_test[:,i],y_pred)
					f_score = F1Score(y_test[:,i],y_pred)
					hamming_loss_score = Hamming_Loss(y_test[:,i],y_pred)
					df_list.append([input_y_name_i,alg,alg_iteration,mcc_score,f_score,hamming_loss_score])
					# except:
					# 	continue
		evaluation_df= pd.DataFrame(df_list,columns=['label', 'alg', 'iter', 'mcc','f1score','hamming_loss'])
		if combination_type == "Type_1":
			##### Type_1 : same algorithms in N round but best to vote; 16 different result 2*16 columns
			# hamming_loss,f1score,mcc
			type_one_list = []
			for input_y_name_i in self.input_y_name:
				sub_evaluation_df= evaluation_df[evaluation_df.label == input_y_name_i]
				for alg_set_i in alg_set:
					sub_evaluation_df_2 = sub_evaluation_df[sub_evaluation_df.alg == alg_set_i].reset_index(drop=True)
					if self.input_best_score_name.lower() in ["hamming_loss","hammingloss"]: 
						type_one_best = sub_evaluation_df_2.sort_values(by=self.input_best_score_name.lower(), ascending=True)[["label","alg","iter"]].agg('--'.join, axis=1).reset_index(drop=True).iloc[0]
					else:
						type_one_best = sub_evaluation_df_2.sort_values(by=self.input_best_score_name.lower(), ascending=False)[["label","alg","iter"]].agg('--'.join, axis=1).reset_index(drop=True).iloc[0]
					type_one_list.append(type_one_best)
			return type_one_list,feature_selected_index_information
		elif combination_type == "Type_2":
			##### Type_2 : best model in N_th round(maybe XGB in 2nd round is best but Logistic in 3rd is the best) to vote; N different reulst; 2*N columns
			type_two_list = []
			for input_y_name_i in self.input_y_name:
				sub_evaluation_df= evaluation_df[evaluation_df.label == input_y_name_i]
				for alg_iteration_set_i in alg_iteration_set:
					sub_evaluation_df_2 = sub_evaluation_df[sub_evaluation_df.iter == alg_iteration_set_i].reset_index(drop=True)
					if self.input_best_score_name.lower() in ["hamming_loss","hammingloss"]: 
						type_two_best = sub_evaluation_df_2.sort_values(by=self.input_best_score_name.lower(), ascending=True)[["label","alg","iter"]].agg('--'.join, axis=1).reset_index(drop=True).iloc[0]
					else:
						type_two_best = sub_evaluation_df_2.sort_values(by=self.input_best_score_name.lower(), ascending=False)[["label","alg","iter"]].agg('--'.join, axis=1).reset_index(drop=True).iloc[0]
					type_two_list.append(type_two_best)
			return type_two_list,feature_selected_index_information
		elif combination_type == "Type_3":
			# Type_3 : all vote
			type_three_list = list(evaluation_df[["label","alg","iter"]].agg('--'.join, axis=1))
			return type_three_list,feature_selected_index_information
		else:
			return None
	def build_prediction(self,combination_type,save):
		model_combanation_list,feature_selected_index_information = self.model_combination(combination_type)

		model_inofrmation = self.get_model(self.input_basic_model_directory)
		info_list = []
		prediction_array = []
		for i,model_name in enumerate(model_combanation_list):
			if i == 0:
				prediction_array.append(self.input_X_array[:,0])
				info_list.append("ID")
			try:
				model_info = model_name.split("--")
				clf = model_inofrmation[model_info[0]][model_info[1]][model_info[2]]
				y_pred = clf.predict(self.input_X_array[:,feature_selected_index_information[model_info[0]]])
				y_predict_proba = clf.predict_proba(self.input_X_array[:,feature_selected_index_information[model_info[0]]])
				y_pred_array = np.column_stack((y_pred,y_predict_proba))
				prediction_array.append(y_pred_array)
				print(model_name," Done !")
				info_list.append(self.input_Learning_Type+"--"+ model_name + "--Hard")
				info_list.append(self.input_Learning_Type+"--"+ model_name + "--Negative")
				info_list.append(self.input_Learning_Type+"--"+ model_name + "--Positive")
			except:
				print("******* ",model_name," no model !!! need to train *******")

		vote_result = pd.DataFrame(np.column_stack(prediction_array),columns=info_list)
		for input_y_name_i in self.input_y_name:
			# debug from "input_y_name_i in x"  to "input_y_name_i in x.split("--")"
			# the string cant definite matching, so require to transform to list
			# and list can find definite matching 
			y_i_hard_col = list(filter(lambda x: input_y_name_i in x.split("--") and "Hard" in x, vote_result.columns))
			y_i_neg_col = list(filter(lambda x: input_y_name_i in x.split("--") and "Negative" in x, vote_result.columns))
			y_i_pos_col = list(filter(lambda x: input_y_name_i in x.split("--") and "Positive" in x, vote_result.columns))
			hard_vote_df = vote_result[y_i_hard_col].apply(lambda x: x.value_counts(), axis=1).fillna(0)
			# make sure two columns; 2022/9/29 debug
			if int(hard_vote_df.shape[1]) == 2:
				vote_result[[input_y_name_i+"--Hard_Zero",input_y_name_i+"--Hard_One"]]=hard_vote_df
			elif int(hard_vote_df.shape[1]) == 1:
				if int(hard_vote_df.columns[0]) == 1:
					hard_vote_df.insert(0,0.0,0)
					vote_result[[input_y_name_i+"--Hard_Zero",input_y_name_i+"--Hard_One"]]=hard_vote_df
				else:
					hard_vote_df.insert(1,1.0,0)
					vote_result[[input_y_name_i+"--Hard_Zero",input_y_name_i+"--Hard_One"]]=hard_vote_df
			else:
				hard_vote_df.insert(0,0.0,0)
				hard_vote_df.insert(1,1.0,0)
				vote_result[[input_y_name_i+"--Hard_Zero",input_y_name_i+"--Hard_One"]]=hard_vote_df
			# check y_i_pos_col and y_i_neg_col is empty list or not
			if not y_i_pos_col:
				vote_result = vote_result.assign(**{input_y_name_i+"--Positive_mean":0,input_y_name_i+"--Positive_median":0})
			else:
				vote_result[input_y_name_i+"--Positive_mean"]= vote_result[y_i_pos_col].mean(axis=1)
				vote_result[input_y_name_i+"--Positive_median"]= vote_result[y_i_pos_col].median(axis=1)
			
			if not y_i_neg_col:
				vote_result = vote_result.assign(**{input_y_name_i+"--Negative_mean":0,input_y_name_i+"--Negative_median":0})
			else:
				vote_result[input_y_name_i+"--Negative_mean"]= vote_result[y_i_neg_col].mean(axis=1)
				vote_result[input_y_name_i+"--Negative_median"]= vote_result[y_i_neg_col].median(axis=1)
		# save ensemble model name
		if save == True:
			basic_name = str(self.input_feature_data_name+"--"+self.input_best_score_name+"--"+self.input_Hyperparameter_Method+"--"+self.input_Learning_Type+"--"+self.input_label_name)
			ensemble_model_name_list= list(map(lambda x: self.input_basic_model_directory+basic_name+"--"+x+"--"+self.tag+".pkl",model_combanation_list))
			with open(self.input_ensemble_model_directory+basic_name+"--"+str(combination_type)+"--model_list"+"--"+self.tag+".txt", '+w') as file:
				for item in ensemble_model_name_list:
					# write each item on a new line
					file.write("%s\n" % item)
			# require to test agian
			# due to permission denial
			vote_result.to_csv(self.input_ensemble_model_directory+basic_name+"--"+str(combination_type)+"--prediction_result"+"--"+self.tag+".txt",sep="\t",header=True,index=None)
			print("saved !!")
		return vote_result

	def predict(self,combination_type,save):
		print("Start to run Ensemble Prediction ...")
		vote_result = self.build_prediction(combination_type,save)
		hard_result = []
		for i,input_y_name_i in enumerate(self.input_y_name):
			#print(vote_result.columns)
			y_i_pred = vote_result.apply(lambda x: 1 if x[input_y_name_i+"--Hard_One"] > x[input_y_name_i+"--Hard_Zero"] else 0,axis=1)
			y_i_pred.name = input_y_name_i
			if i == 0:
				hard_result.append(vote_result["ID"])
			hard_result.append(y_i_pred)
		hard_result = pd.concat(hard_result,axis=1)
		print("hard  DONE")

		soft_mean_result = []
		for i,input_y_name_i in enumerate(self.input_y_name):
			y_i_pred = vote_result.apply(lambda x: 1 if x[input_y_name_i+"--Positive_mean"] > x[input_y_name_i+"--Negative_mean"] else 0,axis=1)
			y_i_pred.name = input_y_name_i
			if i == 0:
				soft_mean_result.append(vote_result["ID"])
			soft_mean_result.append(y_i_pred)
		soft_mean_result = pd.concat(soft_mean_result,axis=1)
		print("mean  DONE")
		soft_median_result = []
		for i,input_y_name_i in enumerate(self.input_y_name):
			y_i_pred = vote_result.apply(lambda x: 1 if x[input_y_name_i+"--Positive_median"] > x[input_y_name_i+"--Negative_median"] else 0,axis=1)
			y_i_pred.name = input_y_name_i
			if i == 0:
				soft_median_result.append(vote_result["ID"])
			soft_median_result.append(y_i_pred)
		soft_median_result = pd.concat(soft_median_result,axis=1)
		print("median  DONE")
		print("Done")
		return hard_result,soft_mean_result,soft_median_result
	
	def get_prediction(self,combination_type):
		basic_name = str(self.input_feature_data_name+"--"+self.input_best_score_name+"--"+self.input_Hyperparameter_Method+"--"+self.input_Learning_Type+"--"+self.input_label_name)
		vote_result_name = self.input_ensemble_model_directory+basic_name+"--"+str(combination_type)+"--prediction_result"+"--"+self.tag+".txt"
		vote_result = pd.read_csv(vote_result_name,sep="\t")
		return vote_result

	def matched_instance(self):
		# looking for partial (union) parts of matched instance name ; minimum instance number
		# looking for complete (intersection) parts of matched instance name ; maximum instance number
		from ML_Algorithms.Measurement import find_TP_ID
		TP_ID = {}
		for TYPE in ["Type_1","Type_2","Type_3"]:
			Type_hard,Type_soft_mean,Type_soft_median = self.predict(TYPE,False)
			MATCH_ID = {}
			for MATCH in ["partial","complete"]:

				Type_hard_ID_name = find_TP_ID(input_y_pred_df=Type_hard,
											   input_y_array=self.input_y_array,
											   input_y_name=self.input_y_name,
											   input_X_array=self.input_X_array,
											   match=MATCH)
				Type_soft_mean_ID_name = find_TP_ID(input_y_pred_df=Type_soft_mean,
													input_y_array=self.input_y_array,
													input_y_name=self.input_y_name,
													input_X_array=self.input_X_array,
													match=MATCH)
				Type_soft_median_ID_name = find_TP_ID(input_y_pred_df=Type_soft_median,
											   input_y_array=self.input_y_array,
											   input_y_name=self.input_y_name,
											   input_X_array=self.input_X_array,
											   match=MATCH)
				MATCH_ID[MATCH]={"hard":Type_hard_ID_name,"soft_mean":Type_soft_mean_ID_name,"soft_median":Type_soft_median_ID_name}
			TP_ID[TYPE]=MATCH_ID

		partial_init = 10**10
		complete_init = 0

		for TYPE in ["Type_1","Type_2","Type_3"]:
			for i in ['hard','soft_mean','soft_median']:
				if len(TP_ID[TYPE]["complete"][i]) > complete_init:
					complete_ID_name = TP_ID[TYPE]["complete"][i]
					complete_init = len(TP_ID[TYPE]["complete"][i])
		for TYPE in ["Type_1","Type_2","Type_3"]:
			for i in ['hard','soft_mean','soft_median']:
				if len(TP_ID[TYPE]["partial"][i]) < partial_init:            
					partial_ID_name = TP_ID[TYPE]["partial"][i]
					partial_init = len(TP_ID[TYPE]["partial"][i])		
		return complete_ID_name, partial_ID_name