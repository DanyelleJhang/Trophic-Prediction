import optuna
import numpy as np
import pandas as pd
import joblib
import sys, os
call_module_for_bash = os.path.dirname(str(os.getcwd())) + "/Python"
sys.path.insert(1, call_module_for_bash)
from utility.usage_measurement import hardware_usage
from ML_Algorithms.Measurement import MCC
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
				 input_balanced_sample,
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
		self.input_balanced_sample = input_balanced_sample
		self.tag = str(tag)
		self.input_SelfTrain_threshold = 0.75
		self.input_SelfTrain_max_iter = 100

		self.basic_name = str(self.input_balanced_sample+"."+self.input_feature_data_name+"--"+self.input_best_score_name+"--"+self.input_Hyperparameter_Method+"--"+self.input_Learning_Type+"--"+self.input_label_name)

	def fit_model(self,input_X_array,input_X_name,input_y_array,input_feature_selected_name_information,input_Algorithm_Name_list):
		from Feature_Production.Data_Management import train_test_split_at_least_one_label,encode_unknown_label
		from ML_Algorithms.Classification_Algorithms import integratSingleModel
		from sklearn.metrics import make_scorer
		from ML_Algorithms.Classification_Algorithms import hyper_parameters
		import warnings
		warnings.filterwarnings("ignore", category=DeprecationWarning)
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
			# because it search original X_name index
			# it will return true feature position
			# unlike previous code
			y_i_feature_index = np.where(np.isin(input_X_name, y_i_feature_list))[0]
			feature_selected_index_information[y_i] = y_i_feature_index


		#print("input_feature_selected_name_information fit_model:",list(map(len,input_feature_selected_name_information.values())))
		#print("feature_selected_index_information fit_model:",list(map(len,feature_selected_index_information.values())))

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
					X_train, X_val, y_train, y_val,random_state_num = train_test_split_at_least_one_label(input_X_array,input_y_array,"MultiLabel",self.input_test_proportion)
					#X_val_train, X_val_test, y_val_train, y_val_test,random_state_num_val = train_test_split_at_least_one_label(X_train,y_train,"MultiLabel",self.input_test_proportion)
					y_train,_ = encode_unknown_label(y_train,self.input_NA_replace_value)
					#y_val_train,_ = encode_unknown_label(y_val_train,self.input_NA_replace_value)
					"""
					Is it nessary to do binay label undersampling here ??
					"""
					y_train_i = y_train[:,i]
					if self.input_balanced_sample == "UnderSampling":
						from imblearn.under_sampling import RandomUnderSampler
						rus = RandomUnderSampler(random_state=42)
						X_train, y_train_i = rus.fit_resample(X_train, y_train_i)
					if self.input_balanced_sample == "OverSampling":
						from imblearn.over_sampling import RandomOverSampler
						ros = RandomOverSampler(random_state=42)
						X_train, y_train_i = ros.fit_resample(X_train, y_train_i)
					print("Iteration ",str(iteration+1),"Label Value: ",list(set(y_train_i)))
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
					model = clf.fit(X_train[:,y_i_feature_index],y_train_i)
					single_model_name = self.basic_name+"--"+str(y_i)+"--"+str(algorithm_name)+"--"+str(iteration+1)+"--"+self.tag+".pkl"
					#y_val_pred = model.predict(X_val_test[:,y_i_feature_index])
					y_val_pred = model.predict(X_val[:,y_i_feature_index])
					#MCC_val_test= MCC(y_val_test[:,i],y_val_pred)
					MCC_val = MCC(y_val[:,i],y_val_pred)
					# add score
					print("MCC_val: ",MCC_val)
					alg_iteration_info[str(iteration+1)]=model
					#alg_iteration_info[str(iteration+1)+"_MCC_validation"]=MCC_val_test
					alg_iteration_info[str(iteration+1)+"_MCC_val"]=MCC_val
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
			#print(main_model_information)
			whole_model_information = main_model_information.replace('\\','/').split("/")[-1].split(".pkl")[0].split("--")
			#print(whole_model_information)
				### all elements in list
			#print([self.input_balanced_sample+"."+self.input_feature_data_name, self.tag, self.input_label_name, self.input_Learning_Type])
			if all(element in whole_model_information for element in [self.input_balanced_sample+"."+self.input_feature_data_name, self.tag, self.input_label_name, self.input_Learning_Type]):
			### at least one element in list
			# any(element in whole_model_information for element in [self.input_feature_data_name, self.tag, self.input_label_name, self.input_Learning_Type])
				model_basic_name = "--".join(whole_model_information[:5])

				model_label_name = whole_model_information[-4]
				model_algorithm_name = whole_model_information[-3]
				model_iteration = whole_model_information[-2]
				#print("model_basic_name: ",model_basic_name)
				#print("model_label_name: ",model_label_name)
				#print("model_algorithm_name: ",model_algorithm_name)
				#print("model_iteration: ", model_iteration)
				#print("main_model_information: ", main_model_information)

				raw_model_info[model_label_name+"--"+model_algorithm_name+"--"+model_iteration] =joblib.load(main_model_information)
		#print(raw_model_info)
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
				input_ensemble_model_data_name,
				input_balanced_sample,
				tag):
		super().__init__(input_feature_data_name,input_basic_model_directory,input_label_name,input_na_index_Info,input_NA_replace_value,input_test_proportion,input_Learning_Type,input_cpu_core,input_inner_cv,input_best_score_name,input_Hyperparameter_Method,input_iterations,input_balanced_sample,tag)
		self.input_feature_selected_name_information = input_feature_selected_name_information
		self.input_X_array = input_X_array
		self.input_X_name = input_X_name
		self.input_y_array = input_y_array
		self.input_y_name = input_y_name
		self.input_ensemble_model_directory = input_ensemble_model_directory
		self.input_ensemble_model_data_name = input_ensemble_model_data_name
		self.basic_name = str(self.input_balanced_sample+"."+self.input_feature_data_name+"--"+self.input_best_score_name+"--"+self.input_Hyperparameter_Method+"--"+self.input_Learning_Type+"--"+self.input_label_name)
	@hardware_usage
	def train_model(self,input_Algorithm_Name_list):
		basic_name, model_info = self.fit_model(self.input_X_array,self.input_X_name,self.input_y_array,self.input_feature_selected_name_information,input_Algorithm_Name_list=input_Algorithm_Name_list)
		return basic_name, model_info
	
	def weighted_prediction(self,input_array,input_weight_array, input_type):
		# hard vote 
		# np.bincount(np.array(pred_test_array)[:10,2].astype(int), weights=[-1,0,1,1,1,1,1,1,1,1])
		# index = [0 , 1]

		# w = [1,1,1,1,1,1,1,1,1,1]
		# count = [7, 3]
		# 1 + 1 + 1 +1 +1 +1 +1

		# w = [-1,1,1,1,1,1,1,1,1,1]
		# count = [5, 3]
		# -1 + 1 +1 +1 +1 +1 +1

		# w = [-1,0,1,1,1,1,1,1,1,1]
		# count = [4, 3]
		# -1 + 0 +1 +1 +1 +1 +1
		if input_type == "hard":
			pred_array_hard = np.apply_along_axis(
				lambda x: np.argmax(np.bincount(x, weights=input_weight_array)),
				axis=0,
				arr=np.array(input_array).astype(int)
			)
			return pred_array_hard
		if input_type in ["soft_mean","soft_median"]:     
			pred_array_soft =np.apply_along_axis(
				lambda x: x*input_weight_array,
				axis=0,
				arr=np.array(input_array)
			)
			pred_array_soft_flip = np.flip(np.array(input_array),axis=2)
			flip_index = np.where(input_weight_array==-1)[0]
			pred_array_soft[flip_index,:,:] = pred_array_soft_flip[flip_index,:,:]
			if input_type == "soft_mean":
				pred_array_soft_mean = np.argmax(np.mean(pred_array_soft,axis=0),axis=1)
				return pred_array_soft_mean
			if input_type == "soft_median":
				pred_array_soft_median = np.argmax(np.median(pred_array_soft,axis=0),axis=1)
				return pred_array_soft_median
			
	def predict(self,combination_type,input_weighted_info=None):

		evaluation = MCC
		X_name_array = np.array(self.input_X_name)
		feature_selected_index_information = {}
		feature_selected_name_information = self.input_feature_selected_name_information
		for _,(y_i,y_i_feature_list) in enumerate(feature_selected_name_information.items()):
			# find index location of first occurrence of each value of interest
			# this is quite useful 
			# because it search original X_name index
			# it will return true feature position
			# unlike previous code
			y_i_feature_index = np.where(np.isin(X_name_array, y_i_feature_list))[0]
			feature_selected_index_information[y_i] = y_i_feature_index
		
		model_inofrmation = self.get_model(self.input_basic_model_directory)
		best_params = {}
		best_score = {}
		prediction_info = {} 
		if combination_type == "Type_1":
			w_i = 1
			weight_range = [1]
		if combination_type == "Type_2":
			w_i = 2
			weight_range = [0,1]
		if combination_type == "Type_3":
			w_i = 3
			weight_range = [-1,0,1]
		
		if input_weighted_info == None:
			y_array = np.delete(self.input_y_array,self.input_na_index_Info,axis=0)
			X_array = np.delete(self.input_X_array,self.input_na_index_Info,axis=0)
		
		if input_weighted_info == None:
			w_list = []
			for l,y_i in enumerate(self.input_y_name):
				## 最後驗證用的
				#y_i_origin_array = self.input_y_array[:,l]
				y_i_origin_X_array = self.input_X_array[:,feature_selected_index_information[y_i]]

				y_i_array = y_array[:,l]
				y_i_X_array = X_array[:,feature_selected_index_information[y_i]]
				y_i_all_model = model_inofrmation[y_i]

				iter_list = []
				for v_i in y_i_all_model.values():
					for v_ii in v_i.keys():
						iter_list.append(v_ii)
				iter_list = list(set(iter_list)
								)				
				alg_name_list = list(y_i_all_model.keys())
				pred_proba_origin_array = []
				pred_origin_array = []				
				pred_proba_array = []
				pred_array = []
				model_name_list = []
				for alg_name in alg_name_list:
					for iter_i in iter_list:

						y_i_single_model = y_i_all_model[alg_name][iter_i]
						
						y_i_origin_pred_proba = y_i_single_model.predict_proba(y_i_origin_X_array)
						y_i_origin_pred = y_i_single_model.predict(y_i_origin_X_array)

						y_i_pred_proba = y_i_single_model.predict_proba(y_i_X_array)
						y_i_pred = y_i_single_model.predict(y_i_X_array)
						single_model_name = self.basic_name+"--"+str(y_i)+"--"+str(alg_name)+"--"+str(iter_i)+"--"+self.tag+".pkl"
						#y_i_train_pred_proba = y_i_single_model.predict_proba(y_i_X_train_array)
						#y_i_train_pred = y_i_single_model.predict(y_i_X_train_array)    

						#pred_proba_train_array.append(y_i_train_pred_proba)
						#pred_train_array.append(y_i_train_pred)
						# print(single_model_name)
						pred_proba_origin_array.append(y_i_origin_pred_proba)
						pred_origin_array.append(y_i_origin_pred)
						pred_proba_array.append(y_i_pred_proba)
						pred_array.append(y_i_pred)
						model_name_list.append(single_model_name)
				#print(model_name_list)
# single_model_name  model_name_list 可能這個有錯
				optuna.logging.set_verbosity(optuna.logging.WARNING)
				sampler_method = optuna.samplers.TPESampler() # optuna.samplers.RandomSampler()#     

				def objective_hard(trial):
					# y_i_all_model
					weight_array = []
					for i in model_name_list:
						# y_i_all_model_weights[i]=np.random.choice([-1,0,1]) # np.random.uniform(low=-1.0, high=1.0) # np.random.choice([-1,1])
						w = trial.suggest_categorical(f"{i}",weight_range) # 
						#print(w)
						weight_array.append(w) # 0
					weight_array = np.array(weight_array)
					pred_array_hard = self.weighted_prediction(input_array=pred_array,
																input_weight_array=weight_array,
																input_type = "hard")

					evaluation_hard = evaluation(pred_array_hard,y_i_array)
					return evaluation_hard

				def objective_soft_mean(trial):
					# y_i_all_model
					weight_array = []

					for i in model_name_list:
						# y_i_all_model_weights[i]=np.random.choice([-1,0,1]) # np.random.uniform(low=-1.0, high=1.0) # np.random.choice([-1,1])
						w = trial.suggest_categorical(f"{i}",weight_range) # 
						weight_array.append(w) # 0
					weight_array = np.array(weight_array)

					pred_array_soft_median = self.weighted_prediction(input_array=pred_proba_array,
														   input_weight_array=weight_array,
														   input_type = "soft_median"
														  )
					evaluation_soft_median = evaluation(pred_array_soft_median,y_i_array)
					return evaluation_soft_median

				def objective_soft_median(trial):
					# y_i_all_model
					weight_array = []
					for i in model_name_list:
						# y_i_all_model_weights[i]=np.random.choice([-1,0,1]) # np.random.uniform(low=-1.0, high=1.0) # np.random.choice([-1,1])
						w = trial.suggest_categorical(f"{i}",weight_range) # 
						weight_array.append(w) # 0
					weight_array = np.array(weight_array)

					pred_array_soft_mean = self.weighted_prediction(input_array=pred_proba_array,
														   input_weight_array=weight_array,
														   input_type = "soft_mean"
														  )

					evaluation_soft_mean = evaluation(pred_array_soft_mean,y_i_array)
					return evaluation_soft_mean         

				study_hard = optuna.create_study(direction="maximize",pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=2)) # , pruner= optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1)
				study_hard.optimize(objective_hard, n_trials=3000, timeout=30*60) # n_trials=3000, timeout=1*24*60*60,  sampler=sampler_method

				study_soft_mean = optuna.create_study(direction="maximize",pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=2)) # , pruner= optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1)
				study_soft_mean.optimize(objective_soft_mean, n_trials=3000, timeout=30*60) # n_trials=3000, timeout=1*24*60*60

				study_soft_median = optuna.create_study(direction="maximize",pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=2)) # , pruner= optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1)
				study_soft_median.optimize(objective_soft_median, n_trials=3000, timeout=30*60) # n_trials=3000, timeout=1*24*60*60


				best_score[y_i]={"hard":study_hard.best_value,"mean":study_soft_mean.best_value,"median":study_soft_median.best_value}
				best_params[y_i]={"hard":study_hard.best_trial.params,"mean":study_soft_mean.best_trial.params,"median":study_soft_median.best_trial.params}
				#best_params[y_i] = study.best_trial.params

				#print(best_params)
				prediction_hard = self.weighted_prediction(
													  input_array=pred_origin_array,
													  input_weight_array=list(best_params[y_i]['hard'].values()),
													  input_type = "hard"
													 )

				prediction_soft_mean= self.weighted_prediction(input_array=pred_proba_origin_array,
														  input_weight_array=list(best_params[y_i]['mean'].values()),
														  input_type = "soft_mean"
														 )

				prediction_soft_median = self.weighted_prediction(input_array=pred_proba_origin_array,
															 input_weight_array=list(best_params[y_i]['median'].values()),
															 input_type = "soft_median"
															)

				prediction_info[y_i]={"hard":prediction_hard,"mean":prediction_soft_mean,"median":prediction_soft_median}
				w = pd.DataFrame.from_dict(best_params[y_i]).reset_index()
				w_list.append(w)
			
			weight_model_df = pd.concat(w_list,axis=0).rename(columns={"index":"model"})
			weight_model_df.to_csv(self.input_ensemble_model_directory+self.basic_name+".Type_"+str(w_i)+"--"+str("weighted")+"--model_list"+"--"+self.tag+".txt",sep="\t",index=False)		
		else: 
			ensemble_info = pd.read_csv(self.input_ensemble_model_directory+self.input_ensemble_model_data_name,sep="\t")
			ensemble_info["label"]= ensemble_info["model"].str.split("--",expand=True).iloc[:,5]#.str.split()
			ensemble_info["model"] = ensemble_info["model"].apply(lambda x: joblib.load((self.input_basic_model_directory+x)))
			prediction_info = {}
			for l,y_i in enumerate(self.input_y_name):
				y_i_X_array = self.input_X_array[:,feature_selected_index_information[y_i]] # self.input_X_array
				y_i_ensemble_info = ensemble_info[ensemble_info.label == y_i]
				pred_array = np.stack(y_i_ensemble_info["model"].apply(lambda x: x.predict(y_i_X_array)).to_numpy())
				pred_proba_array = np.stack(y_i_ensemble_info["model"].apply(lambda x: x.predict_proba(y_i_X_array)).to_numpy())
				prediction_hard = self.weighted_prediction(input_array=pred_array,
													  input_weight_array=list(y_i_ensemble_info["hard"]),
													  input_type = "hard")
				prediction_soft_mean= self.weighted_prediction(input_array=pred_proba_array,
														  input_weight_array=list(y_i_ensemble_info["mean"]),
														  input_type = "soft_mean")
				prediction_soft_median = self.weighted_prediction(input_array=pred_proba_array,
															 input_weight_array=list(y_i_ensemble_info["median"]),
															 input_type = "soft_median")
				prediction_info[y_i]={"hard":prediction_hard,"mean":prediction_soft_mean,"median":prediction_soft_median}
		return prediction_info
	
	def matched_instance(self):
		# looking for partial (union) parts of matched instance name ; minimum instance number
		# looking for complete (intersection) parts of matched instance name ; maximum instance number
		from ML_Algorithms.Measurement import find_TP_ID
		TP_ID = {}
		for TYPE in ["Type_1","Type_2","Type_3"]:
			#weight_model_df = pd.read_csv(self.input_ensemble_model_directory+self.basic_name+".Type_"+str(w_i)+"--"+str("weighted")+"--model_list"+"--"+tag+".txt",sep="\t")
			#weight_model_df = pd.concat([weight_model_df["model"].str.split("--", expand=True)[5],weight_model_df],axis=1) # 5: LABEL NAME; IT MAY CHANGE
			#weight_info = {}
			#for y_i in set(weight_model_df.loc[:,5]):
			#	weight_info[y_i]={"hard":weight_model_df[weight_model_df.loc[:,5] == y_i]["hard"].to_numpy(),"mean":weight_model_df[weight_model_df.loc[:,5] == y_i]["mean"].to_numpy(),"median":weight_model_df[weight_model_df.loc[:,5] == y_i]["median"].to_numpy()}
			prediction_info = self.predict(TYPE)
			y_array_hard = []
			y_array_soft_mean = []
			y_array_soft_median = []
			for y_i in self.input_y_name:
				y_array_hard.append(prediction_info[y_i]["hard"])
				y_array_soft_mean.append(prediction_info[y_i]["mean"])
				y_array_soft_median.append(prediction_info[y_i]["median"])
			
			ID_array = self.input_X_array[:,0] # self.input_X_array[:,0]
			y_array_hard = pd.DataFrame(np.c_[ID_array,np.array(y_array_hard).T],columns=["ID"]+self.input_y_name)
			y_array_soft_mean = pd.DataFrame(np.c_[ID_array,np.array(y_array_soft_mean).T],columns=["ID"]+self.input_y_name)
			y_array_soft_median = pd.DataFrame(np.c_[ID_array,np.array(y_array_soft_median).T],columns=["ID"]+self.input_y_name)
			MATCH_ID = {}
			for MATCH in ["partial","complete"]:

				Type_hard_ID_name = find_TP_ID(input_y_pred_df=y_array_hard,
											   input_y_array=self.input_y_array,
											   input_y_name=self.input_y_name,
											   input_X_array=self.input_X_array,
											   match=MATCH)
				Type_soft_mean_ID_name = find_TP_ID(input_y_pred_df=y_array_soft_mean,
													input_y_array=self.input_y_array,
													input_y_name=self.input_y_name,
													input_X_array=self.input_X_array,
													match=MATCH)
				Type_soft_median_ID_name = find_TP_ID(input_y_pred_df=y_array_soft_median,
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