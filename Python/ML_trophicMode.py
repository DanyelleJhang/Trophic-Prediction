import pandas as pd
import numpy as np
import joblib
import os,argparse,sys
from time import time, strftime, gmtime
# if not sys.warnoptions:
# 	import warnings
# 	warnings.simplefilter("ignore")
from Feature_Production.Data_Management import information_arangement
from sklearn.metrics import classification_report
from ML_Algorithms.Measurement import filter_unknown_label
from Feature_Selection.Ensemble_Method import open_feature_list
from ML_Algorithms.Ensemble_Algorithms import EnsembleLearning

def main():
	parser = argparse.ArgumentParser(
		description= "Code to:\n" +
		"ML trophicMode Classification")

	req_group = parser.add_argument_group(title='REQUIRED INPUT')
	req_group.add_argument('-Data_Directory',required=True)
	req_group.add_argument('-Feature_Data_Name',required=True)
	req_group.add_argument('-Complete_Data_Name',required=True)
	req_group.add_argument('-Feature_Selection_Data_Name',required=True)
	req_group.add_argument('-Save_Path',required=True)
	req_group.add_argument('-Learning_Method',required=True,help="Supervised, SelfTraining")
	req_group.add_argument('-Tag',required=True,help="name as Ensemble--[each round]")

	opt_group = parser.add_argument_group(title='OPTIONAL INPUT')
	opt_group.add_argument('-Algorithm',default="['SupportVectorMachine','AdaBoostClassifier','RandomForestClassifier','DecisionTreeClassifier','ExtraTreesClassifier','GradientBoostingClassifier','KNeighborsClassifier','LogisticRegression','XGBClassifier','LGBMClassifier','NeuralNetworkClassifier','RadiusNeighborsClassifier']",help="include: ['RandomForestClassifier','AdaBoostClassifier','DecisionTreeClassifier','ExtraTreesClassifier','GaussianNB','GradientBoostingClassifier','KNeighborsClassifier','LogisticRegression','XGBClassifier','LGBMClassifier','NeuralNetworkClassifier','SGDClassifier','RadiusNeighborsClassifier','SupportVectorMachine','ComplementNB']")
	opt_group.add_argument('-Best_Score',default="MCC",help="MCC, F1Score, Hamming_Loss")
	opt_group.add_argument('-Hyperparameter',default="RandomizedSearch",help="No, RandomizedSearch, GridSearch")
	opt_group.add_argument('-CPU_Core',default="15")
	opt_group.add_argument('-NA_replace_value',default="-1")
	opt_group.add_argument('-Inner_Cross_Validation',default="10")
	opt_group.add_argument('-test_proportion',default="0.3")
	opt_group.add_argument('-Model_Iterations',default="12",help="due to apply 12 iteration due to include 13 types of Algorithms as default; exclude GaussianNB,ComplementNB,BernoulliNB")
	#opt_group.add_argument('-Label_Name',default="[trophicMode,guild]",help="[trophicMode , guild , growthForm , trait]")


	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(0)
	args = parser.parse_args()

	# label_name_list = ["trophicMode","guild","growthForm","trait"]
	"""
	ML_Prediction
		-- Result
			-- Model_Explanation
			-- Model_Prediction

		-- Model
			-- Basic Model
			-- Ensemble Model

		-- Feature Selection
	"""

	path = args.Data_Directory #"C:\\Users\\fabia\\Local_Work\\Data\\Feature_Production\\"
	feature_data_name = args.Feature_Data_Name#"Genome_Combination_Count_Table.forTest3.txt"
	complete_data_name = args.Complete_Data_Name
	feature_selection_data_name = args.Feature_Selection_Data_Name #"Genome_Combination_Count_Table.forTest3.txt--trophicMode--feature_selection_list--Supervised--2022_10_1.txt"
	save_path = args.Save_Path#"C:\\Users\\fabia\\Local_Work\\Data\\ML_Prediction\\"
	Learning_Method = args.Learning_Method
	Tag = args.Tag

	label_name_list = ["trophicMode"] #args.Label_Name
	ALG = eval(args.Algorithm) # ['RandomForestClassifier','AdaBoostClassifier','BernoulliNB','DecisionTreeClassifier','ExtraTreesClassifier','GaussianNB','GradientBoostingClassifier','KNeighborsClassifier','LogisticRegression','XGBClassifier','LGBMClassifier','NeuralNetworkClassifier','SGDClassifier','ComplementNB','RadiusNeighborsClassifier','SupportVectorMachine']
	print("Algorithm: ",ALG)


	NA_replace_value = args.NA_replace_value #"-1"
	test_proportion = float(args.test_proportion) # 0.3
	CPU_Core = int(args.CPU_Core) # 15
	Inner_CV = int(args.Inner_Cross_Validation) # 10
	Best_Score = args.Best_Score # MCC, F1Score, Hamming_Loss
	Hyperparameter= args.Hyperparameter # "No", "RandomizedSearch", "GridSearch"
	Model_Iterations = int(args.Model_Iterations) # 12



	if not os.path.exists(save_path+"Model/"):
		os.mkdir(save_path+"Model/")
	for ii in ["Basic_Model/","Ensemble_Model/"]:
		if not os.path.exists(save_path+"Model/"+ii):
			os.mkdir(save_path+"Model/"+ii)

	basic_model_path = save_path+"Model/Basic_Model/"
	ensemble_model_path = save_path+"Model/Ensemble_Model/"
	feature_selection_path = save_path+"Feature_Selection/"

	start_time = time()

	print("start to open feature list ...")
	trophicMode_SelectedFeatureNameInformation = open_feature_list(input_feature_selection_path=feature_selection_path,
																   input_feature_selection_list_txt=feature_selection_data_name)

	X_train_array, X_train_name, trophicMode_train_array, label_Info, na_index_train_Info = information_arangement(input_data_dir=path ,
																										input_feature_data_name=feature_data_name,
																										input_label_name_list=label_name_list)
	X_test_array, X_test_name, trophicMode_test_array, label_Info, na_index_test_Info = information_arangement(input_data_dir=path ,
																																 input_feature_data_name=complete_data_name,
																																 input_label_name_list=label_name_list)

	trophicMode_train = EnsembleLearning(input_feature_data_name=feature_data_name,
									  input_basic_model_directory = basic_model_path,
									  input_label_name="trophicMode",
									  input_na_index_Info=na_index_train_Info,
									  input_NA_replace_value = NA_replace_value,
									  input_test_proportion= test_proportion,
									  input_Learning_Type = Learning_Method,
									  input_cpu_core = CPU_Core,
									  input_inner_cv = Inner_CV,
									  input_best_score_name = Best_Score,
									  input_Hyperparameter_Method=Hyperparameter,
									  input_iterations=Model_Iterations,
									  input_feature_selected_name_information = trophicMode_SelectedFeatureNameInformation,
									  input_X_array=X_train_array,
									  input_X_name=X_train_name,
									  input_y_array=trophicMode_train_array,
									  input_y_name=label_Info["trophicMode"],
									  input_ensemble_model_directory = ensemble_model_path,
									  tag=Tag)

	trophicMode_test = EnsembleLearning(input_feature_data_name=feature_data_name,
									  input_basic_model_directory = basic_model_path,
									  input_label_name="trophicMode",
									  input_na_index_Info=na_index_test_Info,
									  input_NA_replace_value = NA_replace_value,
									  input_test_proportion= test_proportion,
									  input_Learning_Type = Learning_Method,
									  input_cpu_core = CPU_Core,
									  input_inner_cv = Inner_CV,
									  input_best_score_name = Best_Score,
									  input_Hyperparameter_Method=Hyperparameter,
									  input_iterations=Model_Iterations,
									  input_feature_selected_name_information = trophicMode_SelectedFeatureNameInformation,
									  input_X_array=X_test_array,
									  input_X_name=X_test_name,
									  input_y_array=trophicMode_test_array,
									  input_y_name=label_Info["trophicMode"],
									  input_ensemble_model_directory = ensemble_model_path,
									  tag=Tag)

	print("=======>>>>>>>>>>>>> Start to Train trophicMode Single Model ....")
	trophicMode_basic_model_information,_= trophicMode_train.train_model(input_Algorithm_Name_list=ALG)
	print(trophicMode_basic_model_information)
	print("\n\n\n")

	print("=======>>>>>>>>>>>>> get completly and partial instance ")
	# Selecte Successful Prediction Instance name
	complete_id_name, partial_id_name = trophicMode_train.matched_instance()
	print("completly instance : ",len(complete_id_name))
	print("partial instance : ",len(partial_id_name))
	with open(feature_selection_path+feature_data_name+"--"+'trophicMode'+"--complete_instance_name"+"--"+Learning_Method+"--"+Tag+".txt","w") as file:
		for i in complete_id_name:
			file.write("%s\n" % i)

	with open(feature_selection_path+feature_data_name+"--"+'trophicMode'+"--partial_instance_name"+"--"+Learning_Method+"--"+Tag+".txt","w") as file:
		for i in partial_id_name:
			file.write("%s\n" % i)
	print("completly instance and partial instance have been saved !!")
	print("\n\n\n")
	print("=======>>>>>>>>>>>>> Start to Test trophicMode Ensemble Model ....")
	for i in ["Type_1","Type_2","Type_3"]:
		Type_hard,Type_soft_mean,Type_soft_median = trophicMode_test.predict(i,True)
		for name,method_df in zip(["hard","mean","median"],[Type_hard,Type_soft_mean,Type_soft_median]): 
			print(">>>>>>>>>>>>>>>>>>>>  REPORT  <<<<<<<<<<<<<<<<<<<")
			print(">>>>>>>>>>   ",i,"  ",name,"   <<<<<<<<<<")
			y_true, y_pred = filter_unknown_label(trophicMode_test_array,np.array(method_df.iloc[:,1:]),na_index_test_Info['trophicMode'])
			print(classification_report(y_true, y_pred, target_names=label_Info['trophicMode']))
	end_time = time()
	print("==============================================")
	print("==============     Time Cost    ==============")
	print(f'{strftime("%H:%M:%S", gmtime(end_time-start_time))} seconds')
	print("==============================================")
	print("==============================================")
if __name__ == '__main__':
	main()