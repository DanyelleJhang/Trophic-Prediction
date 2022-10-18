import pandas as pd
import numpy as np
import joblib
import os,argparse,sys
from time import time, strftime, gmtime
# if not sys.warnoptions:
# 	import warnings
# 	warnings.simplefilter("ignore")
from Feature_Production.Data_Management import information_arangement,reassign_guild
from sklearn.metrics import classification_report
from ML_Algorithms.Measurement import filter_unknown_label
from Feature_Selection.Ensemble_Method import open_feature_list
from ML_Algorithms.Ensemble_Algorithms import EnsembleLearning

def main():
	parser = argparse.ArgumentParser(
		description= "Code to:\n" +
		"ML guildMode Classification")

	req_group = parser.add_argument_group(title='REQUIRED INPUT')
	req_group.add_argument('-Data_Directory',required=True)
	req_group.add_argument('-Feature_Data_Name',required=True)
	req_group.add_argument('-Complete_Data_Name',required=True)
	req_group.add_argument('-Feature_Selection_Data_Name',required=True)
	req_group.add_argument('-Instance_Selection_Data_Name',required=True)
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
	instance_selection_data_name = args.Instance_Selection_Data_Name  #"Genome_Combination_Count_Table.forTest3.txt--trophicMode--partial_instance_name--Supervised--2022_10_1.txt"

	save_path = args.Save_Path#"C:\\Users\\fabia\\Local_Work\\Data\\ML_Prediction\\"
	Learning_Method = args.Learning_Method
	Tag = args.Tag

	label_name_list = ["guild"] #args.Label_Name
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
	print(basic_model_path)
	print(ensemble_model_path)
	print(feature_selection_path)

	start_time = time()



	print("start to open feature list ...")
	guild_SelectedFeatureNameInformation = open_feature_list(input_feature_selection_path=feature_selection_path,
																   input_feature_selection_list_txt=feature_selection_data_name)

	X_train_array, X_train_name, guild_train_array, label_Info, na_index_train_Info = information_arangement(input_data_dir=path,
																											 input_feature_data_name=feature_data_name,
																											 input_label_name_list=label_name_list)
	X_test_array, X_test_name, guild_test_array, label_Info, na_index_test_Info = information_arangement(input_data_dir=path,
																										 input_feature_data_name=complete_data_name,
																										 input_label_name_list=label_name_list)

	print("start to reassign instance ...")

	id_name = []
	with open(feature_selection_path+instance_selection_data_name, 'r') as file:
		for line in file.readlines():
			id_name.append(line.strip()) # remove '\n' at the end
	id_name = np.array(id_name)
	X_train_array,guild_train_array,na_index_train_Info = reassign_guild(id_name,X_train_array,guild_train_array)

	# reivse to only inclue partial intead of complete; completly filter may cause infini loop
	print("\n\n")
	print("=======>>>>>>>>>>>>> Start to train guild Single Model ....")
	print("========================================")
	print(" True Positive instance: ",X_train_array.shape[0])
	print("========================================")
	# due to require different instance space
	# input_y_array_unknown_index = na_index_Info['guild']
	# will be replace by id_na_index_Info['guild']
	# becuase such parameter is used for delete unknown (useless) instance
	# howerever, we would like to remain Selected Instance
	# so we create an array is belonged to Unselected Instance usedless instance including unknown instance
	print("Train "," True Positive instance Model...")
	guild_train = EnsembleLearning(input_feature_data_name=feature_data_name,
								input_basic_model_directory = basic_model_path,
								input_label_name="guild",
								input_na_index_Info=na_index_train_Info,
								input_NA_replace_value = NA_replace_value,
								input_test_proportion= test_proportion,
								input_Learning_Type = Learning_Method,
								input_cpu_core = CPU_Core,
								input_inner_cv = Inner_CV,
								input_best_score_name = Best_Score,
								input_Hyperparameter_Method=Hyperparameter,
								input_iterations=Model_Iterations,
								input_feature_selected_name_information = guild_SelectedFeatureNameInformation,
								input_X_array=X_train_array,
								input_X_name=X_train_name,
								input_y_array=guild_train_array,
								input_y_name=label_Info["guild"],
								input_ensemble_model_directory = ensemble_model_path,
								tag=Tag)
	guild_basic_model_information,_= guild_train.train_model(input_Algorithm_Name_list=ALG)
	print(guild_basic_model_information)
	guild_test = EnsembleLearning(input_feature_data_name=feature_data_name,
								input_basic_model_directory = basic_model_path,
								input_label_name="guild",
								input_na_index_Info=na_index_test_Info,
								input_NA_replace_value = NA_replace_value,
								input_test_proportion= 1.0,
								input_Learning_Type = Learning_Method,
								input_cpu_core = CPU_Core,
								input_inner_cv = Inner_CV,
								input_best_score_name = Best_Score,
								input_Hyperparameter_Method=Hyperparameter,
								input_iterations=Model_Iterations,
								input_feature_selected_name_information = guild_SelectedFeatureNameInformation,
								input_X_array=X_test_array,
								input_X_name=X_test_name,
								input_y_array=guild_test_array,
								input_y_name=label_Info["guild"],
								input_ensemble_model_directory = ensemble_model_path,
								tag=Tag)
	print("\n\n\n")
	print("=======>>>>>>>>>>>>> Start to build guild Ensemble Model ....")
	for i in ["Type_1","Type_2","Type_3"]:
		print("Build "," ",i, " Ensemble Model")
		Type_hard,Type_soft_mean,Type_soft_median = guild_test.predict(i,True)
		for name,method_df in zip(["hard","mean","median"],[Type_hard,Type_soft_mean,Type_soft_median]): 
			print(">>>>>>>>>>>>>>>>>>>>  REPORT  <<<<<<<<<<<<<<<<<<<")
			print(">>>>>>>>>>   ",i,"  ",name,"   <<<<<<<<<<")
			y_true, y_pred = filter_unknown_label(guild_test_array,np.array(method_df.iloc[:,1:]),na_index_test_Info['guild'])
			print(classification_report(y_true, y_pred, target_names=label_Info['guild']))
	print("\n\n\n")

	end_time = time()
	print("==============================================")
	print("==============     Time Cost    ==============")
	print(f'{strftime("%H:%M:%S", gmtime(end_time-start_time))} seconds')
	print("==============================================")
	print("==============================================")
if __name__ == '__main__':
	main()