import os,argparse,sys
from time import time, strftime, gmtime
from Feature_Production.Data_Management import information_arangement,reassign_guild
from Feature_Selection.Ensemble_Method import open_feature_list
from ML_Algorithms.Explaination import multiLabelMulitModelExplaination

parser = argparse.ArgumentParser(
	description= "Code to:\n" +
	"ML guildMode Explaination")

req_group = parser.add_argument_group(title='REQUIRED INPUT')
req_group.add_argument('-Data_Directory',required=True)
req_group.add_argument('-Feature_Data_Name',required=True)
req_group.add_argument('-Complete_Data_Name',required=True)
req_group.add_argument('-Feature_Selection_Data_Name',required=True)
req_group.add_argument('-Ensemble_Model_Data_Name',required=True)
req_group.add_argument('-Save_Path',required=True)
req_group.add_argument('-Tag',required=True,help="name as Ensemble--[each round]")

opt_group = parser.add_argument_group(title='OPTIONAL INPUT')
opt_group.add_argument('-Probability',default="True")
opt_group.add_argument('-Stratified_Sampling',default="True")
opt_group.add_argument('-CPU_Core',default="15")

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(0)
args = parser.parse_args()

path = args.Data_Directory #"C:\\Users\\fabia\\Local_Work\\Data\\Feature_Production\\"
feature_data_name = args.Feature_Data_Name#"Genome_Combination_Count_Table.forTest3.txt"
complete_data_name = args.Complete_Data_Name
feature_selection_data_name = args.Feature_Selection_Data_Name #"Genome_Combination_Count_Table.forTest3.txt--trophicMode--feature_selection_list--Supervised--2022_10_1.txt"
ensemble_model_data_name = args.Ensemble_Model_Data_Name
save_path = args.Save_Path #"C:\\Users\\fabia\\Local_Work\\Data\\ML_Prediction\\"
Tag = args.Tag
based_on_probability = eval(args.Probability) # "True"
stratified_sampling = eval(args.Stratified_Sampling) # True
CPU_Core = int(args.CPU_Core) # 15
label_name_list = ["guild"] #args.Label_Name
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


basic_model_path = save_path+"Model/Basic_Model/"
ensemble_model_path = save_path+"Model/Ensemble_Model/"
feature_selection_path = save_path+"Feature_Selection/"



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
id_name = []
with open(feature_selection_path+instance_selection_data_name, 'r') as file:
	for line in file.readlines():
		id_name.append(line.strip()) # remove '\n' at the end
id_name = np.array(id_name)
X_train_array,guild_train_array,na_index_train_Info = reassign_guild(id_name,X_train_array,guild_train_array)




with open(ensemble_model_path+ensemble_model_data_name,"r") as file:
	ensemble_model_list = list(map(lambda x: x.split("\n")[0],file.readlines()))
	ensemble_model_list = list(map(lambda x: basic_model_path+x.split("/")[-1],ensemble_model_list))


guild_SAVE = multiLabelMulitModelExplaination(input_ml_prediction_path=basic_model_path,
													input_incomplete_X_array=X_train_array, 
													input_incomplete_X_name=X_train_name,
													input_complete_X_array=X_test_array,
													input_complete_X_name=X_test_name,
													input_y_name_list=label_Info['guild'],
													input_ensemble_model_list=ensemble_model_list,
													input_feature_selection_info=guild_SelectedFeatureNameInformation,
													input_based_on_probability=based_on_probability,
													input_stratified_sampling=stratified_sampling,
													input_tag=Tag,
													n_jobs=CPU_Core)


end_time = time()
print("==============================================")
print("==============     Time Cost    ==============")
print(f'{strftime("%H:%M:%S", gmtime(end_time-start_time))} seconds')
print("==============================================")
print("==============================================")