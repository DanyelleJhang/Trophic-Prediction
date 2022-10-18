import os,argparse,sys
from time import time, strftime, gmtime
if not sys.warnoptions:
	import warnings
	warnings.simplefilter("ignore")
import pandas as pd
from Feature_Production.Data_Management import information_arangement
from Feature_Selection.Ensemble_Method import EnsembleFeatureSelection

def main():
	parser = argparse.ArgumentParser(
		description= "Code to:\n" +
		"Feature Selection on guildMode")

	req_group = parser.add_argument_group(title='REQUIRED INPUT')
	req_group.add_argument('-Data_Directory',required=True)
	req_group.add_argument('-Feature_Data_Name',required=True)
	req_group.add_argument('-Save_Path',required=True)
	req_group.add_argument('-Tag',required=True,help="name as Ensemble--[each round]")
	# Feature Selection
	fs_group = parser.add_argument_group(title='Ensemble Feauture Selection OPTIONAL INPUT')
	fs_group.add_argument('-CPU_Core',default="15")
	fs_group.add_argument('-Statistic_Difference_Test',default="independent_WilcoxonRankSum",help="independent_Ttest, paired_Ttest, independent_WilcoxonRankSum, paired_WilcoxonSignedRank, MannWhitney_U_Rank, KolmogorovSmirnov")
	fs_group.add_argument('-pvalue_Correction_Mehthod',default="bonferroni",help="bonferroni:one-step correction; sidak: one-step correction; holm-sidak: step down method using Sidak adjustments; holm: step-down method using Bonferroni adjustments; simes-hochberg: step-up method (independent); hommel: closed method based on Simes tests (non-negative); fdr_bh: Benjamini/Hochberg (non-negative); fdr_by: Benjamini/Yekutieli (negative); fdr_tsbh: two stage fdr correction (non-negative); fdr_tsbky: two stage fdr correction (non-negative)")
	fs_group.add_argument('-Statistic_Difference',default="median", help="median, mean")
	fs_group.add_argument('-Permutation_Type_Name',default="independent", help ="independent, samples, pairings")
	fs_group.add_argument('-Resample_Number',default="1000",help="default: 1000") #remember to change value
	fs_group.add_argument('-Alternative_Name',default='two-sided',help="greater, less, two-sided")
	fs_group.add_argument('-Correlation_Test',default="Spearman",help="Pearson, Spearman, KendallTau, PointBiserial")
	fs_group.add_argument('-Relevance_Threshold',default="0.3",help="less than 0.3 due to study suggestion")
	fs_group.add_argument('-Statistic_Iterations',default="30",help=" default: 30 ")
	fs_group.add_argument('-p_value',default="0.05",help=" default: 0.05 ")
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(0)
	args = parser.parse_args()

	path = args.Data_Directory #"C:\\Users\\fabia\\Local_Work\\Data\\Feature_Production\\"
	feature_data_name = args.Feature_Data_Name#"Genome_Combination_Count_Table.forTest3.txt"
	save_path = args.Save_Path#"C:\\Users\\fabia\\Local_Work\\Data\\ML_Prediction\\"
	Tag = args.Tag

	label_name_list = ["guild"] #args.Label_Name


	CPU_Core = int(args.CPU_Core) # 15
	Statistic_Difference_Test = args.Statistic_Difference_Test
	pvalue_Correction_Mehthod = args.pvalue_Correction_Mehthod
	Statistic_Iterations = int(args.Statistic_Iterations)
	Statistic_Difference = args.Statistic_Difference
	Permutation_Type_Name = args.Permutation_Type_Name
	Resample_Number = int(args.Resample_Number)
	Alternative_Name = args.Alternative_Name
	Correlation_Test = args.Correlation_Test
	Relevance_Threshold = float(args.Relevance_Threshold)
	p_value = float(args.p_value)



	if not os.path.exists(save_path+"Feature_Selection/"):
		os.mkdir(save_path+"Feature_Selection/")
	feature_selection_path = save_path+"Feature_Selection/"


	start_time = time()

	X_train_array, X_train_name, guild_train_array, label_Info, na_index_train_Info = information_arangement(input_data_dir=path ,
																										input_feature_data_name=feature_data_name,
																										input_label_name_list=label_name_list)



	print("=======>>>>>>>>>>>>> Start to run guild feature selction ....")
	# due to feature selection of partial selected array and completly selected array comes error
	# so we decide to do fully feature selection based on original array
	guild_EFS = EnsembleFeatureSelection(input_feature_data_name=feature_data_name,
											input_feature_selection_path=feature_selection_path,
											input_label_name="guild",
											input_X_name=X_train_name,
											statistic_difference_test = Statistic_Difference_Test,
											pvalue_correction_mehthod = pvalue_Correction_Mehthod,
											n_iter = Statistic_Iterations,
											statistic_difference = Statistic_Difference,
											permutation_type_name = Permutation_Type_Name,
											resample_number = Resample_Number,
											alternative_name = Alternative_Name,
											correlation_test = Correlation_Test,
											relevance_threshold = Relevance_Threshold,
											pvalue = p_value,
											n_jobs = CPU_Core,
											tag=Tag)
	guild_SelectedFeatureNameInformation = guild_EFS.get_result(input_X_array= X_train_array,
																input_y_array= guild_train_array,
																input_y_array_unknown_index= na_index_train_Info['guild'],
																input_y_name= label_Info['guild'])
	with open(feature_selection_path+feature_data_name+"--"+'guild'+"--feature_selection_list"+"--"+Tag+".txt","w") as file:
		for i in guild_SelectedFeatureNameInformation.keys():
			feature_string = ",".join(guild_SelectedFeatureNameInformation[i])
			file.write("%s\t" % i)
			file.write("%s\n" % feature_string)

	selected_list = []
	for _,k in guild_SelectedFeatureNameInformation.items():
		selected_list.append(k)
	selected_feature_list= list(set([item for sublist in selected_list for item in sublist]))
	union_selection_dataframe = pd.DataFrame(X_train_array,columns=X_train_name).loc[:,["genome_file_name"]+selected_feature_list]
	union_selection_name = feature_data_name.split(".txt")[0]+".feature_selection."+label_name_list[0]+"."+Tag+".txt"
	union_selection_dataframe.to_csv(path+union_selection_name,sep="\t",header=True,index=False)


	print("\n\n\n")
	end_time = time()
	print("==============================================")
	print("==============     Time Cost    ==============")
	print(f'{strftime("%H:%M:%S", gmtime(end_time-start_time))} seconds')
	print("==============================================")
	print("==============================================")
if __name__ == '__main__':
	main()