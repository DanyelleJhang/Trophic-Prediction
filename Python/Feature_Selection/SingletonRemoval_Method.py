import pandas as pd
import sys, os
call_module_for_bash = os.path.dirname(str(os.getcwd())) + "/Python"
sys.path.insert(1, call_module_for_bash)
from utility.usage_measurement import hardware_usage

@hardware_usage
def FilterZeroCountFeautureColIndex(input_X_array):
	print("remove singleton ....")
	input_data = pd.DataFrame(input_X_array,columns=["genome_file_name"] + list(range(0,input_X_array.shape[1]-1,1)))
	######## input_data
	##### genome_file_name   WD40&&MFS_1   WD40&&Pkinase   WD40&&Zn_clus   ... ... ...
	##### 'Absidia_glauca_gca_900079185.AG_v1.pep.all.fa--0.95--round_1' ... ... ... 
	input_data["Genome_Name"] = input_data["genome_file_name"].str.split("--",expand=True).iloc[:,0]
	######## input_data
	##### genome_file_name   WD40&&MFS_1   WD40&&Pkinase   WD40&&Zn_clus   ... ... ... Genome_Name
	##### 'Absidia_glauca_gca_900079185.AG_v1.pep.all.fa--0.95--round_1' ... ... ... 'Absidia_glauca_gca_900079185.AG_v1.pep.all.fa'
	genome_name_list = list(set(input_data["Genome_Name"]))
	feature_name_list = input_data.columns.drop(['genome_file_name','Genome_Name']).tolist()
	remain_feature_index = []
	for feature_name in feature_name_list:
		single_feature_count_by_genome = input_data.loc[:,["Genome_Name",feature_name]].groupby(["Genome_Name"]).sum()
		# general number of genome - single and specific genome - single but belonged to other genome
		at_least_two_different_genome_num = len(genome_name_list) - 1 - 1
		single_feature_count_by_genome_freq = single_feature_count_by_genome.value_counts()
		zero = 0
		if zero in list(map(lambda x: x[0],single_feature_count_by_genome_freq.index)):
			if single_feature_count_by_genome_freq[zero] <= at_least_two_different_genome_num:
				remain_feature_index.append(feature_name)
		else:
			remain_feature_index.append(feature_name)
	del input_data
	print("remove singleton done !")
	return remain_feature_index