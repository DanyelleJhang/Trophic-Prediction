import numpy as np

def find_union_feature_index(input_seletion_information,input_X_name):
	union_seletion_information= list(set([item for sublist in input_seletion_information.values() for item in sublist]))
	sorter = np.argsort(input_X_name)
	index_array = sorter[np.searchsorted(input_X_name,union_seletion_information, sorter=sorter)]
	return index_array

def open_feature_list(input_feature_selection_path,input_feature_selection_list_txt):
	from numpy import array
	with open(input_feature_selection_path+input_feature_selection_list_txt,"r") as file:
		selected_information ={}
		for line in file:
			line = line.strip() # debug: remove \n 
			selected_information[line.split("\t")[0]] = array(line.split("\t")[1].split(","))
	return selected_information

def save_feature_list(input_feature_data_name,
					  input_label_name,
					  input_feature_selection_path,
					  input_X_name,
					  input_y_name,
					  input_feature_selected_index_information,
					  input_feature_selection_method,
					  input_tag):
	feature_selection_result_name_info = {}
	for y_i_name in input_y_name:
		feature_selection_result_name_info[y_i_name]=np.array(input_X_name)[input_feature_selected_index_information[y_i_name]]
		with open(input_feature_selection_path+input_feature_data_name+"--"+y_i_name+"--feature_selection_list"+"--"+input_feature_selection_method+"--"+input_tag+".txt","w") as file:
			for i in feature_selection_result_name_info.keys():
				feature_string = ",".join(feature_selection_result_name_info[i])
				file.write("%s\t" % i)
				file.write("%s\n" % feature_string)
	print(input_feature_data_name+"--feature_selection_list"+"--"+input_feature_selection_method+"--"+input_tag+".txt"," has been saved")
class EnsembleFeatureSelection:
	def __init__(self,
				 input_feature_data_name,
				 input_feature_selection_path,
				 input_label_name,
				 input_X_name,
				 statistic_difference_test,
				 pvalue_correction_mehthod,
				 n_iter,
				 statistic_difference,
				 permutation_type_name,
				 resample_number,
				 alternative_name,
				 correlation_test,
				 relevance_threshold,
				 pvalue,
				 n_jobs,
				 tag):
		# Discrimination Method parameters
		self.input_feature_data_name = input_feature_data_name
		self.input_feature_selection_path = input_feature_selection_path
		self.input_label_name = input_label_name
		self.input_X_name = input_X_name
		self.statistic_difference_test = statistic_difference_test
		self.pvalue_correction_mehthod = pvalue_correction_mehthod
		self.n_iter = n_iter
		# Permutation Method parameters
		self.statistic_difference = statistic_difference
		self.permutation_type_name = permutation_type_name
		self.resample_number = resample_number
		self.alternative_name = alternative_name
		# Informative Method parameters
		self.correlation_test = correlation_test
		self.relevance_threshold = relevance_threshold
		# common parameters
		self.pvalue = pvalue
		self.n_jobs = n_jobs
		self.tag = tag
	def get_result(self,input_X_array,input_y_array,input_y_array_unknown_index,input_y_name):
		from Feature_Selection.Information_Method import multiLabelInformativeColIndex
		from Feature_Selection.Discrimination_Method import multiLabelDiscriminationColIndex
		from Feature_Selection.Discrimination_Method import multiLabelPermutationDiscriminationColIndex
		from Feature_Selection.Correlation_Method import FeatureFeatureCorrelationColIndex
		# remove unknown label at first (supervise) ; remove instance
		print("[  Without remove singleton:",input_X_array.shape[1],"  ]")
		X_array_remove_unknown = np.delete(input_X_array, input_y_array_unknown_index, axis=0)
		y_array_removee_unknown = np.delete(input_y_array, input_y_array_unknown_index, axis=0)
		# remove singletom; remove feature
		from Feature_Selection.SingletonRemoval_Method import FilterZeroCountFeautureColIndex
		SingletonRemovalColIndex = FilterZeroCountFeautureColIndex(X_array_remove_unknown)
		# remove feature about genome information
		SingletonRemovalColIndex = np.add(SingletonRemovalColIndex, 1)
		X_array_remove_unknown = X_array_remove_unknown[:,SingletonRemovalColIndex] # replace so as to save memory
		selected_X_name = np.array(self.input_X_name)[SingletonRemovalColIndex]
		# execute supervised feature selection  
		print("[  After remove singleton:",len(SingletonRemovalColIndex),"  ]")
		LabelDiscrimination_Results = multiLabelDiscriminationColIndex(X_array_remove_unknown,y_array_removee_unknown,input_y_name,
																	   self.statistic_difference_test,
																	   self.pvalue,
																	   self.pvalue_correction_mehthod,
																	   self.n_iter,
																	   self.n_jobs)
		save_1 = save_feature_list(input_feature_data_name=self.input_feature_data_name,
									input_label_name=self.input_label_name,
									input_feature_selection_path=self.input_feature_selection_path,
									input_X_name=selected_X_name,
									input_y_name=input_y_name,
									input_feature_selected_index_information=LabelDiscrimination_Results,
									input_feature_selection_method="LabelDiscrimination",
									input_tag=self.tag)
		LabelPermutation_Results = multiLabelPermutationDiscriminationColIndex(X_array_remove_unknown,y_array_removee_unknown,input_y_name,
																			   self.statistic_difference,
																			   self.permutation_type_name,
																			   self.resample_number,
																			   self.alternative_name,
																			   self.pvalue,
																			   self.n_jobs)
		save_2 = save_feature_list(input_feature_data_name=self.input_feature_data_name,
									input_label_name=self.input_label_name,
									input_feature_selection_path=self.input_feature_selection_path,
									input_X_name=selected_X_name,
									input_y_name=input_y_name,
									input_feature_selected_index_information=LabelPermutation_Results,
									input_feature_selection_method="LabelPermutation",
									input_tag=self.tag)

		LabelInformative_Results = multiLabelInformativeColIndex(X_array_remove_unknown,y_array_removee_unknown,input_y_name,
																 self.n_jobs)
		save_3 = save_feature_list(input_feature_data_name=self.input_feature_data_name,
									input_label_name=self.input_label_name,
									input_feature_selection_path=self.input_feature_selection_path,
									input_X_name=selected_X_name,
									input_y_name=input_y_name,
									input_feature_selected_index_information=LabelInformative_Results,
									input_feature_selection_method="LabelInformative",
									input_tag=self.tag)

		# restore union of feature selection result
		feature_selection_union_result = {}
		for y_i_name in input_y_name:
			print("[ ",y_i_name," ]")
			LabelDiscrimination_Results_y_i = LabelDiscrimination_Results[y_i_name]
			print("LabelDiscrimination: ",len(LabelDiscrimination_Results_y_i))
			LabelPermutation_Results_y_i = LabelPermutation_Results[y_i_name]
			print("LabelPermutation: ",len(LabelPermutation_Results_y_i))
			LabelInformative_Results_y_i = LabelInformative_Results[y_i_name]
			print("LabelInformative: ",len(LabelInformative_Results_y_i))
			union_result = list(set(LabelDiscrimination_Results_y_i+LabelPermutation_Results_y_i+LabelInformative_Results_y_i))
			print("Union result: ", len(union_result))
			feature_selection_union_result[y_i_name]=union_result
		# execute inner indepent feature selection
		feature_selection_result_name_info = {}
		for y_i_name in input_y_name:
			feature_selection_union_result_i = feature_selection_union_result[y_i_name]
			FeatureFeatureCorrelatio_Result_i = FeatureFeatureCorrelationColIndex(X_array_remove_unknown[:,feature_selection_union_result_i],
																				  self.correlation_test,
																				  self.relevance_threshold)
			feature_selection_result_name_info[y_i_name]=selected_X_name[FeatureFeatureCorrelatio_Result_i]
			print("[ ",y_i_name," ]")
			print("Feature Independence: ",len(FeatureFeatureCorrelatio_Result_i))
		return feature_selection_result_name_info