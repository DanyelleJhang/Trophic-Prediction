import shap
import pandas as pd
import sys, os
from joblib import load,Parallel, delayed


def transformation_dataframe(input_X_array,input_X_name):
	feature_dataframe = pd.DataFrame(input_X_array, columns=input_X_name)
	genome_file_name_expand = feature_dataframe["genome_file_name"].str.split("--", expand = True).rename(columns = {0:"genome_file_name",1:"remain_proportion",2:"round"})
	feature_dataframe = feature_dataframe.drop(columns=["genome_file_name"])
	feature_dataframe = pd.concat([genome_file_name_expand,feature_dataframe],axis=1)
	return feature_dataframe

def singleLabelsingleModelExplaination(input_ml_prediction_path,
										input_y_i_model_name,
										input_y_i_X_train,
										input_y_i_X_test,
										input_X_test_ID,
										input_based_on_probability,
										input_tag):
	y_i_single_model_name = input_y_i_model_name
	trained_clf= load(y_i_single_model_name)
	print("explaining ...",y_i_single_model_name.split("/")[-1])
	if not os.path.exists(input_ml_prediction_path+"Result/"):
		os.mkdir(input_ml_prediction_path+"Result/")
	if not os.path.exists(input_ml_prediction_path+"Result/"+"Model_Explanation/"):
		os.mkdir(input_ml_prediction_path+"Result/"+"Model_Explanation/")

	if input_based_on_probability == True:
		explainer = shap.KernelExplainer(model = trained_clf.predict_proba,data=input_y_i_X_train)
		shap_values = explainer.shap_values(X=input_y_i_X_test)

		neg_proba_array = shap_values[0]
		neg_base_value = explainer.expected_value[0]
		pos_proba_array = shap_values[1]
		pos_base_value = explainer.expected_value[1]

		pos_df = pd.DataFrame(pos_proba_array,columns=input_y_i_X_test.columns)
		pos_df.insert(0, "Prediction_Type", 1, True)
		neg_df = pd.DataFrame(neg_proba_array,columns=input_y_i_X_test.columns)
		neg_df.insert(0, "Prediction_Type", 0, True)

		shap_value_df = pd.concat([pos_df,neg_df],axis=0).reset_index(drop=True)
		shap_value_df = pd.concat([input_X_test_ID,shap_value_df],axis=1)
		shap_value_df.to_csv(input_ml_prediction_path+"Result/"+"Model_Explanation/"+y_i_single_model_name.split("/")[-1]+"_proba_explaination--"+input_tag+".txt",sep="\t",header=True,index=False)

		if not os.path.isfile(input_ml_prediction_path+"Result/"+"Model_Explanation/"+'proba_base_value.txt'):
			out2 = open(input_ml_prediction_path+"Result/"+"Model_Explanation/"+'proba_base_value.txt', 'a')
			out2.write('Model_name\tPositive_Value\tNegative_Value\tTag')
			out2.close()

		out2 = open(input_ml_prediction_path+"Result/"+"Model_Explanation/"+'proba_base_value.txt', 'a')
		out2.write('\n%s\t%s\t%s\t%s' % (y_i_single_model_name.split("/")[-1],pos_base_value,neg_base_value,input_tag))
		out2.close()
		print("DONE")
		return y_i_single_model_name.split("/")[-1]
	else:
		explainer = shap.KernelExplainer(model = trained_clf.predict,data=input_y_i_X_train)
		shap_values = explainer.shap_values(X=input_y_i_X_test)
		shap_value_df = pd.DataFrame(shap_values,columns=input_y_i_X_test.columns)
		shap_value_df = pd.concat([input_X_test_ID,shap_value_df],axis=1)
		base_value = explainer.expected_value
		shap_value_df.to_csv(input_ml_prediction_path+"Result/"+"Model_Explanation/"+y_i_single_model_name.split("/")[-1]+"_binary_explaination--"+input_tag+".txt",sep="\t",header=True,index=False)

		if not os.path.isfile(input_ml_prediction_path+"Result/"+"Model_Explanation/"+'binary_base_value.txt'):
			out2 = open(input_ml_prediction_path+"Result/"+"Model_Explanation/"+'binary_base_value.txt', 'a')
			out2.write('Model_name\tBinary_Value\tTag')
			out2.close()

		out2 = open(input_ml_prediction_path+"Result/"+"Model_Explanation/"+'binary_base_value.txt', 'a')
		out2.write('\n%s\t%s\t%s' % (y_i_single_model_name.split("/")[-1],base_value,input_tag))
		out2.close()
		print("DONE")
		return y_i_single_model_name.split("/")[-1]


def singleLabelMulitModelExplaination(input_ml_prediction_path,
									  input_incomplete_dataframe,
									  input_complete_dataframe,
									  input_ensemble_model_list,
									  input_feature_selection_info,
									  input_y_name,
									  input_based_on_probability,
									  input_stratified_sampling,
									  input_tag,
									  n_jobs):
	if input_stratified_sampling == True:
		y_i_X_train = input_incomplete_dataframe.groupby(["genome_file_name","remain_proportion"]).sample(n=1).reset_index(drop=True)[input_feature_selection_info[input_y_name]]
	else:
		y_i_X_train = input_incomplete_dataframe.groupby(["genome_file_name"]).sample(n=1).reset_index(drop=True)[input_feature_selection_info[input_y_name]]	
	y_i_X_test = input_complete_dataframe[input_feature_selection_info[input_y_name]]
	X_test_ID = input_complete_dataframe["genome_file_name"]
	y_i_model_list = list(filter(lambda x: input_y_name in x,input_ensemble_model_list))
	save = Parallel(n_jobs=n_jobs)(delayed(singleLabelsingleModelExplaination)(input_ml_prediction_path=input_ml_prediction_path,
																			   input_y_i_model_name=y_i_model_list[i],
																			   input_y_i_X_train=y_i_X_train,
																			   input_y_i_X_test=y_i_X_test,
																			   input_X_test_ID=X_test_ID,
																			   input_based_on_probability=input_based_on_probability,
																			   input_tag=input_tag) for i in range(len(y_i_model_list)))
	return save



def multiLabelMulitModelExplaination(input_ml_prediction_path,
									 input_incomplete_X_array, 
									 input_incomplete_X_name,
									 input_complete_X_array,
									 input_complete_X_name,
									 input_y_name_list,
									 input_ensemble_model_list,
									 input_feature_selection_info,
									 input_based_on_probability,
									 input_stratified_sampling,
									 input_tag,
									 n_jobs):
	incomplete_dataframe = transformation_dataframe(input_incomplete_X_array, input_incomplete_X_name)
	complete_dataframe = pd.DataFrame(input_complete_X_array, columns=input_complete_X_name)
	save = Parallel(n_jobs=n_jobs)(delayed(singleLabelMulitModelExplaination)(input_ml_prediction_path=input_ml_prediction_path,
																			  input_incomplete_dataframe=incomplete_dataframe,
																			  input_complete_dataframe=complete_dataframe,
																			  input_ensemble_model_list=input_ensemble_model_list,
																			  input_feature_selection_info=input_feature_selection_info,
																			  input_y_name = input_y_name_list[i],
																			  input_based_on_probability=input_based_on_probability,
																			  input_stratified_sampling=input_stratified_sampling,
																			  input_tag=input_tag,
																			  n_jobs=n_jobs) for i in range(len(input_y_name_list))
																					  )
	return save

## 如果搞不懂看一下這個圖
## 一個MODEL的ROW都從BASE VALUE相加
# shap.initjs()
# shap.force_plot(explainer.expected_value[0], shap_values[0][1,:], y_i_test.iloc[1,:])



