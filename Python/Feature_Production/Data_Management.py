import pandas as pd
#import modin.pandas as pd
import numpy as np
from glob import glob
import os, sys
# import time
call_module_for_bash = os.path.dirname(str(os.getcwd())) + "/Python"
sys.path.insert(1, call_module_for_bash)
from utility.usage_measurement import hardware_usage

def dismantle_data(input_data_dir,input_feature_data_name,input_label_name_list):
	import warnings
	warnings.filterwarnings('ignore')
	chunk = pd.read_csv(str(input_data_dir)+str(input_feature_data_name),sep="\t",chunksize=1000,low_memory=False,dtype=float,converters={"genome_file_name":str,"Calculation_Characteristics":str})
	#end = time.time()
	#print("Read csv with chunks: ",(end-start),"sec")
	Feature = pd.concat(chunk,ignore_index=True)
	# We require to do multi label and hierachical classification
	# So Encoding as Flag pattern or MultiClass pattern was not suitable for our requirement
	label_file_name_list = glob(input_data_dir+"*"+'_Multi_Label.txt')
	
	
	labeldata_info = {}
	labeldata = []
	for label_name in input_label_name_list:
		sigle_labeldata = pd.read_csv([i for i in label_file_name_list if label_name in i][0],sep="\t").set_index(["genome_file_name","confidenceRanking"])
		labeldata_info[label_name]=sigle_labeldata.columns[-1]
		labeldata.append(sigle_labeldata)
	del sigle_labeldata
	labeldata = pd.concat(labeldata,axis=1,sort=False).reset_index()
	
	Feature_info = Feature["genome_file_name"].str.split("--", expand=True).rename(columns = {0:'genome_file_name', 1:'Gene_Remain_Proportion',2:"iteration"})
	Feature["genome_file_name"] = Feature_info["genome_file_name"]
	Feature.set_index('genome_file_name',inplace=True) # inplace: No need to create new object
	labeldata.set_index('genome_file_name',inplace=True)
	"""
	## Testing Logic Example

	df1 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'oo'],
						'value_2':["a","b","c",444]})
	df2 = pd.DataFrame({'key': ['foo', 'bar', 'baz', 'foo', 'oo', 'oo','baz','baz', 'bar'],
						'VALUE_3':['f', 'b', 'z', 'f', 'o', 'o','z','z', 'b']})
	df1.set_index('key',inplace=True)
	df2.set_index('key',inplace=True)
	df1.merge(df2,right_on="key",left_on="key")
	"""
	label_feature_data = labeldata.merge(Feature,right_on=["genome_file_name"],left_on=["genome_file_name"])
	try:
		label_feature_data.index = Feature_info.agg('--'.join, axis=1)
	except:
		label_feature_data.index = label_feature_data.index
	del Feature
	del labeldata
	del Feature_info
	
	drop_col_list = []
	for label_name in input_label_name_list:
		drop_col_list.append(labeldata_info[label_name])
	#drop_col_list.append("confidenceRanking")
	
	feature_dataframe = label_feature_data.drop(drop_col_list, axis=1).reset_index().rename(columns={"index":"genome_file_name"})

	label_dataframe_in_list =[]
	label_name_list = []
	for label_name in input_label_name_list:
		label_dataframe = label_feature_data[[labeldata_info[label_name]]].reset_index().rename(columns={"index":"genome_file_name"})
		label_dataframe_in_list.append(label_dataframe)
		label_name_list.append(label_name)
	#trophicMode_dataframe = label_feature_data[[labeldata_info["trophicMode"]]].reset_index().rename(columns={"index":"genome_file_name"})
	#guild_dataframe = label_feature_data[[labeldata_info["guild"]]].reset_index().rename(columns={"index":"genome_file_name"})
	#growthForm_dataframe = label_feature_data[[labeldata_info["growthForm"]]].reset_index().rename(columns={"index":"genome_file_name"})
	#trait_dataframe = label_feature_data[[labeldata_info["trait"]]].reset_index().rename(columns={"index":"genome_file_name"})
	del input_label_name_list
	del label_feature_data
	del labeldata_info
	
	return feature_dataframe, label_dataframe_in_list, label_name_list

@hardware_usage
def information_arangement(input_data_dir,input_feature_data_name,input_label_name_list):
	X,Ys,label_names = dismantle_data(input_data_dir,input_feature_data_name,input_label_name_list)
	
	X = X.fillna(0)
	X_Array = X.to_numpy()
	X_Name = list(X.columns)
	
	y_Arrays = []
	na_index_info = {}
	y_Names = {}
	for y,label_name in zip(Ys,label_names):
		y_name = eval(y.columns[1])
		
		y_arr = np.stack((y.iloc[:,1].apply(eval).values))
		na_index_arr = np.where(y_arr.sum(axis=1) == 0)[0]
		
		# y_arr = np.delete(y_arr, na_index_arr, axis=0) 
		# X_Array = np.delete(X_Array, na_index_arr, axis=0) # out of bound due to different na index
		# y_arr = y_arr.flatten()
		na_index_info[label_name]=na_index_arr
		y_Names[label_name]=y_name
		y_Arrays.append(y_arr)
	return X_Array,X_Name,*y_Arrays,y_Names,na_index_info


# Inorder to ensure containing at least !! ONE !! Class(Label)
# Otherwise, the Classification Algorithms will CRASH while empty Class(Label) compared to exisiting Class(Label)
def train_test_split_at_least_one_label(X,y,input_Label_type,testing_size,input_random_state=None):
	from sklearn.model_selection import train_test_split
	if input_random_state == None:
		if input_Label_type == "MultiLabel":
			while True:
				i = np.random.randint(low=0, high=1000000, size=1, dtype=int)[0]
				feature_train, feature_test, label_train, label_test = train_test_split(X, y,test_size=testing_size, random_state=i)
				if np.all(label_train.sum(axis=0)>0) == True:
					break # 連 else 都會直接跳掉
				else:
					continue
		if input_Label_type == "MultiClass":
			while True:
				i = np.random.randint(low=0, high=1000000, size=1, dtype=int)[0]
				feature_train, feature_test, label_train, label_test = train_test_split(X, y,test_size=testing_size, random_state=i)
				if set(label_train) == set(Label_test):
					break # 連 else 都會直接跳掉
				else:
					continue
		return feature_train, feature_test, label_train, label_test, i 
	else:
		feature_train, feature_test, label_train, label_test = train_test_split(X, y,test_size=testing_size, random_state=input_random_state)
		return feature_train, feature_test, label_train, label_test, input_random_state

def encode_unknown_label(label_array,input_NA_replace_value):
	import numpy as np
	label_array = label_array.astype('float')
	na_index_arr = np.where(label_array.sum(axis=1) == 0)[0]
	if input_NA_replace_value == "NA":
		NA_replace_value = np.nan 
	else:
		NA_replace_value = eval(input_NA_replace_value)
	label_array[na_index_arr] = NA_replace_value
	return label_array,na_index_arr
	
def MultiClass_Information(true_test_array,preedict_test_array,predict_probability_array,label_name_list,Class_Name):
	Class_Info = pd.DataFrame(
		np.concatenate(
			(np.stack((true_test_array,preedict_test_array), axis=-1),
			 predict_probability_array),axis=1),
		columns=["True_"+Class_Name,"Predict_"+Class_Name]+list(map(lambda x: "Probability of "+x,label_name_list)
															   )
	)
	return Class_Info

def MultiLabel_Information(true_test_array,predict_probability_array,label_name_list):
	## predict_probability_array
	## ==> (y_name,sample_number,proba)
	## ==> proba = [probability of 0, probability of 1]
	DF = []
	for i in range(len(true_test_array)):
		Negative_Class = pd.DataFrame([np.array(predict_probability_array)[:,i,:][:,0]],columns=list(map(lambda x: "Negative Probability of "+x,label_name_list)))
		Positive_Class = pd.DataFrame([np.array(predict_probability_array)[:,i,:][:,1]],columns=list(map(lambda x: "Positive Probability of "+x,label_name_list)))
		class_probability = pd.concat([Positive_Class,Negative_Class],axis=1)
		DF.append(class_probability)
	Class_Probability = pd.concat(DF).reset_index(drop=True)
	Class_True = pd.DataFrame(true_test_array,columns=list(map(lambda x: "True Class of "+x, label_name_list)))
	Class_Info = pd.concat([Class_True,Class_Probability],axis=1)
	return Class_Info

def reassign_guild(input_id_name,input_X_array,input_guild_array):
	# there might have label at trophic mode, but it could display no label at guild mode
	# so the new form of unknown label will rebuild slightly
	# it could show incosistent shape after summing together
	X_df = pd.DataFrame(input_X_array)
	selected_index = X_df[X_df.iloc[:,0].isin(input_id_name)].index
	remain_id_X_array = input_X_array[selected_index,:]
	remain_id_guild_array = input_guild_array[selected_index,:]
	a = remain_id_guild_array.sum(axis=1)
	remain_id_na_index_Info={"guild":np.where(a==0)[0]}
	return remain_id_X_array, remain_id_guild_array, remain_id_na_index_Info