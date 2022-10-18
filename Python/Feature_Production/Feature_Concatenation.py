#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import glob
import numpy as np
import os, warnings,argparse,sys
parser = argparse.ArgumentParser(
	description= "Code to:\n" +
	"Feature Concatenation")

req_group = parser.add_argument_group(title='REQUIRED INPUT')
req_group.add_argument('-Data_Directory',required=True)
req_group.add_argument('-Complete_Feature_Data',required=True)
req_group.add_argument('-Save_Path',required=True)
opt_group = parser.add_argument_group(title='OPTIONAL INPUT')
opt_group.add_argument('-Random_Select_Domain_Number',default="None",help="100 ,400 , 1000, ")
opt_group.add_argument('-Sampling_Frequency',default="10")

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(0)
args = parser.parse_args()

feature_data_dir = args.Data_Directory
Complete_Feature_Data = args.Complete_Feature_Data
Save_Path = args.Save_Path
random_sample_domain_num = eval(args.Random_Select_Domain_Number)
sampling_iter= eval(args.Sampling_Frequency)

complete_feature_data= feature_data_dir+Complete_Feature_Data
with open(complete_feature_data) as f:
	main_feature_name = f.readline().rstrip().split("\t")
# example: 
# Raw_Count_Table.Gene_Remain_Proportion_0.8.round_30.txt
gene_remain_proportion_entire_data_list = list(filter(lambda x: ("Raw_Count_Table" in x)&("Gene_Remain_Proportion" in x)&("round_" in x)&(".txt" in x)&("random_domain" not in x)&("iter" not in x)&("Concatenation" not in x)&("complete" not in x),glob.glob(feature_data_dir+"*")))

r = []
for i in gene_remain_proportion_entire_data_list:
	r_i = pd.read_csv(i,sep="\t")
	r.append(r_i)
gene_remain_proportion_entire_data= pd.concat(r,axis=0).reset_index(drop=True).fillna(0)
unremain_features_name = list(set(main_feature_name)- set(list(gene_remain_proportion_entire_data.columns)))
gene_remain_proportion_entire_data_drop = gene_remain_proportion_entire_data.drop_duplicates().reset_index(drop=True)
unremain_features_zero_data= pd.DataFrame(np.zeros((len(gene_remain_proportion_entire_data_drop), len(unremain_features_name))), columns=unremain_features_name)
concat_dataframe = pd.concat([gene_remain_proportion_entire_data_drop,unremain_features_zero_data],axis=1)

if random_sample_domain_num == None:
	domain_dataframe = concat_dataframe.drop(columns=['genome_file_name'])
	infomation_dataframe = concat_dataframe[['genome_file_name']]
	del concat_dataframe
	domain_dataframe_SortedByPrevelance = list(domain_dataframe.sum(axis=0).sort_values(ascending=False).index)
	domain_dataframe = domain_dataframe[domain_dataframe_SortedByPrevelance]
	finale_dataframe = pd.concat([infomation_dataframe,domain_dataframe],axis=1)
	sored_column_name_list= list(finale_dataframe.drop(columns=["genome_file_name"]).sum(axis=0).sort_values(axis=0, ascending=False).index)
	finale_dataframe = finale_dataframe[["genome_file_name"]+sored_column_name_list]
	finale_dataframe.to_csv(Save_Path+"Raw_Count_Table.Gene_Remain_Proportion.Concatenation.txt",header=True,sep="\t",index=False)
	with open(Save_Path+"domain_frequncy_rank.Concatenation.txt","w") as file:
		file.write("# High Frequency to Low Frequency; To to Down ..\n")
		r_col= finale_dataframe.drop(columns=["genome_file_name"]).columns
		for i in list(r_col):
			file.write("%s\n"%i)
else:
	domain_dataframe = concat_dataframe.drop(columns=['genome_file_name'])
	infomation_dataframe = concat_dataframe[['genome_file_name']]
	del concat_dataframe
	for i in range(sampling_iter):
		sampled_domain_dataframe = domain_dataframe.sample(n=random_sample_domain_num, axis='columns', random_state=i)
		sampled_domain_dataframe_SortedByPrevelance = list(sampled_domain_dataframe.sum(axis=0).sort_values(ascending=False).index)
		sampled_domain_dataframe = sampled_domain_dataframe[sampled_domain_dataframe_SortedByPrevelance]
		finale_dataframe = pd.concat([infomation_dataframe,sampled_domain_dataframe],axis=1)
		finale_dataframe.to_csv(Save_Path+"Raw_Count_Table.Gene_Remain_Proportion.Concatenation.random_domain--"+str(random_sample_domain_num)+".iter--"+str(i)+".txt",header=True,sep="\t",index=False)