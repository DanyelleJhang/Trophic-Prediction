#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from time import time
import sys, os, warnings, argparse
call_module_for_bash = os.path.dirname(str(os.getcwd())) + "/Python"
print(call_module_for_bash)
sys.path.insert(1, call_module_for_bash)
import ML_Algorithms.Unsupervised_Reduction as UR

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(
	description= "Code to:\n" +
	"Feature Production")

req_group = parser.add_argument_group(title='REQUIRED INPUT')
req_group.add_argument('-FUNGuild_Info_xlsx',required=True)
req_group.add_argument('-ProteinFamily_dir',required=True)
req_group.add_argument('-Save_CountTable_dir',required=True)
req_group.add_argument('-Remain_Proportion',required=True)
req_group.add_argument('-Tag',required=True)
opt_group = parser.add_argument_group(title='OPTIONAL INPUT')
opt_group.add_argument('-Save_FragmentedProteinFamily_dir',default="None")
if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(0)
args = parser.parse_args()

#import Feature_Production.Label_Encond as le
FUNGuild_info = pd.read_excel(args.FUNGuild_Info_xlsx)
protein_family_dir = args.ProteinFamily_dir
output_dir = args.Save_CountTable_dir
Gene_Remain_Proportion = float(args.Remain_Proportion)
Tag = args.Tag
fragmented_output_dir = args.Save_FragmentedProteinFamily_dir
if fragmented_output_dir == "None":
	fragmented_output_dir = output_dir

class Feature_Table: 
	def __init__(self,FUNGuild_excel,protein_family_directory):
		from collections import Counter
		from itertools import chain
		FUNGuild_protein_file_list = list(FUNGuild_excel["protein_file_name"])
		dir_list = list(map(lambda x: protein_family_directory+x+".out.format",FUNGuild_protein_file_list))
		# Build Count Tabel
		genome_dict = {}
		data_list = []
		for one_genome_file in dir_list:
			r_0 = pd.read_csv(one_genome_file,sep="\t",header=None,usecols=[1],names=["domain_location"])
			if Gene_Remain_Proportion != 1.0:
				#print("origin row size:",len(r_0))
				gene_remain_proportion = Gene_Remain_Proportion
				drop_index_array = np.random.choice(len(r_0), size = round(gene_remain_proportion*len(r_0)), replace=False)
				r = r_0.drop(list(drop_index_array)).reset_index(drop=True)
				if not os.path.exists(fragmented_output_dir+"Fragmented_Peptide_domain/"):
					os.makedirs(fragmented_output_dir+"Fragmented_Peptide_domain/")
				one_genome_file_name= one_genome_file.split("/")[-1]
				r.to_csv(fragmented_output_dir+"Fragmented_Peptide_domain/"+one_genome_file_name+".____."+str(Gene_Remain_Proportion)+".____."+Tag+".txt",sep="\t",header=False,index=False)
				#print("after droping row size:",len(r))
			else:
				r = r_0
			origin_genome_file_name = one_genome_file.split("/")[-1].replace(".out.format","")
			r_dropnohit_list = list(r[~r["domain_location"].isin(["no_hit"])]["domain_location"])
			one_genome_domain_list_unflat = list(map(lambda x: x.split(" "),r_dropnohit_list))
			one_genome_domain_list_raw = [list(map(lambda x: x.split(":")[0],i)) for i in one_genome_domain_list_unflat]
			one_genome_domain_list = list(chain(*one_genome_domain_list_raw))
			domain_freq_count_dict = dict(Counter(one_genome_domain_list))
			# domain_freq_count_dict["Calculation_Characteristics"] = "Raw_Count" ##### Reduncdance , Unnecessary
			domain_freq_count_dict["genome_file_name"] = origin_genome_file_name
			one_genome_domain_row = pd.DataFrame([domain_freq_count_dict])
			# Append Important Information
			genome_dict[origin_genome_file_name]=[one_genome_domain_list_raw]
			# Append Raw Count Table
			data_list.append(one_genome_domain_row)
		table = pd.concat(data_list, ignore_index=True, sort=False).fillna(0)
		protein_family_list = list(table.columns)
		#protein_family_list.remove("Calculation_Characteristics")
		protein_family_list.remove("genome_file_name")
		self.genome_raw_info = pd.DataFrame(genome_dict)
		self.genome_raw_count = table
		self.entire_domain_family = protein_family_list
		# Calculation_Characteristics	file_name
	def count_by_gene(self,one_genome):
		l = []
		for one_gene in one_genome:
			all_domain_family_indice = {k:i for i, k in enumerate(self.entire_domain_family) if k in one_gene}
			one_gene_domain_family_indice= [all_domain_family_indice[e] for e in one_gene]
			one_gene_count = np.zeros(len(self.entire_domain_family), dtype=int)
			np.add.at(one_gene_count, one_gene_domain_family_indice, 1)
			l.append(one_gene_count)
		return np.column_stack(l)

if not os.path.exists(output_dir+"Feature_Production/"):
	os.makedirs(output_dir+"Feature_Production/")
s = time()
FT = Feature_Table(FUNGuild_info,protein_family_dir)
#Genome_Raw_Info = FT.genome_raw_info
# entire_domain_family_list = FT.entire_domain_family
#all_genome_file_name = list(Genome_Raw_Info.columns)

print("start to build Raw Table")
Raw_count_table = FT.genome_raw_count
Raw_count_table[['genome_file_name']] = Raw_count_table[['genome_file_name']]+"--"+str(Gene_Remain_Proportion)+"--"+Tag
Raw_count_table_2 = Raw_count_table[["genome_file_name"]+list(Raw_count_table.drop(columns=["genome_file_name"]).sum(axis=0).sort_values(axis=0, ascending=False).index)]
Raw_count_table_2.to_csv(output_dir+"Feature_Production/"+"Raw_Count_Table"+".Gene_Remain_Proportion_"+str(Gene_Remain_Proportion)+"."+Tag+".txt",sep="\t",header=True,index=False)

with open(output_dir+"Feature_Production/"+"domain_frequncy_rank."+str(Gene_Remain_Proportion)+"."+Tag+".txt","w") as file:
	file.write("# High Frequency to Low Frequency; To to Down ..\n")
	r_col= Raw_count_table_2.drop(columns=["genome_file_name"]).columns
	for i in list(r_col):
		file.write("%s\n"%i)

print("Done!!!")
e = time()
print("time cost: ",str((e-s)/60)," mins")
## Data Augmentation
#error_list = []
#for i in all_genome_file_name:
#	print("start to build decomposed table by ",i)
#	individual_gene_table = FT.count_by_gene(Genome_Raw_Info[i][0])
#	individual_gene_decomposed_table, error_algorithms_list = FT.gene_decomposed_table(individual_gene_table)
#	individual_gene_decomposed_table_2 = individual_gene_decomposed_table.assign(genome_file_name=[i]*len(individual_gene_decomposed_table))
#	Individual_Gene_Decomposed_Table= individual_gene_decomposed_table_2[['genome_file_name','Calculation_Characteristics']+entire_domain_family_list]
#	Individual_Gene_Decomposed_Table.to_csv(output_dir+"Feature_Production/"+i+"_Count_Docomposed_Table.txt",sep="\t",header=True,index=False)
#	del Individual_Gene_Decomposed_Table
#	error_array = np.column_stack([[i]*len(error_algorithms_list),error_algorithms_list])
#	error_list.append(error_array)
#	print("==================================RUN ",i," Done==================================")
#Error_DataFrame = pd.DataFrame(np.concatenate(error_list),columns=["Error_Genome_File","error_Algorithms"])
#Error_DataFrame.to_csv(output_dir+"Feature_Production/"+i+"Error_Data.txt",sep="\t",header=True,index=False)