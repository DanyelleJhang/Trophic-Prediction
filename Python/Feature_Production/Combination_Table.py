#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from glob import glob
from collections import Counter
from itertools import combinations
from time import time
import sys, os, warnings, argparse
call_module_for_bash = os.path.dirname(str(os.getcwd())) + "/Python"
sys.path.insert(1, call_module_for_bash)
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(
	description= "Code to:\n" +
	"Combination Feature Production Table")

req_group = parser.add_argument_group(title='REQUIRED INPUT')
req_group.add_argument('-ProteinFamilyList',required=True)
req_group.add_argument('-Save_CombanationTable_dir',required=True)
opt_group = parser.add_argument_group(title='OPTIONAL INPUT')
opt_group.add_argument('-ProteinFamily_dir',default="None")
opt_group.add_argument('-FragmentedProteinFamily_dir',default="None")
opt_group.add_argument('-Combinations_Count',default="2")
if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(0)
args = parser.parse_args()


if args.FragmentedProteinFamily_dir != "None":
	main_file_dir = args.FragmentedProteinFamily_dir 
	file_type = "fragment"
elif args.ProteinFamily_dir != "None":
	main_file_dir = args.ProteinFamily_dir
	file_type = "complete"
else:
	print("Require only one of following parameter -FragmentedProteinFamily_dir or -ProteinFamily_dir !!") 
protein_domain_list = args.ProteinFamilyList
output_dir = args.Save_CombanationTable_dir
combinations_kount = int(args.Combinations_Count)


def over_one(n):
	return len(n)> 1

class Combination_Table: 
	def __init__(self,input_directory,input_file_type,input_protein_domain_file_path,input_combinations_kount):
		dir_list = glob(input_directory+"*.out.format")
		with open(input_protein_domain_file_path) as f:
			protein_domain_list = [line.split("\n")[0] for line in f.readlines()]
		data_list = []
		for one_genome_file in dir_list:
			if input_file_type == "fragment":
				r = pd.read_csv(one_genome_file,header=None,usecols=[0],names=["domain_location"])
				origin_genome_file_name = one_genome_file.split("/")[-1].split(".____.")[0].replace(".out.format","")
				remain_proportion = one_genome_file.split("/")[-1].split(".____.")[1]
				round_type = one_genome_file.split("/")[-1].split(".____.")[2].split(".")[0]
			elif input_file_type == "complete":
				r = pd.read_csv(one_genome_file,sep="\t",header=None,usecols=[1],names=["domain_location"])
				origin_genome_file_name = one_genome_file.split("/")[-1].replace(".out.format","")
				# _candida_auris_gca_002759435.Cand_auris_B8441_V2.pep.all.fa.out
			else:
				print("parameter require to be fragment or complete")
			r_dropnohit_list = list(r[~r["domain_location"].isin(["no_hit"])]["domain_location"])
			one_genome_domain_list_unflat = list(map(lambda x: x.split(" "),r_dropnohit_list))
			one_genome_domain_list_raw = [list(map(lambda x: x.split(":")[0],i)) for i in one_genome_domain_list_unflat]
			################################
			# 目前都計算單基因上的DOMAIN組合 #
			# 並沒有計算跨基因上的DOMAIN組合 #
			################################
			one_genome_domain_list_rm_duplicate = list(map(lambda x: list(set(x)),one_genome_domain_list_raw)) # remove duplicated domain in one gene
			one_genome_domain_list_multiple_domain = list(filter(over_one,one_genome_domain_list_rm_duplicate))
			one_genome_domain_list_multiple_domain = list(map(lambda x: combinations(x,input_combinations_kount),one_genome_domain_list_multiple_domain))
			#
			one_genome_combinations_domain_list = []
			for i in one_genome_domain_list_multiple_domain:
				for ii in i:
					one_genome_combinations_domain_list.append("&&".join(ii))
			domain_freq_count_dict = Counter(one_genome_combinations_domain_list)

			protein_domain_comba_list = ["&&".join(i) for i in combinations(protein_domain_list,input_combinations_kount)]

			protein_domain_comba_count_dict = {}
			for protein_domain_comba in protein_domain_comba_list:
				if protein_domain_comba in domain_freq_count_dict.keys():
					protein_domain_comba_count_dict[protein_domain_comba] = domain_freq_count_dict[protein_domain_comba]
				else:
					protein_domain_comba_count_dict[protein_domain_comba] = 0
			if input_file_type == "fragment":
				protein_domain_comba_count_dict["genome_file_name"] = origin_genome_file_name +"--"+str(remain_proportion)+"--"+str(round_type)
			elif input_file_type == "complete":
				protein_domain_comba_count_dict["genome_file_name"] = origin_genome_file_name
			one_genome_domain_row = pd.DataFrame([protein_domain_comba_count_dict])
			data_list.append(one_genome_domain_row)
		table = pd.concat(data_list, ignore_index=True, sort=False).fillna(0)
		feature_list = list(table.columns)
		feature_list.remove("genome_file_name")
		self.combination_count = table
		self.entire_feature = feature_list

s = time()
print("start to build combination table ...")
Genome_Combination_Table = Combination_Table(main_file_dir,file_type,protein_domain_list,combinations_kount)
print("Success!!")
Genome_Combination_count_table = Genome_Combination_Table.combination_count
sored_column_name_list= list(Genome_Combination_count_table.drop(columns=["genome_file_name"]).sum(axis=0).sort_values(axis=0, ascending=False).index)
Genome_Combination_count_table = Genome_Combination_count_table[["genome_file_name"]+sored_column_name_list]

if not os.path.exists(output_dir+"Feature_Production/"):
	os.makedirs(output_dir+"Feature_Production/")

if args.FragmentedProteinFamily_dir != "None":
	Genome_Combination_count_table.to_csv(output_dir+"Feature_Production/"+"Genome_Combination_Count_Table."+protein_domain_list.split("/")[-1].split(".txt")[0]+".fragment.txt",chunksize=10,sep="\t",header=True,index=False)
elif args.ProteinFamily_dir != "None":
	Genome_Combination_count_table.to_csv(output_dir+"Feature_Production/"+"Genome_Combination_Count_Table."+protein_domain_list.split("/")[-1].split(".txt")[0]+".complete.txt",chunksize=10,sep="\t",header=True,index=False)
else:
	print("SAVING ERROR ....")
print("Saved!!")
e = time()
print("time cost: ",str((e-s)/60)," mins")