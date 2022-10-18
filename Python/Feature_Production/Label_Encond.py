#!/usr/bin/env python
# coding: utf-8

###### Main Label
# trophicMode
# guild
# trait
# growthForm

###### Other Feature
# confidenceRanking

###### Data Concatenation Key
# protein_file_name ==> genome_file_name

import numpy as np
import pandas as pd
import sys, os, warnings, argparse
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(
	description= "Code to:\n" +
	"Lable Encoding .....")

req_group = parser.add_argument_group(title='REQUIRED INPUT')
req_group.add_argument('-FUNGuild_Info_xlsx',required=True)
req_group.add_argument('-Output_directory',required=True)
if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(0)
args = parser.parse_args()


FUNGuild_info = pd.read_excel(args.FUNGuild_Info_xlsx,
							  usecols=["protein_file_name","confidenceRanking","trophicMode","guild","trait","growthForm"])
output_dir = args.Output_directory

FUNGuild_info["trait"] = FUNGuild_info["trait"].replace({'Soft Rot fungus (Nilsson 1973)':'Soft Rot'})
FUNGuild_info["growthForm"] = FUNGuild_info["growthForm"].replace({'Microfungus; Facultative Yeast (Tedersoo et al. 2014)':'Facultative Yeast-Microfungus'})
FUNGuild_info = FUNGuild_info.rename(columns={"protein_file_name":"genome_file_name"})


####### Label Coding Methods (HCL prefer)
#|             | Flag remain NA |
#| ----------: | -------------- |
#|          na | 0              |
#|  Pathotroph | 1              |
#|  Saprotroph | 2              |
#| Symbiotroph | 4              |
####### Coding Example
#|                                   | Flag remain NA | Multi-Class remain NA | Multi-Label remain NA (Pathotroph, Saprotroph, Symbiotroph) | 
#| --------------------------------- | -------------- |--------------------- | ----------------------------------------------------------- |
#| na                                | 0              | 0                     | (0,0,0)                                               |                   
#| Pathotroph                        | 1              | 1                     | (1,0,0)                                                     |                   
#| Pathotroph-Symbiotroph            | 1+4=5          | 2                     | (1,0,1)                                                     |                   
#| Pathotroph-Saprotroph             | 1+2=3          | 3                     | (1,1,0)                                                     |                   
#| Pathotroph-Saprotroph-Symbiotroph | 1+2+4=7        | 4                     | (1,1,1)                                                     |                   
#| Saprotroph                        | 2              | 5                     | (0,1,0)                                                     |                   
#| Saprotroph-Symbiotroph            | 2+4=6          | 6                     | (0,1,1)                                                     |                  
#| Symbiotroph                       | 4              | 7                     | (0,0,1)                                                     |                   



def transform_flag_label(data,label,NA_as_Zero):
	original_lable_list = list(set(data[label]))
	original_lable_list_sep = list(map(lambda x: x.split("-"),original_lable_list))
	########### Main Lable
	# trophicMode
	# guild
	# trait
	# growthForm
	if label == "trophicMode":
		flag_NA = {'na':0,
					 'Pathotroph':pow(2,0),
					 'Saprotroph':pow(2,1),
					 'Symbiotroph':pow(2,2)
				  }
	if label == "guild":
		flag_NA={'na':0,
				 # Pathotroph
				 'Animal Pathogen':pow(2,0),
				 'Bryophyte Parasite':pow(2,1),
				 'Clavicipitaceous Endophyte':pow(2,2),
				 'Fungal Parasite':pow(2,4),
				 'Plant Pathogen':pow(2,5),
				 'Lichen Parasite':pow(2,6),
				 'Insect Pathogen':pow(2,7),
				# Saprotroph
				 'Dung Saprotroph':pow(2,8),
				 'Leaf Saprotroph':pow(2,9),
				 'Plant Saprotroph':pow(2,10),
				 'Soil Saprotroph':pow(2,11),
				 'Undefined Saprotroph':pow(2,12),
				 'Wood Saprotroph':pow(2,13),
				 'Litter Saprotroph':pow(2,14),
				# Symbiotroph
				 'Ectomycorrhizal':pow(2,15),
				 'Ericoid Mycorrhizal':pow(2,16),
				 'Arbuscular Mycorrhizal':pow(2,17),
				 'Orchid Mycorrhizal':pow(2,18),
				 'Endophyte':pow(2,19),
				 'Epiphyte':pow(2,20),
				 'Lichenized':pow(2,21),
				 'Animal Endosymbiont':pow(2,22)
				}
	if label == "trait":
		###### trait
		flag_NA = {'na':0,
				   'Blue':pow(2,0),
				   'Brown Rot':pow(2,1),
				   'Hypogeous':pow(2,2),
				   'Poisonous':pow(2,3),
				   'Soft Rot':pow(2,4),
				   'Staining':pow(2,5),
				   'White Rot':pow(2,6),
				   'Soft Rot fungus (Nilsson 1973)':pow(2,4)
				  }
	if label == "growthForm":
		flag_NA={'na':0,
				 'Agaricoid':pow(2,0),
				 'Boletoid':pow(2,1),
				 'Clavarioid':pow(2,2),
				 'Corticioid':pow(2,3),
				 'Dimorphic Yeast':pow(2,4),
				 'Gasteroid':pow(2,5),
				 'Polyporoid':pow(2,6),
				 'Rust':pow(2,7),
				 'Secotioid':pow(2,8),
				 'Smut':pow(2,9),
				 'Thallus':pow(2,10),
				 'Tremelloid':pow(2,11),
				 'Yeast':pow(2,12),
				 'Microfungus':pow(2,13),
				 'Facultative Yeast':pow(2,14),
				 'Microfungus; Facultative Yeast (Tedersoo et al. 2014)':pow(2,13)+pow(2,14)
				}
	if label not in ["trophicMode","guild","trait","growthForm"]:
		print("please input trophicMode, guild, trait, growthForm")
	y = []
	for i in original_lable_list_sep:
		x =[]
		for o_i in i:
			x.append(flag_NA[o_i])
		y.append(sum(x))
	encode = {}
	for i,ii in enumerate(original_lable_list):
		encode[ii]=y[i]

	NA_as_zero = NA_as_Zero.capitalize()
	if eval(NA_as_zero) == bool(1):
		return encode
	else:
		encode.pop('na', None)
		return encode

def transform_multi_label(data,label,NA_as_Zero):
	from itertools import chain
	original_lable_list = list(set(data[label]))

	original_lable_list_sep = list(map(lambda x: x.split("-"),original_lable_list))
	original_lable_type_list = sorted(list(set(list(chain(*original_lable_list_sep)))))
	# To ensure 'na' will be a full zero vector
	original_lable_type_list.remove('na')
	# 'na' vector will be like [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	lable_encode_raw = [[1 if i_in in i_out else 0 for i_in in original_lable_type_list] for i_out in original_lable_list_sep]
	original_lable_list_undo = list(map(lambda x: "-".join(x),original_lable_list_sep))
	encode={}
	for i,k in enumerate(original_lable_list_undo):
		encode[k]=str(lable_encode_raw[i])
	encode["Lable_Type"]=str(original_lable_type_list)
	NA_as_zero = NA_as_Zero.capitalize()
	if eval(NA_as_zero) == bool(1):
		return encode
	else:
		encode.pop('na', None)
		return encode

def transform_ordinal_label(data,label,NA_as_Zero):
	original_lable_list = list(sorted(list(set(data[label]))))
	original_lable_list.remove("na")
	encode = {}
	for i,k in enumerate(original_lable_list):
		encode[k]=i+1
	encode["na"] = 0
	NA_as_zero = NA_as_Zero.capitalize()
	if eval(NA_as_zero) == bool(1):
		return encode
	else:
		enocode.pop('na', None)
		return encode

def transform_confidenceRanking():
	# 'Highly Probable':99.7
	# 'Possible':(99.7+95)/2 == 97.35
	# 'Probable':95
	encode = {'Highly Probable':99.7,'Possible':97.35,'Probable':95, 'na':0}
	return encode



def Flag_Label_DataFrame(data,label,NA_as_Zero):
	Flag_Label = transform_flag_label(data,label,NA_as_Zero)
	empty_DataFrame = pd.DataFrame()
	empty_DataFrame[label]= data[label].replace(Flag_Label)
	Label_note = pd.DataFrame({"Names":list(Flag_Label.keys()),"Transform":list(Flag_Label.values())})
	DataFrame = pd.concat([data[["genome_file_name","confidenceRanking"]],empty_DataFrame],axis=1)
	DataFrame[["confidenceRanking"]] = DataFrame[["confidenceRanking"]].replace(transform_confidenceRanking())
	return DataFrame,Label_note



def Ordinal_Label_DataFrame(data,label,NA_as_Zero):
	Ordinal_Label = transform_ordinal_label(data,label,NA_as_Zero)
	empty_DataFrame = pd.DataFrame()
	empty_DataFrame[label]= data[label].replace(Ordinal_Label)
	Label_note = pd.DataFrame({"Names":list(Ordinal_Label.keys()),"Transform":list(Ordinal_Label.values())})
	DataFrame = pd.concat([data[["genome_file_name","confidenceRanking"]],empty_DataFrame],axis=1)
	DataFrame[["confidenceRanking"]] = DataFrame[["confidenceRanking"]].replace(transform_confidenceRanking())
	return DataFrame,Label_note



def Multi_Label_DataFrame(data,label,NA_as_Zero):
	Multi_Label = transform_multi_label(data,label,NA_as_Zero)
	empty_DataFrame = pd.DataFrame()
	empty_DataFrame[Multi_Label["Lable_Type"]]= data[label].replace(Multi_Label)
	Label_note = pd.DataFrame({"Names":list(Multi_Label.keys()),"Transform":list(Multi_Label.values())})
	DataFrame = pd.concat([data[["genome_file_name","confidenceRanking"]],empty_DataFrame],axis=1)
	DataFrame[["confidenceRanking"]] = DataFrame[["confidenceRanking"]].replace(transform_confidenceRanking())
	return DataFrame,Label_note




if not os.path.exists(output_dir+"Feature_Production/"):
	os.makedirs(output_dir+"Feature_Production/")


for L in ["trophicMode","guild","trait","growthForm"]:
	FLD,FLD_note = Flag_Label_DataFrame(FUNGuild_info,L,"true")
	OLD,OLD_note = Ordinal_Label_DataFrame(FUNGuild_info,L,"true")
	MLD,MLD_note = Multi_Label_DataFrame(FUNGuild_info,L,"true")
	FLD.to_csv(output_dir+"Feature_Production/"+L+"_Flag_Label.txt",sep="\t",header=True,index=False)
	OLD.to_csv(output_dir+"Feature_Production/"+L+"_Ordinal_Label.txt",sep="\t",header=True,index=False)
	MLD.to_csv(output_dir+"Feature_Production/"+L+"_Multi_Label.txt",sep="\t",header=True,index=False)
	FLD_note.to_csv(output_dir+"Feature_Production/"+L+"_Flag_Label_note.txt",sep="\t",header=True,index=False)
	OLD_note.to_csv(output_dir+"Feature_Production/"+L+"_Ordinal_Label_note.txt",sep="\t",header=True,index=False)
	MLD_note.to_csv(output_dir+"Feature_Production/"+L+"_Multi_Label_note.txt",sep="\t",header=True,index=False)