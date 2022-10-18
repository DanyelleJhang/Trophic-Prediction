#!/bin/bash
python /data/home/kaijyun/Code/CountTable_to_UnsupervisedDR_v2.py\
    -FUNGuild_Info /data/home/kaijyun/Data/Feature_Preprocessing/Guild_data/FUNGuild_with_pep_filename_merge_downsize.xlsx\
		-Domain_Directory /data/home/kaijyun/Data/Feature_Preprocessing/Protein_data/_Peptide_domain/\
    -Label_Name guild\
		-Dimension_Reduction_method MiniBatchDictionaryLearning\
		-N_Components 3