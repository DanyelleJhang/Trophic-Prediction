#!/bin/bash

for L in {trophicMode,guild,trait,growthForm};
do

	python /data/home/kaijyun/Code/Feature_Preprocessing_code/category_feature_encoding.py \
	-Peptide_Frequency_dir /data/home/kaijyun/Data/Feature_Preprocessing/Protein_data/Peptide_Frequency/domain/ \
	-freq_type domain \
	-Target_label ${L} \
	-Category_column [domain_family]
done;
for L in {trophicMode,guild,trait,growthForm};
do
	python /data/home/kaijyun/Code/Feature_Preprocessing_code/category_feature_encoding.py \
	-Peptide_Frequency_dir /data/home/kaijyun/Data/Feature_Preprocessing/Protein_data/Peptide_Frequency/FL/ \
	-freq_type FL \
	-Target_label ${L} \
	-Category_column [domain_family]
done;