#!/bin/bash
## remember to revise direcory ##
defult_path='/data/home/kaijyun/1_Data/Feature_Preprocessing/'
protein_dir='Protein_data/'
##
seq_dir='_Protein_seq/'
##
python_code='/data/home/kaijyun/2_Code/Feature_Preprocessing_code/1_protein_fasta_to_full_length_table.py'
cd ${defult_path}${protein_dir}${seq_dir}
for f in *.fa;
do
	{
		#file_name_prefix=${f%"$suffix"} ## remove suffix string
		## Try 
		#echo "${file_name_prefix}"
		python ${python_code} \
		-protein_origin_fasta ${defult_path}${protein_dir}${seq_dir}${f}
	} || { ## Catch 
		echo "${f}" >> ${defult_path}fasta_to_table_MISS.txt
		set -e ## error but continued; keep diving in ''set --help''
	}
done