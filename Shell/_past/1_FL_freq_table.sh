#!/bin/bash
guild_summary_dir="/data/home/kaijyun/1_Data/Feature_Preprocessing/Guild_data/FUNGuild_with_pep_filename_merge_label_encoder.xlsx"
##
defult_path='/data/home/kaijyun/1_Data/Feature_Preprocessing/'
protein_dir='Protein_data/'
##
FL_table_dir='Protein_Full_Length/'
##

python_code='/data/home/kaijyun/2_Code/Descriptive_Statistics_code/FL_freq_table_v2.py'
dos2unix ${python_code}

cd ${defult_path}${protein_dir}${FL_table_dir}

suffix=".fafull_length_table.txt"

for f in *.fafull_length_table.txt;
do
	{
		file_name_prefix=${f%"$suffix"} ## remove suffix string
		## Try 
		echo "${file_name_prefix}"
		python ${python_code} \
		-FUNGuild_downsized_summary_xlsx  ${guild_summary_dir} \
		-Protein_full_length_table ${defult_path}${protein_dir}${FL_table_dir}${f}
	} || { ## Catch 
		echo "${file_name_prefix}.fafull_length_table.txt" > /data/home/kaijyun/3_Shell/FL_freq_table_MISS.txt
		set -e ## error but continued; keep diving in ''set --help''
	}
done

