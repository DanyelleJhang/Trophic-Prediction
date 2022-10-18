#!/bin/bash
guild_summary_dir="/data/home/kaijyun/1_Data/Feature_Preprocessing/Guild_data/FUNGuild_with_pep_filename_merge_label_encoder.xlsx"
##
defult_path='/data/home/kaijyun/1_Data/Feature_Preprocessing/'
protein_dir='Protein_data/'
##
domain_table_dir='Domain_and_Peptide/'
##

python_code='/data/home/kaijyun/2_Code/Descriptive_Statistics_code/domain_freq_table_v2.py'
dos2unix ${python_code}

cd ${defult_path}${protein_dir}${domain_table_dir}

suffix=".fa.out.format_info.txt"

for f in *.fa.out.format_info.txt;
do
	{
		file_name_prefix=${f%"$suffix"} ## remove suffix string
		## Try 
		echo "${file_name_prefix}"
		python ${python_code} \
		-FUNGuild_downsized_summary_xlsx  ${guild_summary_dir} \
		-Domain_Data ${defult_path}${protein_dir}${domain_table_dir}${file_name_prefix}.fa.out.format_info.txt
	} || { ## Catch 
		echo "${file_name_prefix}.fa.out.format_info.txt" > /data/home/kaijyun/3_Shell/domain_freq_table_MISS.txt
		set -e ## error but continued; keep diving in ''set --help''
	}
done

