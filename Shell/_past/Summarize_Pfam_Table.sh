#!/bin/bash
## remember to revise direcory ##
defult_path='/data/home/kaijyun/Data/Feature_Preprocessing/'
protein_dir='Protein_data/'
##
domain_dir='_Peptide_domain/'
seq_dir='_Protein_seq/'
##
python_code='/data/home/kaijyun/Code/Feature_Preprocessing_code/summarize_pfam_table.py'

# search domain information as priority
# not protein full length sequence
# there will cause tremendous negative effect

cd ${defult_path}${protein_dir}${domain_dir}

suffix=".fa.out.format"

for f in *.fa.out.format;
do
	{
		file_name_prefix=${f%"$suffix"} ## remove suffix string
		## Try 
		echo "${file_name_prefix}"
		python ${python_code} \
	-protein_domain ${defult_path}${protein_dir}${domain_dir}${file_name_prefix}.fa.out.format \
	-protein_full_len ${defult_path}${protein_dir}${seq_dir}${file_name_prefix}.fa \
	-output_format pandas_table \
	-drop_no_hit True
	} || { ## Catch 
		echo "${f}.fa.out.format" >> ${defult_path}summarize_pfam_table_MISS.txt
		set -e ## error but continued; keep diving in ''set --help''
	}
done


