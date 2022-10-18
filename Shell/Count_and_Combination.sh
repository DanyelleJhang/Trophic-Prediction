code_dir="/home/kaijyun/Code/"
data_dir="/home/kaijyun/Data/"
protein_data_dir="/home/kaijyun/Data/Feature_Preprocessing/Protein_data/"

dos2unix "${code_dir}Python/Feature_Production/Count_Table.py"
dos2unix "${code_dir}Python/Feature_Production/Feature_Concatenation.py"
dos2unix "${code_dir}Python/Feature_Production/Combination_Table.py"

# Produce Single Domain Complete Raw Count
python "${code_dir}Python/Feature_Production/Count_Table.py" \
-FUNGuild_Info_xlsx "${data_dir}Feature_Preprocessing/Guild_data/FUNGuild_with_pep_filename_merge_downsize.xlsx" \
-ProteinFamily_dir "${protein_data_dir}_Peptide_domain/" \
-Save_CountTable_dir "${data_dir}" \
-Remain_Proportion "1" \
-Tag "complete"



declare -a round_num=(2 3 4 5 6 7 8 9 10)
declare -a top_rank=(100 150 200)
declare -a comba_num=(2)
declare -a drop_proportion=(0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5)
for num in "${round_num[@]}"
do
	# Produce Single Domain Fragmented Raw Count
	# $(seq 1 1 30)
	for i in $(seq 1 1 1)
	do
		for drop in "${drop_proportion[@]}";
		do
			python "${code_dir}Python/Feature_Production/Count_Table.py" \
			-FUNGuild_Info_xlsx "${data_dir}Feature_Preprocessing/Guild_data/FUNGuild_with_pep_filename_merge_downsize.xlsx" \
			-ProteinFamily_dir "${protein_data_dir}_Peptide_domain/" \
			-Save_CountTable_dir "${data_dir}" \
			-Remain_Proportion "${drop}" \
			-Tag "round_${i}" \
			-Save_FragmentedProteinFamily_dir ${protein_data_dir}
		done
	done

	# Concatenate all Single Domain Fragmented Raw Count to one dataframe
	python "${code_dir}Python/Feature_Production/Feature_Concatenation.py" \
	-Data_Directory "${data_dir}Feature_Production/" \
	-Complete_Feature_Data Raw_Count_Table.Gene_Remain_Proportion_1.0.complete.txt \
	-Save_Path "${data_dir}Feature_Production/"
	echo "Concatenation Done"

	# Get domain name which belongs to top rank of high domain frequency
	for rank in "${top_rank[@]}"
	do
		tail "${data_dir}Feature_Production/domain_frequncy_rank.Concatenation.txt" -n +2 | head -n "${rank}" > "${data_dir}Feature_Production/domain_frequncy_rank.Concatenation.${rank}.txt"
	done

	echo "Start to run combination of complete data"
	# Produce Domain Combination Count Table based on top rank of domain frequncy
	# complete data
	for rank in "${top_rank[@]}"
	do
		for i in "${comba_num[@]}"
		do
			python "${code_dir}Python/Feature_Production/Combination_Table.py" \
			-ProteinFamilyList "${data_dir}Feature_Production/domain_frequncy_rank.Concatenation.${rank}.txt" \
			-Save_CombanationTable_dir "${data_dir}" \
			-ProteinFamily_dir "${protein_data_dir}_Peptide_domain/" \
			-Combinations_Count "${i}"
		done
	done

	echo "Start to run combination of fragment data"
	# Produce Domain Combination Count Table based on top rank of domain frequncy
	# fragment data
	for rank in "${top_rank[@]}"
	do
		for i in "${comba_num[@]}"
		do
			python "${code_dir}Python/Feature_Production/Combination_Table.py" \
			-ProteinFamilyList "${data_dir}Feature_Production/domain_frequncy_rank.Concatenation.${rank}.txt" \
			-Save_CombanationTable_dir "${data_dir}" \
			-FragmentedProteinFamily_dir "${protein_data_dir}Fragmented_Peptide_domain/" \
			-Combinations_Count "${i}"
		done
	done
	
	for rank in "${top_rank[@]}" 
	do
		cat "${data_dir}Feature_Production/Genome_Combination_Count_Table.domain_frequncy_rank.Concatenation.${rank}.fragment.txt" > "${data_dir}Feature_Production/Genome_Combination_Count_Table.domain_frequncy_rank.Concatenation.${rank}.round_${num}.fragment.txt"
		rm "${data_dir}Feature_Production/Genome_Combination_Count_Table.domain_frequncy_rank.Concatenation.${rank}.fragment.txt"
		cat "${data_dir}Feature_Production/Genome_Combination_Count_Table.domain_frequncy_rank.Concatenation.${rank}.complete.txt" > "${data_dir}Feature_Production/Genome_Combination_Count_Table.domain_frequncy_rank.Concatenation.${rank}.round_${num}.complete.txt"
		rm "${data_dir}Feature_Production/Genome_Combination_Count_Table.domain_frequncy_rank.Concatenation.${rank}.complete.txt"
	done
done





