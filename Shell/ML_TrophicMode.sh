user_dir="./"
dos2unix "${user_dir}Code/Python/ML_trophicMode.py"
mkdir "${user_dir}Data/ML_Prediction/"

declare -a top_rank=(100 150 200)
#declare -a round_num=(2 3 4 5 6 7 8 9 10)
declare -a round_num=(1 2 3 4 5 6 7 8 9 10)
for rank in "${top_rank[@]}"
do
	for num in "${round_num[@]}" 
	do
		feature_data_name="Genome_Combination_Count_Table.domain_frequncy_rank.Concatenation.${rank}.fragment.txt"
		complete_data_name="Genome_Combination_Count_Table.domain_frequncy_rank.Concatenation.${rank}.complete.txt"
		tag="round_${num}"

		python -u "${user_dir}Code/Python/ML_trophicMode.py" \
		-Data_Directory "${user_dir}Data/Feature_Production/" \
		-Feature_Data_Name ${feature_data_name} \
		-Complete_Data_Name ${complete_data_name} \
		-Feature_Selection_Data_Name "${feature_data_name}--trophicMode--feature_selection_list--${tag}.txt" \
		-Save_Path "${user_dir}Data/ML_Prediction/" \
		-Learning_Method "Supervised" \
		-Tag ${tag}

		python -u "${user_dir}Code/Python/ML_trophicMode.py" \
		-Data_Directory "${user_dir}Data/Feature_Production/" \
		-Feature_Data_Name ${feature_data_name} \
		-Complete_Data_Name ${complete_data_name} \
		-Feature_Selection_Data_Name "${feature_data_name}--trophicMode--feature_selection_list--${tag}.txt" \
		-Save_Path "${user_dir}Data/ML_Prediction/" \
		-Learning_Method "SelfTraining" \
		-Tag ${tag}
	done
done