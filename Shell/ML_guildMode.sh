user_dir="./"
dos2unix "${user_dir}Code/Python/ML_guildMode.py"
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

		python -u "${user_dir}Code/Python/ML_guildMode.py" \
		-Data_Directory "${user_dir}Data/Feature_Production/" \
		-Feature_Data_Name ${feature_data_name} \
		-Complete_Data_Name ${complete_data_name} \
		-Feature_Selection_Data_Name "${feature_data_name}--guild--feature_selection_list--${tag}.txt" \
		-Instance_Selection_Data_Name "${feature_data_name}--trophicMode--partial_instance_name--Supervised--${tag}.txt" \
		-Save_Path "${user_dir}Data/ML_Prediction/" \
		-Learning_Method "Supervised" \
		-Tag ${tag}

		python -u "${user_dir}Code/Python/ML_guildMode.py" \
		-Data_Directory "${user_dir}Data/Feature_Production/" \
		-Feature_Data_Name ${feature_data_name} \
		-Complete_Data_Name ${complete_data_name} \
		-Feature_Selection_Data_Name "${feature_data_name}--guild--feature_selection_list--${tag}.txt" \
		-Instance_Selection_Data_Name "${feature_data_name}--trophicMode--partial_instance_name--Supervised--${tag}.txt" \
		-Save_Path "${user_dir}Data/ML_Prediction/" \
		-Learning_Method "SelfTraining" \
		-Tag ${tag}
	done
done



# feature_data_name="Genome_Combination_Count_Table.forTest3.txt"
# complete_data_name="Genome_Combination_Count_Table.domain_frequncy_rank.Concatenation.100.complete.txt"
# tag="2022_10_1_2"

# python -u -Wignore "${user_dir}Code/Python/ML_guildMode.py" \
# -Data_Directory "${user_dir}Data/Feature_Production/" \
# -Feature_Data_Name ${feature_data_name} \
# -Complete_Data_Name ${complete_data_name} \
# -Feature_Selection_Data_Name "${feature_data_name}--guild--feature_selection_list--${tag}.txt" \
# -Instance_Selection_Data_Name "${feature_data_name}--trophicMode--partial_instance_name--Supervised--${tag}.txt" \
# -Save_Path "${user_dir}Data/ML_Prediction/" \
# -Learning_Method "Supervised" \
# -Tag ${tag} \
# -Algorithm "['DecisionTreeClassifier','LogisticRegression']" \
# -Model_Iterations 3
