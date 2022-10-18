user_dir="/home/kaijyun/"
dos2unix "${user_dir}Code/Python/FS_trophicMode.py"
dos2unix "${user_dir}Code/Python/FS_guildMode.py"
mkdir "${user_dir}Data/ML_Prediction/"

#declare -a top_rank=(100 150 200)
#declare -a round_num=(2 3 4 5 6 7 8 9 10)
#declare -a round_num=(1)


tag="round_${num}"
# trophicMode
python -u "${user_dir}Code/Python/FS_trophicMode.py" \
-Data_Directory "${user_dir}Data/Feature_Production/" \
-Feature_Data_Name "Genome_Combination_Count_Table.forTest3.txt" \
-Save_Path "${user_dir}Data/ML_Prediction/" \
-Tag "test"