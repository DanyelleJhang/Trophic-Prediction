ALL_TYPES=("type1" "type2" "type3A" "type3B" "type4" "type5" "type6A" "type6B" "type6C" "type7" "type8" "type9" "type10" "type11" "type12" "type13" "type14" "type15" "type16")

declare A TYPES
TYPES[type1]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
TYPES[type2]='2 3 4 5 6 8 15 20'
TYPES[type3A]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
TYPES[type3B]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
TYPES[type4]='5 8 9 11 13 20'
TYPES[type5]='3 4 8 10 15 20'
TYPES[type6A]='4 5 20'
TYPES[type6B]='5'
TYPES[type6C]='5'
TYPES[type7]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
TYPES[type8]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
TYPES[type9]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
TYPES[type10]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
TYPES[type11]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
TYPES[type12]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20'
TYPES[type13]='4 12 17 20'
TYPES[type14]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
TYPES[type15]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 20'
TYPES[type16]='2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 20'


subtype_array=('g-gap' 'lambda-correlation')
ktuple_array=(1 2 3)
gap_lambda_array=(0 1 2 3 4 5 6 7 8 9)

cd /data/home/kaijyun/Feature_Production/Data_input/
for t in "${ALL_TYPES[@]}";
do
	raactype_array=(${TYPES[t]})
	for r in "${raactype_array[@]}"; 
	do 
		for g in "${gap_lambda_array[@]}"; 
		do
			for k in "${ktuple_array[@]}";
			do
				for s in "${subtype_array[@]}";
				do
					for f in *.fasta; 
					do
						if [ -f "${f}.fasta" ]; then
							echo "${f}.fasta exists."
						else 
							echo "${f}"
							echo "===================================="
							python /data/home/kaijyun/Feature_Production/iFeature/iFeaturePseKRAAC.py --type ${t} --subtype ${s} --ktuple ${k} --gap_lambda ${g} --raactype ${r} \
							--file "${f}" \
							--out "/data/home/kaijyun/Feature_Production/Data_out/PseKRAAC/${f}.PseKRAAC_type.${t}_subtype.${s}_ktuple.${k}_gap_lambda.${g}_raactype.${r}.txt"
							echo "===================================="
						fi 
					done
				done	
			done	
		done	
	done
done