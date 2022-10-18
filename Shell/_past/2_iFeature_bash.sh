cd /data/home/kaijyun/Feature_Production/Data_input/
input_array=(AAC CKSAAP DPC DDE TPC GAAC CKSAAGP GDPC GTPC NMBroto Moran Geary CTDC CTDT CTDD CTriad KSCTriad SOCNumber QSOrder PAAC APAAC)
for i in "${input_array[@]}";
do
	for f in *.fasta; 
	do
		if [ -f "${f}.${i}.txt" ]; then
			echo "${f}.${i}.txt exists."
		else 
			echo "${f}"
			echo "===================================="
			python /data/home/kaijyun/Feature_Production/iFeature/iFeature.py --type "${i}" --file "${f}" --out "/data/home/kaijyun/Feature_Production/Data_out/${f}.${i}.txt"
			echo "===================================="
		fi 
	done
done